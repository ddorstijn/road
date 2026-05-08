use engine::car_renderer::{CarRenderPushConstants, CarRenderer};
use engine::gpu_resources::GpuBuffer;
use engine::pipeline::{
    GraphicsPipelineDesc, allocate_descriptor_set, compile_wgsl, create_compute_pipeline,
    create_descriptor_set_layout, create_graphics_pipeline,
};
use engine::sdf::{
    ATLAS_TILES, RoadSegmentInfo, SdfTileManager, TILE_RESOLUTION, TILE_SIZE, compute_segment_aabbs,
};
use engine::vk::{DeviceV1_0, DeviceV1_3, Handle, HasBuilder};
use engine::{App, EngineContext, transition_image, vk};
use glam::Vec2;
use rand::Rng;
use road::fitting::{ControlPoint, ReferenceLine};
use road::network::Road;
use road::network::RoadNetwork;

const GRID_SHADER: &str = include_str!("../../../assets/shaders/grid.wgsl");
const ROAD_LINE_SHADER: &str = include_str!("../../../assets/shaders/road_line.wgsl");
const SDF_TYPES_WGSL: &str = include_str!("../../../assets/shaders/shared/types.wgsl");
const SDF_ROAD_EVAL_WGSL: &str = include_str!("../../../assets/shaders/shared/road_eval.wgsl");
const SDF_GENERATE_WGSL: &str = include_str!("../../../assets/shaders/sdf_generate.wgsl");
const ROAD_RENDER_WGSL: &str = include_str!("../../../assets/shaders/road_render.wgsl");
const TRAFFIC_UPDATE_WGSL: &str = include_str!("../../../assets/shaders/traffic_update.wgsl");
const CAR_RENDER_WGSL: &str = include_str!("../../../assets/shaders/car_render.wgsl");
const TRAFFIC_SORT_KEYS_WGSL: &str = include_str!("../../../assets/shaders/traffic_sort_keys.wgsl");
const TRAFFIC_SORT_HISTOGRAM_WGSL: &str =
    include_str!("../../../assets/shaders/traffic_sort_histogram.wgsl");
const TRAFFIC_SORT_SCAN_WGSL: &str = include_str!("../../../assets/shaders/traffic_sort_scan.wgsl");
const TRAFFIC_SORT_SCATTER_WGSL: &str =
    include_str!("../../../assets/shaders/traffic_sort_scatter.wgsl");
const TRAFFIC_IDM_WGSL: &str = include_str!("../../../assets/shaders/traffic_idm.wgsl");
const TRAFFIC_LANE_CHANGE_WGSL: &str =
    include_str!("../../../assets/shaders/traffic_lane_change.wgsl");

// ---------------------------------------------------------------------------
// GPU Timestamp profiler
// ---------------------------------------------------------------------------

/// Timestamp query indices
const TS_FRAME_BEGIN: u32 = 0;
const TS_SDF_END: u32 = 1;
const TS_GRID_END: u32 = 2;
const TS_TRAFFIC_END: u32 = 3;
const TS_RENDER_END: u32 = 4;
const TS_COUNT: u32 = 5;

struct GpuTimestamps {
    query_pool: vk::QueryPool,
    timestamp_period_ns: f32,
    // Accumulated results (ms) — smoothed over multiple frames
    sdf_ms: f32,
    grid_ms: f32,
    traffic_ms: f32,
    render_ms: f32,
    total_ms: f32,
}

impl GpuTimestamps {
    fn new(device: &engine::VkDevice, timestamp_period: f32) -> anyhow::Result<Self> {
        let pool_info = vk::QueryPoolCreateInfo::builder()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(TS_COUNT);
        let query_pool = unsafe { device.create_query_pool(&pool_info, None)? };
        Ok(Self {
            query_pool,
            timestamp_period_ns: timestamp_period,
            sdf_ms: 0.0,
            grid_ms: 0.0,
            traffic_ms: 0.0,
            render_ms: 0.0,
            total_ms: 0.0,
        })
    }

    fn read_results(&mut self, device: &engine::VkDevice) {
        let mut timestamps = [0u64; TS_COUNT as usize];
        let result = unsafe {
            let data = std::slice::from_raw_parts_mut(
                timestamps.as_mut_ptr() as *mut u8,
                std::mem::size_of_val(&timestamps),
            );
            device.get_query_pool_results(
                self.query_pool,
                0,
                TS_COUNT,
                data,
                std::mem::size_of::<u64>() as u64,
                vk::QueryResultFlags::_64,
            )
        };
        if result.is_ok() {
            let to_ms = |t: u64| -> f32 { t as f32 * self.timestamp_period_ns / 1_000_000.0 };
            let sdf = to_ms(
                timestamps[TS_SDF_END as usize].wrapping_sub(timestamps[TS_FRAME_BEGIN as usize]),
            );
            let grid = to_ms(
                timestamps[TS_GRID_END as usize].wrapping_sub(timestamps[TS_SDF_END as usize]),
            );
            let traffic = to_ms(
                timestamps[TS_TRAFFIC_END as usize].wrapping_sub(timestamps[TS_GRID_END as usize]),
            );
            let render = to_ms(
                timestamps[TS_RENDER_END as usize]
                    .wrapping_sub(timestamps[TS_TRAFFIC_END as usize]),
            );
            let total = to_ms(
                timestamps[TS_RENDER_END as usize]
                    .wrapping_sub(timestamps[TS_FRAME_BEGIN as usize]),
            );

            // Exponential moving average (α = 0.1)
            let a = 0.1f32;
            self.sdf_ms = self.sdf_ms * (1.0 - a) + sdf * a;
            self.grid_ms = self.grid_ms * (1.0 - a) + grid * a;
            self.traffic_ms = self.traffic_ms * (1.0 - a) + traffic * a;
            self.render_ms = self.render_ms * (1.0 - a) + render * a;
            self.total_ms = self.total_ms * (1.0 - a) + total * a;
        }
    }

    fn reset_and_begin(&self, device: &engine::VkDevice, cmd: vk::CommandBuffer) {
        unsafe {
            device.cmd_reset_query_pool(cmd, self.query_pool, 0, TS_COUNT);
            device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                self.query_pool,
                TS_FRAME_BEGIN,
            );
        }
    }

    fn write(&self, device: &engine::VkDevice, cmd: vk::CommandBuffer, index: u32) {
        unsafe {
            device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                self.query_pool,
                index,
            );
        }
    }

    fn destroy(&self, device: &engine::VkDevice) {
        unsafe {
            device.destroy_query_pool(self.query_pool, None);
        }
    }
}

// ---------------------------------------------------------------------------
// GPU data structs (SSBOs for later phases)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSegment {
    segment_type: u32,
    s_start: f32,
    origin: [f32; 2],
    heading: f32,
    length: f32,
    k_start: f32,
    k_end: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuRoad {
    segment_offset: u32,
    segment_count: u32,
    lane_section_offset: u32,
    lane_section_count: u32,
    total_length: f32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuLaneSection {
    s_start: f32,
    s_end: f32,
    lane_offset: u32,
    lane_count: u32,
    left_lane_count: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuLane {
    width: f32,
    lane_type: u32,
    marking_type: u32,
    _pad: u32,
}

// ---------------------------------------------------------------------------
// Vertex for polyline rendering
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LineVertex {
    position: [f32; 2],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraParams {
    inv_vp: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LinePushConstants {
    view_proj: [[f32; 4]; 4],
}

// ---------------------------------------------------------------------------
// Draw range for polyline rendering
// ---------------------------------------------------------------------------

struct DrawRange {
    first_vertex: u32,
    vertex_count: u32,
}

// ---------------------------------------------------------------------------
// Traffic simulation constants & push constants
// ---------------------------------------------------------------------------

/// Maximum number of cars (power of 2, > 500k)
const MAX_CARS: u32 = 524_288;

/// Fixed simulation timestep (1/60 s)
const SIM_DT: f32 = 1.0 / 60.0;

/// Sort workgroup tile size (elements per workgroup)
const SORT_TILE_SIZE: u32 = 256;

/// Number of radix sort workgroups
const NUM_SORT_WG: u32 = MAX_CARS / SORT_TILE_SIZE; // 2048

/// MOBIL lane change frequency (every N simulation ticks)
const LANE_CHANGE_INTERVAL: u32 = 30;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TrafficUpdatePushConstants {
    dt: f32,
    car_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SortKeysPushConstants {
    car_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SortHistogramPushConstants {
    pass_id: u32,
    car_count: u32,
    num_workgroups: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SortScanPushConstants {
    num_workgroups: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SortScatterPushConstants {
    pass_id: u32,
    car_count: u32,
    num_workgroups: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct IdmPushConstants {
    dt: f32,
    car_count: u32,
    a_max: f32,
    b_comfort: f32,
    s0: f32,
    time_headway: f32,
    car_length: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MobilPushConstants {
    car_count: u32,
    a_max: f32,
    b_comfort: f32,
    s0: f32,
    time_headway: f32,
    car_length: f32,
    politeness: f32,
    threshold: f32,
    b_safe: f32,
    max_right_lanes: i32,
    max_left_lanes: i32,
    _pad: u32,
}

// ---------------------------------------------------------------------------
// Traffic simulation state
// ---------------------------------------------------------------------------

struct TrafficSim {
    // Car SoA buffers
    car_road_id_buf: Option<GpuBuffer>,
    car_s_buf: Option<GpuBuffer>,
    car_lane_buf: Option<GpuBuffer>,
    car_speed_buf: Option<GpuBuffer>,
    car_desired_speed_buf: Option<GpuBuffer>,
    // Road lengths buffer (one f32 per road, for the compute shader)
    road_lengths_buf: Option<GpuBuffer>,

    // Sort buffers (ping-pong)
    sort_keys_a_buf: Option<GpuBuffer>,
    sort_keys_b_buf: Option<GpuBuffer>,
    sort_vals_a_buf: Option<GpuBuffer>,
    sort_vals_b_buf: Option<GpuBuffer>,
    sort_histogram_buf: Option<GpuBuffer>,

    // Compute pipeline (traffic_update — used as fallback before sort is ready)
    update_pipeline: vk::Pipeline,
    update_pipeline_layout: vk::PipelineLayout,
    update_descriptor_set_layout: vk::DescriptorSetLayout,
    update_descriptor_pool: vk::DescriptorPool,
    update_descriptor_set: vk::DescriptorSet,

    // Sort pipelines
    sort_keys_pipeline: vk::Pipeline,
    sort_keys_pipeline_layout: vk::PipelineLayout,
    sort_keys_descriptor_set_layout: vk::DescriptorSetLayout,
    sort_keys_descriptor_pool: vk::DescriptorPool,
    sort_keys_descriptor_set: vk::DescriptorSet,

    sort_histogram_pipeline: vk::Pipeline,
    sort_histogram_pipeline_layout: vk::PipelineLayout,
    sort_histogram_descriptor_set_layout: vk::DescriptorSetLayout,
    sort_histogram_descriptor_pool: vk::DescriptorPool,
    sort_histogram_descriptor_sets: [vk::DescriptorSet; 2], // [read A, read B]

    sort_scan_pipeline: vk::Pipeline,
    sort_scan_pipeline_layout: vk::PipelineLayout,
    sort_scan_descriptor_set_layout: vk::DescriptorSetLayout,
    sort_scan_descriptor_pool: vk::DescriptorPool,
    sort_scan_descriptor_set: vk::DescriptorSet,

    sort_scatter_pipeline: vk::Pipeline,
    sort_scatter_pipeline_layout: vk::PipelineLayout,
    sort_scatter_descriptor_set_layout: vk::DescriptorSetLayout,
    sort_scatter_descriptor_pool: vk::DescriptorPool,
    sort_scatter_descriptor_sets: [vk::DescriptorSet; 2], // [A→B, B→A]

    // IDM pipeline
    idm_pipeline: vk::Pipeline,
    idm_pipeline_layout: vk::PipelineLayout,
    idm_descriptor_set_layout: vk::DescriptorSetLayout,
    idm_descriptor_pool: vk::DescriptorPool,
    idm_descriptor_set: vk::DescriptorSet,

    // MOBIL pipeline
    mobil_pipeline: vk::Pipeline,
    mobil_pipeline_layout: vk::PipelineLayout,
    mobil_descriptor_set_layout: vk::DescriptorSetLayout,
    mobil_descriptor_pool: vk::DescriptorPool,
    mobil_descriptor_set: vk::DescriptorSet,

    // Car renderer (instanced rendering)
    car_renderer: Option<CarRenderer>,

    // Simulation state
    car_count: u32,
    sim_accumulator: f32,
    sim_tick: u32,
    initialized: bool,
}

impl Default for TrafficSim {
    fn default() -> Self {
        Self {
            car_road_id_buf: None,
            car_s_buf: None,
            car_lane_buf: None,
            car_speed_buf: None,
            car_desired_speed_buf: None,
            road_lengths_buf: None,

            sort_keys_a_buf: None,
            sort_keys_b_buf: None,
            sort_vals_a_buf: None,
            sort_vals_b_buf: None,
            sort_histogram_buf: None,

            update_pipeline: vk::Pipeline::null(),
            update_pipeline_layout: vk::PipelineLayout::null(),
            update_descriptor_set_layout: vk::DescriptorSetLayout::null(),
            update_descriptor_pool: vk::DescriptorPool::null(),
            update_descriptor_set: vk::DescriptorSet::null(),

            sort_keys_pipeline: vk::Pipeline::null(),
            sort_keys_pipeline_layout: vk::PipelineLayout::null(),
            sort_keys_descriptor_set_layout: vk::DescriptorSetLayout::null(),
            sort_keys_descriptor_pool: vk::DescriptorPool::null(),
            sort_keys_descriptor_set: vk::DescriptorSet::null(),

            sort_histogram_pipeline: vk::Pipeline::null(),
            sort_histogram_pipeline_layout: vk::PipelineLayout::null(),
            sort_histogram_descriptor_set_layout: vk::DescriptorSetLayout::null(),
            sort_histogram_descriptor_pool: vk::DescriptorPool::null(),
            sort_histogram_descriptor_sets: [vk::DescriptorSet::null(); 2],

            sort_scan_pipeline: vk::Pipeline::null(),
            sort_scan_pipeline_layout: vk::PipelineLayout::null(),
            sort_scan_descriptor_set_layout: vk::DescriptorSetLayout::null(),
            sort_scan_descriptor_pool: vk::DescriptorPool::null(),
            sort_scan_descriptor_set: vk::DescriptorSet::null(),

            sort_scatter_pipeline: vk::Pipeline::null(),
            sort_scatter_pipeline_layout: vk::PipelineLayout::null(),
            sort_scatter_descriptor_set_layout: vk::DescriptorSetLayout::null(),
            sort_scatter_descriptor_pool: vk::DescriptorPool::null(),
            sort_scatter_descriptor_sets: [vk::DescriptorSet::null(); 2],

            idm_pipeline: vk::Pipeline::null(),
            idm_pipeline_layout: vk::PipelineLayout::null(),
            idm_descriptor_set_layout: vk::DescriptorSetLayout::null(),
            idm_descriptor_pool: vk::DescriptorPool::null(),
            idm_descriptor_set: vk::DescriptorSet::null(),

            mobil_pipeline: vk::Pipeline::null(),
            mobil_pipeline_layout: vk::PipelineLayout::null(),
            mobil_descriptor_set_layout: vk::DescriptorSetLayout::null(),
            mobil_descriptor_pool: vk::DescriptorPool::null(),
            mobil_descriptor_set: vk::DescriptorSet::null(),

            car_renderer: None,

            car_count: 0,
            sim_accumulator: 0.0,
            sim_tick: 0,
            initialized: false,
        }
    }
}

impl TrafficSim {
    fn create_pipelines(&mut self, ctx: &EngineContext) -> anyhow::Result<()> {
        let device = ctx.device.as_ref();

        // --- Compute pipeline: traffic_update ---
        {
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..6)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::builder()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .build()
                })
                .collect();

            self.update_descriptor_set_layout = create_descriptor_set_layout(device, &bindings)?;

            let pool_sizes = [vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(6)
                .build()];
            let pool_info = vk::DescriptorPoolCreateInfo::builder()
                .max_sets(1)
                .pool_sizes(&pool_sizes);
            self.update_descriptor_pool =
                unsafe { device.create_descriptor_pool(&pool_info, None)? };

            self.update_descriptor_set = allocate_descriptor_set(
                device,
                self.update_descriptor_pool,
                self.update_descriptor_set_layout,
            )?;

            let spirv = compile_wgsl(TRAFFIC_UPDATE_WGSL)?;

            let push_constant_ranges = [vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(std::mem::size_of::<TrafficUpdatePushConstants>() as u32)
                .build()];

            let (pipeline, layout) = create_compute_pipeline(
                device,
                &spirv,
                &[self.update_descriptor_set_layout],
                &push_constant_ranges,
            )?;
            self.update_pipeline = pipeline;
            self.update_pipeline_layout = layout;
        }

        // --- Car renderer (instanced rendering) ---
        {
            let shared_wgsl = format!("{}\n{}", SDF_TYPES_WGSL, SDF_ROAD_EVAL_WGSL);
            self.car_renderer = Some(CarRenderer::new(
                device,
                ctx.draw_image.format,
                &shared_wgsl,
                CAR_RENDER_WGSL,
            )?);
        }

        // --- Sort: build keys pipeline ---
        {
            // Bindings: 0-3 car_road_id/car_s/car_lane/road_lengths (read),
            //           4-5 sort_keys/sort_vals (write)
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..6)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::builder()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .build()
                })
                .collect();

            self.sort_keys_descriptor_set_layout = create_descriptor_set_layout(device, &bindings)?;

            let pool_sizes = [vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(6)
                .build()];
            let pool_info = vk::DescriptorPoolCreateInfo::builder()
                .max_sets(1)
                .pool_sizes(&pool_sizes);
            self.sort_keys_descriptor_pool =
                unsafe { device.create_descriptor_pool(&pool_info, None)? };

            self.sort_keys_descriptor_set = allocate_descriptor_set(
                device,
                self.sort_keys_descriptor_pool,
                self.sort_keys_descriptor_set_layout,
            )?;

            let spirv = compile_wgsl(TRAFFIC_SORT_KEYS_WGSL)?;
            let push_constant_ranges = [vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(std::mem::size_of::<SortKeysPushConstants>() as u32)
                .build()];

            let (pipeline, layout) = create_compute_pipeline(
                device,
                &spirv,
                &[self.sort_keys_descriptor_set_layout],
                &push_constant_ranges,
            )?;
            self.sort_keys_pipeline = pipeline;
            self.sort_keys_pipeline_layout = layout;
        }

        // --- Sort: histogram pipeline ---
        {
            // Bindings: 0 keys_in (read), 1 histograms (read_write)
            let bindings = [
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
            ];

            self.sort_histogram_descriptor_set_layout =
                create_descriptor_set_layout(device, &bindings)?;

            let pool_sizes = [vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(4) // 2 bindings × 2 sets
                .build()];
            let pool_info = vk::DescriptorPoolCreateInfo::builder()
                .max_sets(2)
                .pool_sizes(&pool_sizes);
            self.sort_histogram_descriptor_pool =
                unsafe { device.create_descriptor_pool(&pool_info, None)? };

            for i in 0..2 {
                self.sort_histogram_descriptor_sets[i] = allocate_descriptor_set(
                    device,
                    self.sort_histogram_descriptor_pool,
                    self.sort_histogram_descriptor_set_layout,
                )?;
            }

            let spirv = compile_wgsl(TRAFFIC_SORT_HISTOGRAM_WGSL)?;
            let push_constant_ranges = [vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(std::mem::size_of::<SortHistogramPushConstants>() as u32)
                .build()];

            let (pipeline, layout) = create_compute_pipeline(
                device,
                &spirv,
                &[self.sort_histogram_descriptor_set_layout],
                &push_constant_ranges,
            )?;
            self.sort_histogram_pipeline = pipeline;
            self.sort_histogram_pipeline_layout = layout;
        }

        // --- Sort: prefix sum (scan) pipeline ---
        {
            // Binding: 0 histograms (read_write)
            let bindings = [vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build()];

            self.sort_scan_descriptor_set_layout = create_descriptor_set_layout(device, &bindings)?;

            let pool_sizes = [vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .build()];
            let pool_info = vk::DescriptorPoolCreateInfo::builder()
                .max_sets(1)
                .pool_sizes(&pool_sizes);
            self.sort_scan_descriptor_pool =
                unsafe { device.create_descriptor_pool(&pool_info, None)? };

            self.sort_scan_descriptor_set = allocate_descriptor_set(
                device,
                self.sort_scan_descriptor_pool,
                self.sort_scan_descriptor_set_layout,
            )?;

            let spirv = compile_wgsl(TRAFFIC_SORT_SCAN_WGSL)?;
            let push_constant_ranges = [vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(std::mem::size_of::<SortScanPushConstants>() as u32)
                .build()];

            let (pipeline, layout) = create_compute_pipeline(
                device,
                &spirv,
                &[self.sort_scan_descriptor_set_layout],
                &push_constant_ranges,
            )?;
            self.sort_scan_pipeline = pipeline;
            self.sort_scan_pipeline_layout = layout;
        }

        // --- Sort: scatter pipeline ---
        {
            // Bindings: 0 keys_in, 1 vals_in, 2 keys_out, 3 vals_out, 4 histograms
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..5)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::builder()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .build()
                })
                .collect();

            self.sort_scatter_descriptor_set_layout =
                create_descriptor_set_layout(device, &bindings)?;

            let pool_sizes = [vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(10) // 5 bindings × 2 sets
                .build()];
            let pool_info = vk::DescriptorPoolCreateInfo::builder()
                .max_sets(2)
                .pool_sizes(&pool_sizes);
            self.sort_scatter_descriptor_pool =
                unsafe { device.create_descriptor_pool(&pool_info, None)? };

            for i in 0..2 {
                self.sort_scatter_descriptor_sets[i] = allocate_descriptor_set(
                    device,
                    self.sort_scatter_descriptor_pool,
                    self.sort_scatter_descriptor_set_layout,
                )?;
            }

            let spirv = compile_wgsl(TRAFFIC_SORT_SCATTER_WGSL)?;
            let push_constant_ranges = [vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(std::mem::size_of::<SortScatterPushConstants>() as u32)
                .build()];

            let (pipeline, layout) = create_compute_pipeline(
                device,
                &spirv,
                &[self.sort_scatter_descriptor_set_layout],
                &push_constant_ranges,
            )?;
            self.sort_scatter_pipeline = pipeline;
            self.sort_scatter_pipeline_layout = layout;
        }

        // --- IDM car-following pipeline ---
        {
            // Bindings: 0-4 car SoA, 5 road_lengths, 6 sorted_indices
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..7)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::builder()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .build()
                })
                .collect();

            self.idm_descriptor_set_layout = create_descriptor_set_layout(device, &bindings)?;

            let pool_sizes = [vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(7)
                .build()];
            let pool_info = vk::DescriptorPoolCreateInfo::builder()
                .max_sets(1)
                .pool_sizes(&pool_sizes);
            self.idm_descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

            self.idm_descriptor_set = allocate_descriptor_set(
                device,
                self.idm_descriptor_pool,
                self.idm_descriptor_set_layout,
            )?;

            let spirv = compile_wgsl(TRAFFIC_IDM_WGSL)?;
            let push_constant_ranges = [vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(std::mem::size_of::<IdmPushConstants>() as u32)
                .build()];

            let (pipeline, layout) = create_compute_pipeline(
                device,
                &spirv,
                &[self.idm_descriptor_set_layout],
                &push_constant_ranges,
            )?;
            self.idm_pipeline = pipeline;
            self.idm_pipeline_layout = layout;
        }

        // --- MOBIL lane change pipeline ---
        {
            // Bindings: 0-4 car SoA, 5 road_lengths, 6 sorted_indices
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..7)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::builder()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .build()
                })
                .collect();

            self.mobil_descriptor_set_layout = create_descriptor_set_layout(device, &bindings)?;

            let pool_sizes = [vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(7)
                .build()];
            let pool_info = vk::DescriptorPoolCreateInfo::builder()
                .max_sets(1)
                .pool_sizes(&pool_sizes);
            self.mobil_descriptor_pool =
                unsafe { device.create_descriptor_pool(&pool_info, None)? };

            self.mobil_descriptor_set = allocate_descriptor_set(
                device,
                self.mobil_descriptor_pool,
                self.mobil_descriptor_set_layout,
            )?;

            let spirv = compile_wgsl(TRAFFIC_LANE_CHANGE_WGSL)?;
            let push_constant_ranges = [vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(std::mem::size_of::<MobilPushConstants>() as u32)
                .build()];

            let (pipeline, layout) = create_compute_pipeline(
                device,
                &spirv,
                &[self.mobil_descriptor_set_layout],
                &push_constant_ranges,
            )?;
            self.mobil_pipeline = pipeline;
            self.mobil_pipeline_layout = layout;
        }

        Ok(())
    }

    /// Allocate car SoA buffers and initialize with random cars on the given roads.
    fn initialize_cars(
        &mut self,
        allocator: &engine::vma::Allocator,
        network: &RoadNetwork,
        cars_per_road: u32,
    ) -> anyhow::Result<()> {
        self.destroy_buffers(allocator);

        if network.roads.is_empty() {
            self.car_count = 0;
            self.initialized = false;
            return Ok(());
        }

        let num_roads = network.roads.len() as u32;
        let total_cars = (num_roads * cars_per_road).min(MAX_CARS);
        self.car_count = total_cars;

        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let n = total_cars as usize;

        // Allocate buffers
        let (road_id_buf, road_id_ptr) = GpuBuffer::new_mapped(allocator, (n * 4) as u64, usage)?;
        let (s_buf, s_ptr) = GpuBuffer::new_mapped(allocator, (n * 4) as u64, usage)?;
        let (lane_buf, lane_ptr) = GpuBuffer::new_mapped(allocator, (n * 4) as u64, usage)?;
        let (speed_buf, speed_ptr) = GpuBuffer::new_mapped(allocator, (n * 4) as u64, usage)?;
        let (desired_buf, desired_ptr) = GpuBuffer::new_mapped(allocator, (n * 4) as u64, usage)?;

        // Fill with random car data
        let mut rng = rand::rng();
        let road_ids = unsafe { std::slice::from_raw_parts_mut(road_id_ptr as *mut u32, n) };
        let s_vals = unsafe { std::slice::from_raw_parts_mut(s_ptr as *mut f32, n) };
        let lane_vals = unsafe { std::slice::from_raw_parts_mut(lane_ptr as *mut i32, n) };
        let speed_vals = unsafe { std::slice::from_raw_parts_mut(speed_ptr as *mut f32, n) };
        let desired_vals = unsafe { std::slice::from_raw_parts_mut(desired_ptr as *mut f32, n) };

        for i in 0..n {
            let road_idx = (i as u32) % num_roads;
            let road = &network.roads[road_idx as usize];
            let road_len = road.reference_line.total_length;

            // Count available lanes (right = 0..N-1, left = -1..-M)
            let (right_lane_count, left_lane_count) = if road.lane_sections.is_empty() {
                (1i32, 0i32)
            } else {
                (
                    road.lane_sections[0].right_lanes.len() as i32,
                    road.lane_sections[0].left_lanes.len() as i32,
                )
            };

            road_ids[i] = road_idx;
            s_vals[i] = rng.random::<f32>() * road_len;

            // Assign to either right lanes (0..right_count-1) or left lanes (-1..-left_count)
            let total_lanes = right_lane_count + left_lane_count;
            if total_lanes > 0 {
                let pick = rng.random_range(0..total_lanes);
                if pick < right_lane_count {
                    lane_vals[i] = pick; // Right lane
                } else {
                    lane_vals[i] = -(pick - right_lane_count + 1); // Left lane: -1, -2, ...
                }
            } else {
                lane_vals[i] = 0;
            }

            // Desired speed: mean 30 m/s, std 5 m/s (clamped to [15, 45])
            let desired_speed = (30.0 + rng.random_range(-10.0f32..10.0)).clamp(15.0, 45.0);
            desired_vals[i] = desired_speed;
            speed_vals[i] = desired_speed * rng.random_range(0.7f32..1.0);
        }

        // Road lengths buffer
        let road_count = network.roads.len();
        let (lengths_buf, lengths_ptr) =
            GpuBuffer::new_mapped(allocator, (road_count * 4) as u64, usage)?;
        let lengths =
            unsafe { std::slice::from_raw_parts_mut(lengths_ptr as *mut f32, road_count) };
        for (i, road) in network.roads.iter().enumerate() {
            lengths[i] = road.reference_line.total_length;
        }

        self.car_road_id_buf = Some(road_id_buf);
        self.car_s_buf = Some(s_buf);
        self.car_lane_buf = Some(lane_buf);
        self.car_speed_buf = Some(speed_buf);
        self.car_desired_speed_buf = Some(desired_buf);
        self.road_lengths_buf = Some(lengths_buf);

        // Allocate sort buffers (always MAX_CARS to avoid reallocation)
        let sort_buf_size = (MAX_CARS as u64) * 4;
        let (keys_a, _) = GpuBuffer::new_mapped(allocator, sort_buf_size, usage)?;
        let (keys_b, _) = GpuBuffer::new_mapped(allocator, sort_buf_size, usage)?;
        let (vals_a, _) = GpuBuffer::new_mapped(allocator, sort_buf_size, usage)?;
        let (vals_b, _) = GpuBuffer::new_mapped(allocator, sort_buf_size, usage)?;

        // Histogram buffer: 256 digits × NUM_SORT_WG workgroups × 4 bytes
        let hist_size = (256u64) * (NUM_SORT_WG as u64) * 4;
        let hist_usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let (hist_buf, _) = GpuBuffer::new_mapped(allocator, hist_size, hist_usage)?;

        self.sort_keys_a_buf = Some(keys_a);
        self.sort_keys_b_buf = Some(keys_b);
        self.sort_vals_a_buf = Some(vals_a);
        self.sort_vals_b_buf = Some(vals_b);
        self.sort_histogram_buf = Some(hist_buf);

        self.sim_tick = 0;
        self.initialized = true;
        Ok(())
    }

    /// Update descriptor sets to point at current buffers.
    fn update_descriptors(
        &mut self,
        device: &engine::VkDevice,
        segment_buffer: Option<&GpuBuffer>,
        road_buffer: Option<&GpuBuffer>,
        lane_section_buffer: Option<&GpuBuffer>,
        lane_buffer: Option<&GpuBuffer>,
    ) {
        if !self.initialized {
            return;
        }

        let car_road_id = match &self.car_road_id_buf {
            Some(b) => b,
            None => return,
        };
        let car_s = match &self.car_s_buf {
            Some(b) => b,
            None => return,
        };
        let car_lane = match &self.car_lane_buf {
            Some(b) => b,
            None => return,
        };
        let car_speed = match &self.car_speed_buf {
            Some(b) => b,
            None => return,
        };
        let car_desired = match &self.car_desired_speed_buf {
            Some(b) => b,
            None => return,
        };
        let road_lengths = match &self.road_lengths_buf {
            Some(b) => b,
            None => return,
        };

        // Update compute descriptor set (bindings 0-5: car buffers + road_lengths)
        {
            let bufs = [
                car_road_id,
                car_s,
                car_lane,
                car_speed,
                car_desired,
                road_lengths,
            ];
            let buf_infos: Vec<[vk::DescriptorBufferInfo; 1]> = bufs
                .iter()
                .map(|b| {
                    [vk::DescriptorBufferInfo::builder()
                        .buffer(b.buffer)
                        .offset(0)
                        .range(b.size)
                        .build()]
                })
                .collect();

            let writes: Vec<vk::WriteDescriptorSet> = buf_infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::builder()
                        .dst_set(self.update_descriptor_set)
                        .dst_binding(i as u32)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                        .build()
                })
                .collect();

            unsafe {
                device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
            }
        }

        // Update car renderer descriptors (bindings 0-4: car buffers, 5-8: road data)
        if let (Some(seg_buf), Some(rd_buf), Some(ls_buf), Some(ln_buf)) = (
            segment_buffer,
            road_buffer,
            lane_section_buffer,
            lane_buffer,
        ) {
            if let Some(renderer) = &mut self.car_renderer {
                renderer.update_descriptors(
                    device,
                    car_road_id,
                    car_s,
                    car_lane,
                    car_speed,
                    car_desired,
                    seg_buf,
                    rd_buf,
                    ls_buf,
                    ln_buf,
                );
            }
        }

        // --- Update sort key build descriptors ---
        // Bindings: 0 car_road_id, 1 car_s, 2 car_lane, 3 road_lengths, 4 sort_keys_a, 5 sort_vals_a
        let sort_keys_a = match &self.sort_keys_a_buf {
            Some(b) => b,
            None => return,
        };
        let sort_keys_b = match &self.sort_keys_b_buf {
            Some(b) => b,
            None => return,
        };
        let sort_vals_a = match &self.sort_vals_a_buf {
            Some(b) => b,
            None => return,
        };
        let sort_vals_b = match &self.sort_vals_b_buf {
            Some(b) => b,
            None => return,
        };
        let hist_buf = match &self.sort_histogram_buf {
            Some(b) => b,
            None => return,
        };

        {
            let bufs = [
                car_road_id,
                car_s,
                car_lane,
                road_lengths,
                sort_keys_a,
                sort_vals_a,
            ];
            let buf_infos: Vec<[vk::DescriptorBufferInfo; 1]> = bufs
                .iter()
                .map(|b| {
                    [vk::DescriptorBufferInfo::builder()
                        .buffer(b.buffer)
                        .offset(0)
                        .range(b.size)
                        .build()]
                })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = buf_infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::builder()
                        .dst_set(self.sort_keys_descriptor_set)
                        .dst_binding(i as u32)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                        .build()
                })
                .collect();
            unsafe {
                device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
            }
        }

        // --- Update sort histogram descriptors (2 sets: read A, read B) ---
        {
            // Set 0: keys_in = A, histograms
            let bufs_a = [sort_keys_a, hist_buf];
            let buf_infos_a: Vec<[vk::DescriptorBufferInfo; 1]> = bufs_a
                .iter()
                .map(|b| {
                    [vk::DescriptorBufferInfo::builder()
                        .buffer(b.buffer)
                        .offset(0)
                        .range(b.size)
                        .build()]
                })
                .collect();
            let writes_a: Vec<vk::WriteDescriptorSet> = buf_infos_a
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::builder()
                        .dst_set(self.sort_histogram_descriptor_sets[0])
                        .dst_binding(i as u32)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                        .build()
                })
                .collect();
            unsafe {
                device.update_descriptor_sets(&writes_a, &[] as &[vk::CopyDescriptorSet]);
            }

            // Set 1: keys_in = B, histograms
            let bufs_b = [sort_keys_b, hist_buf];
            let buf_infos_b: Vec<[vk::DescriptorBufferInfo; 1]> = bufs_b
                .iter()
                .map(|b| {
                    [vk::DescriptorBufferInfo::builder()
                        .buffer(b.buffer)
                        .offset(0)
                        .range(b.size)
                        .build()]
                })
                .collect();
            let writes_b: Vec<vk::WriteDescriptorSet> = buf_infos_b
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::builder()
                        .dst_set(self.sort_histogram_descriptor_sets[1])
                        .dst_binding(i as u32)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                        .build()
                })
                .collect();
            unsafe {
                device.update_descriptor_sets(&writes_b, &[] as &[vk::CopyDescriptorSet]);
            }
        }

        // --- Update sort scan descriptor ---
        {
            let buf_info = [vk::DescriptorBufferInfo::builder()
                .buffer(hist_buf.buffer)
                .offset(0)
                .range(hist_buf.size)
                .build()];
            let writes = [vk::WriteDescriptorSet::builder()
                .dst_set(self.sort_scan_descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buf_info)
                .build()];
            unsafe {
                device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
            }
        }

        // --- Update sort scatter descriptors (2 sets: A→B, B→A) ---
        {
            // Set 0: read A, write B
            let bufs_0 = [sort_keys_a, sort_vals_a, sort_keys_b, sort_vals_b, hist_buf];
            let buf_infos_0: Vec<[vk::DescriptorBufferInfo; 1]> = bufs_0
                .iter()
                .map(|b| {
                    [vk::DescriptorBufferInfo::builder()
                        .buffer(b.buffer)
                        .offset(0)
                        .range(b.size)
                        .build()]
                })
                .collect();
            let writes_0: Vec<vk::WriteDescriptorSet> = buf_infos_0
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::builder()
                        .dst_set(self.sort_scatter_descriptor_sets[0])
                        .dst_binding(i as u32)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                        .build()
                })
                .collect();
            unsafe {
                device.update_descriptor_sets(&writes_0, &[] as &[vk::CopyDescriptorSet]);
            }

            // Set 1: read B, write A
            let bufs_1 = [sort_keys_b, sort_vals_b, sort_keys_a, sort_vals_a, hist_buf];
            let buf_infos_1: Vec<[vk::DescriptorBufferInfo; 1]> = bufs_1
                .iter()
                .map(|b| {
                    [vk::DescriptorBufferInfo::builder()
                        .buffer(b.buffer)
                        .offset(0)
                        .range(b.size)
                        .build()]
                })
                .collect();
            let writes_1: Vec<vk::WriteDescriptorSet> = buf_infos_1
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::builder()
                        .dst_set(self.sort_scatter_descriptor_sets[1])
                        .dst_binding(i as u32)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                        .build()
                })
                .collect();
            unsafe {
                device.update_descriptor_sets(&writes_1, &[] as &[vk::CopyDescriptorSet]);
            }
        }

        // --- Update IDM descriptor set ---
        // Bindings: 0-4 car SoA, 5 road_lengths, 6 sorted_indices (sort_vals_a after sort)
        {
            let bufs = [
                car_road_id,
                car_s,
                car_lane,
                car_speed,
                car_desired,
                road_lengths,
                sort_vals_a, // After 4 passes: A→B→A→B→A, final sorted indices are in A
            ];
            let buf_infos: Vec<[vk::DescriptorBufferInfo; 1]> = bufs
                .iter()
                .map(|b| {
                    [vk::DescriptorBufferInfo::builder()
                        .buffer(b.buffer)
                        .offset(0)
                        .range(b.size)
                        .build()]
                })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = buf_infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::builder()
                        .dst_set(self.idm_descriptor_set)
                        .dst_binding(i as u32)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                        .build()
                })
                .collect();
            unsafe {
                device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
            }
        }

        // --- Update MOBIL descriptor set ---
        // Same layout as IDM: 0-4 car SoA, 5 road_lengths, 6 sorted_indices
        {
            let bufs = [
                car_road_id,
                car_s,
                car_lane,
                car_speed,
                car_desired,
                road_lengths,
                sort_vals_a,
            ];
            let buf_infos: Vec<[vk::DescriptorBufferInfo; 1]> = bufs
                .iter()
                .map(|b| {
                    [vk::DescriptorBufferInfo::builder()
                        .buffer(b.buffer)
                        .offset(0)
                        .range(b.size)
                        .build()]
                })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = buf_infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::builder()
                        .dst_set(self.mobil_descriptor_set)
                        .dst_binding(i as u32)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                        .build()
                })
                .collect();
            unsafe {
                device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
            }
        }
    }

    fn destroy_buffers(&mut self, allocator: &engine::vma::Allocator) {
        if let Some(b) = self.car_road_id_buf.take() {
            b.destroy(allocator);
        }
        if let Some(b) = self.car_s_buf.take() {
            b.destroy(allocator);
        }
        if let Some(b) = self.car_lane_buf.take() {
            b.destroy(allocator);
        }
        if let Some(b) = self.car_speed_buf.take() {
            b.destroy(allocator);
        }
        if let Some(b) = self.car_desired_speed_buf.take() {
            b.destroy(allocator);
        }
        if let Some(b) = self.road_lengths_buf.take() {
            b.destroy(allocator);
        }
        if let Some(b) = self.sort_keys_a_buf.take() {
            b.destroy(allocator);
        }
        if let Some(b) = self.sort_keys_b_buf.take() {
            b.destroy(allocator);
        }
        if let Some(b) = self.sort_vals_a_buf.take() {
            b.destroy(allocator);
        }
        if let Some(b) = self.sort_vals_b_buf.take() {
            b.destroy(allocator);
        }
        if let Some(b) = self.sort_histogram_buf.take() {
            b.destroy(allocator);
        }
    }

    fn destroy(&mut self, device: &engine::VkDevice, allocator: &engine::vma::Allocator) {
        unsafe {
            device.destroy_pipeline(self.update_pipeline, None);
            device.destroy_pipeline_layout(self.update_pipeline_layout, None);
            device.destroy_descriptor_pool(self.update_descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.update_descriptor_set_layout, None);

            device.destroy_pipeline(self.sort_keys_pipeline, None);
            device.destroy_pipeline_layout(self.sort_keys_pipeline_layout, None);
            device.destroy_descriptor_pool(self.sort_keys_descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.sort_keys_descriptor_set_layout, None);

            device.destroy_pipeline(self.sort_histogram_pipeline, None);
            device.destroy_pipeline_layout(self.sort_histogram_pipeline_layout, None);
            device.destroy_descriptor_pool(self.sort_histogram_descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.sort_histogram_descriptor_set_layout, None);

            device.destroy_pipeline(self.sort_scan_pipeline, None);
            device.destroy_pipeline_layout(self.sort_scan_pipeline_layout, None);
            device.destroy_descriptor_pool(self.sort_scan_descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.sort_scan_descriptor_set_layout, None);

            device.destroy_pipeline(self.sort_scatter_pipeline, None);
            device.destroy_pipeline_layout(self.sort_scatter_pipeline_layout, None);
            device.destroy_descriptor_pool(self.sort_scatter_descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.sort_scatter_descriptor_set_layout, None);

            device.destroy_pipeline(self.idm_pipeline, None);
            device.destroy_pipeline_layout(self.idm_pipeline_layout, None);
            device.destroy_descriptor_pool(self.idm_descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.idm_descriptor_set_layout, None);

            device.destroy_pipeline(self.mobil_pipeline, None);
            device.destroy_pipeline_layout(self.mobil_pipeline_layout, None);
            device.destroy_descriptor_pool(self.mobil_descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.mobil_descriptor_set_layout, None);
        }
        if let Some(renderer) = self.car_renderer.take() {
            renderer.destroy(device);
        }
        self.destroy_buffers(allocator);
    }
}

// ---------------------------------------------------------------------------
// SDF debug visualization push constants
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SdfDebugPushConstants {
    view_proj: [[f32; 4]; 4],
    atlas_uv_offset: [f32; 2],
    atlas_uv_scale: [f32; 2],
    tile_world_origin: [f32; 2],
    tile_world_size: [f32; 2],
}

// ---------------------------------------------------------------------------
// GPU Road Data manager
// ---------------------------------------------------------------------------

struct GpuRoadData {
    segment_buffer: Option<GpuBuffer>,
    road_buffer: Option<GpuBuffer>,
    lane_section_buffer: Option<GpuBuffer>,
    lane_buffer: Option<GpuBuffer>,
}

impl GpuRoadData {
    fn new() -> Self {
        Self {
            segment_buffer: None,
            road_buffer: None,
            lane_section_buffer: None,
            lane_buffer: None,
        }
    }

    fn destroy(&mut self, allocator: &engine::vma::Allocator) {
        if let Some(b) = self.segment_buffer.take() {
            b.destroy(allocator);
        }
        if let Some(b) = self.road_buffer.take() {
            b.destroy(allocator);
        }
        if let Some(b) = self.lane_section_buffer.take() {
            b.destroy(allocator);
        }
        if let Some(b) = self.lane_buffer.take() {
            b.destroy(allocator);
        }
    }

    fn upload(
        &mut self,
        allocator: &engine::vma::Allocator,
        network: &RoadNetwork,
    ) -> anyhow::Result<()> {
        self.destroy(allocator);

        if network.roads.is_empty() {
            return Ok(());
        }

        let mut segments = Vec::new();
        let mut roads = Vec::new();
        let mut lane_sections = Vec::new();
        let mut lanes = Vec::new();

        for road_data in &network.roads {
            let seg_offset = segments.len() as u32;
            let rl = &road_data.reference_line;

            for (i, seg) in rl.segments.iter().enumerate() {
                let (seg_type, k_start, k_end) = match seg {
                    road::primitives::Segment::Line { .. } => (0u32, 0.0f32, 0.0f32),
                    road::primitives::Segment::Arc { curvature, .. } => (1, *curvature, *curvature),
                    road::primitives::Segment::Spiral { k_start, k_end, .. } => {
                        (2, *k_start, *k_end)
                    }
                };

                segments.push(GpuSegment {
                    segment_type: seg_type,
                    s_start: rl.s_offsets[i],
                    origin: [rl.origins[i].x, rl.origins[i].y],
                    heading: rl.headings[i],
                    length: seg.length(),
                    k_start,
                    k_end,
                });
            }

            let ls_offset = lane_sections.len() as u32;
            for section in &road_data.lane_sections {
                let lane_offset = lanes.len() as u32;
                let left_count = section.left_lanes.len() as u32;
                let all_lanes: Vec<_> = section
                    .left_lanes
                    .iter()
                    .chain(section.right_lanes.iter())
                    .collect();

                for lane in &all_lanes {
                    lanes.push(GpuLane {
                        width: lane.width,
                        lane_type: lane.lane_type as u32,
                        marking_type: lane.marking as u32,
                        _pad: 0,
                    });
                }

                lane_sections.push(GpuLaneSection {
                    s_start: section.s_start,
                    s_end: section.s_end,
                    lane_offset,
                    lane_count: all_lanes.len() as u32,
                    left_lane_count: left_count,
                    _pad: 0,
                });
            }

            roads.push(GpuRoad {
                segment_offset: seg_offset,
                segment_count: rl.segments.len() as u32,
                lane_section_offset: ls_offset,
                lane_section_count: road_data.lane_sections.len() as u32,
                total_length: rl.total_length,
                _pad: [0; 3],
            });
        }

        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;

        if !segments.is_empty() {
            let data = bytemuck::cast_slice::<_, u8>(&segments);
            let (buf, ptr) = GpuBuffer::new_mapped(allocator, data.len() as u64, usage)?;
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()) };
            self.segment_buffer = Some(buf);
        }

        if !roads.is_empty() {
            let data = bytemuck::cast_slice::<_, u8>(&roads);
            let (buf, ptr) = GpuBuffer::new_mapped(allocator, data.len() as u64, usage)?;
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()) };
            self.road_buffer = Some(buf);
        }

        if !lane_sections.is_empty() {
            let data = bytemuck::cast_slice::<_, u8>(&lane_sections);
            let (buf, ptr) = GpuBuffer::new_mapped(allocator, data.len() as u64, usage)?;
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()) };
            self.lane_section_buffer = Some(buf);
        }

        if !lanes.is_empty() {
            let data = bytemuck::cast_slice::<_, u8>(&lanes);
            let (buf, ptr) = GpuBuffer::new_mapped(allocator, data.len() as u64, usage)?;
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()) };
            self.lane_buffer = Some(buf);
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Traffic App
// ---------------------------------------------------------------------------

struct TrafficApp {
    // Grid pipeline (compute)
    grid_pipeline: vk::Pipeline,
    grid_pipeline_layout: vk::PipelineLayout,
    grid_descriptor_set_layout: vk::DescriptorSetLayout,
    grid_descriptor_pool: vk::DescriptorPool,
    grid_descriptor_set: vk::DescriptorSet,

    // Road line pipeline (graphics)
    line_pipeline: vk::Pipeline,
    line_pipeline_layout: vk::PipelineLayout,

    // Road editing
    pending_points: Vec<ControlPoint>,
    network: RoadNetwork,
    dragging: Option<(usize, usize)>,
    dirty: bool,

    // Polyline rendering
    polyline_buffer: Option<GpuBuffer>,
    retired_buffers: Vec<GpuBuffer>,
    road_draws: Vec<DrawRange>,
    pending_draw: Option<DrawRange>,
    cp_draw: Option<DrawRange>,

    // GPU road data (SSBOs)
    gpu_road_data: GpuRoadData,

    // SDF tile system
    sdf_manager: Option<SdfTileManager>,

    // Road rendering (replaces SDF debug visualization)
    road_render_pipeline: vk::Pipeline,
    road_render_pipeline_layout: vk::PipelineLayout,
    road_render_descriptor_set_layout: vk::DescriptorSetLayout,
    road_render_descriptor_pool: vk::DescriptorPool,
    road_render_descriptor_set: vk::DescriptorSet,
    road_render_sampler: vk::Sampler,
    road_render_descriptors_valid: bool,

    // Traffic simulation
    traffic: TrafficSim,

    // Simulation speed multiplier (0 = paused, 0.5, 1.0, 2.0, 4.0)
    sim_speed: f32,

    // FPS tracking
    fps_accumulator: f32,
    fps_frame_count: u32,
    fps_display: f32,

    // GPU timestamp profiling
    gpu_timestamps: Option<GpuTimestamps>,
}

impl Default for TrafficApp {
    fn default() -> Self {
        Self {
            grid_pipeline: vk::Pipeline::null(),
            grid_pipeline_layout: vk::PipelineLayout::null(),
            grid_descriptor_set_layout: vk::DescriptorSetLayout::null(),
            grid_descriptor_pool: vk::DescriptorPool::null(),
            grid_descriptor_set: vk::DescriptorSet::null(),

            line_pipeline: vk::Pipeline::null(),
            line_pipeline_layout: vk::PipelineLayout::null(),

            pending_points: Vec::new(),
            network: RoadNetwork::new(),
            dragging: None,
            dirty: false,

            polyline_buffer: None,
            retired_buffers: Vec::new(),
            road_draws: Vec::new(),
            pending_draw: None,
            cp_draw: None,

            gpu_road_data: GpuRoadData::new(),

            sdf_manager: None,

            road_render_pipeline: vk::Pipeline::null(),
            road_render_pipeline_layout: vk::PipelineLayout::null(),
            road_render_descriptor_set_layout: vk::DescriptorSetLayout::null(),
            road_render_descriptor_pool: vk::DescriptorPool::null(),
            road_render_descriptor_set: vk::DescriptorSet::null(),
            road_render_sampler: vk::Sampler::null(),
            road_render_descriptors_valid: false,

            traffic: TrafficSim::default(),

            sim_speed: 1.0,
            fps_accumulator: 0.0,
            fps_frame_count: 0,
            fps_display: 0.0,
            gpu_timestamps: None,
        }
    }
}

impl TrafficApp {
    fn create_grid_pipeline(&mut self, ctx: &EngineContext) -> anyhow::Result<()> {
        let device = ctx.device.as_ref();

        let bindings = [vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build()];

        self.grid_descriptor_set_layout = create_descriptor_set_layout(device, &bindings)?;

        let pool_sizes = [vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .build()];
        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        self.grid_descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        self.grid_descriptor_set = allocate_descriptor_set(
            device,
            self.grid_descriptor_pool,
            self.grid_descriptor_set_layout,
        )?;

        self.update_grid_descriptors(ctx);

        let spirv = compile_wgsl(GRID_SHADER)?;

        let push_constant_ranges = [vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<CameraParams>() as u32)
            .build()];

        let (pipeline, layout) = create_compute_pipeline(
            device,
            &spirv,
            &[self.grid_descriptor_set_layout],
            &push_constant_ranges,
        )?;

        self.grid_pipeline = pipeline;
        self.grid_pipeline_layout = layout;

        Ok(())
    }

    fn create_line_pipeline(&mut self, ctx: &EngineContext) -> anyhow::Result<()> {
        let device = ctx.device.as_ref();
        let spirv = compile_wgsl(ROAD_LINE_SHADER)?;

        let push_constant_ranges = [vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<LinePushConstants>() as u32)
            .build()];

        let vertex_bindings = [vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<LineVertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()];

        let vertex_attrs = [
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(0)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(8)
                .build(),
        ];

        let desc = GraphicsPipelineDesc {
            vertex_spirv: &spirv,
            fragment_spirv: &spirv,
            vertex_entry: "vs_main",
            fragment_entry: "fs_main",
            vertex_binding_descriptions: &vertex_bindings,
            vertex_attribute_descriptions: &vertex_attrs,
            topology: vk::PrimitiveTopology::LINE_STRIP,
            color_attachment_format: ctx.draw_image.format,
            push_constant_ranges: &push_constant_ranges,
            descriptor_set_layouts: &[],
            line_width: 1.0,
        };

        let (pipeline, layout) = create_graphics_pipeline(device, &desc)?;
        self.line_pipeline = pipeline;
        self.line_pipeline_layout = layout;

        Ok(())
    }

    fn update_grid_descriptors(&self, ctx: &EngineContext) {
        let image_info = [vk::DescriptorImageInfo::builder()
            .image_view(ctx.draw_image.view)
            .image_layout(vk::ImageLayout::GENERAL)
            .build()];

        let writes = [vk::WriteDescriptorSet::builder()
            .dst_set(self.grid_descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&image_info)
            .build()];

        unsafe {
            ctx.device
                .update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
        }
    }

    fn create_sdf_system(&mut self, ctx: &EngineContext) -> anyhow::Result<()> {
        let device = ctx.device.as_ref();

        // Create SDF tile manager
        let sdf_manager = SdfTileManager::new(
            ctx.device,
            ctx.allocator,
            SDF_TYPES_WGSL,
            SDF_ROAD_EVAL_WGSL,
            SDF_GENERATE_WGSL,
        )?;

        // Compile road render shader (prepend shared types for GpuRoad/GpuLaneSection/GpuLane)
        let full_source = format!("{}\n{}", SDF_TYPES_WGSL, ROAD_RENDER_WGSL);
        let spirv = compile_wgsl(&full_source)?;

        // Sampler for SDF atlas
        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        self.road_render_sampler = unsafe { device.create_sampler(&sampler_info, None)? };

        // Descriptor layout: 5 bindings
        //   0: sampled image (SDF atlas)
        //   1: sampler
        //   2: storage buffer (roads)
        //   3: storage buffer (lane_sections)
        //   4: storage buffer (lanes)
        let bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build(),
        ];
        self.road_render_descriptor_set_layout = create_descriptor_set_layout(device, &bindings)?;

        let pool_sizes = [
            vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .build(),
            vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .build(),
            vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(3)
                .build(),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        self.road_render_descriptor_pool =
            unsafe { device.create_descriptor_pool(&pool_info, None)? };

        self.road_render_descriptor_set = allocate_descriptor_set(
            device,
            self.road_render_descriptor_pool,
            self.road_render_descriptor_set_layout,
        )?;

        // Write atlas image + sampler descriptors (these don't change)
        let image_info = [vk::DescriptorImageInfo::builder()
            .image_view(sdf_manager.atlas.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .build()];
        let sampler_info_desc = [vk::DescriptorImageInfo::builder()
            .sampler(self.road_render_sampler)
            .build()];
        let writes = [
            vk::WriteDescriptorSet::builder()
                .dst_set(self.road_render_descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&image_info)
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(self.road_render_descriptor_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(&sampler_info_desc)
                .build(),
        ];
        unsafe {
            device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
        }

        // Graphics pipeline for road rendering
        let push_constant_ranges = [vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<SdfDebugPushConstants>() as u32)
            .build()];

        let desc = GraphicsPipelineDesc {
            vertex_spirv: &spirv,
            fragment_spirv: &spirv,
            vertex_entry: "vs_main",
            fragment_entry: "fs_main",
            vertex_binding_descriptions: &[],
            vertex_attribute_descriptions: &[],
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            color_attachment_format: ctx.draw_image.format,
            push_constant_ranges: &push_constant_ranges,
            descriptor_set_layouts: &[self.road_render_descriptor_set_layout],
            line_width: 1.0,
        };

        let (pipeline, layout) = create_graphics_pipeline(device, &desc)?;
        self.road_render_pipeline = pipeline;
        self.road_render_pipeline_layout = layout;

        self.sdf_manager = Some(sdf_manager);
        Ok(())
    }

    /// Update road render descriptors when road data buffers change.
    fn update_road_render_descriptors(&mut self, ctx: &EngineContext) {
        let device = ctx.device.as_ref();
        let road_buf = match &self.gpu_road_data.road_buffer {
            Some(b) => b,
            None => {
                self.road_render_descriptors_valid = false;
                return;
            }
        };
        let ls_buf = match &self.gpu_road_data.lane_section_buffer {
            Some(b) => b,
            None => {
                self.road_render_descriptors_valid = false;
                return;
            }
        };
        let lane_buf = match &self.gpu_road_data.lane_buffer {
            Some(b) => b,
            None => {
                self.road_render_descriptors_valid = false;
                return;
            }
        };

        let road_buf_info = [vk::DescriptorBufferInfo::builder()
            .buffer(road_buf.buffer)
            .offset(0)
            .range(road_buf.size)
            .build()];
        let ls_buf_info = [vk::DescriptorBufferInfo::builder()
            .buffer(ls_buf.buffer)
            .offset(0)
            .range(ls_buf.size)
            .build()];
        let lane_buf_info = [vk::DescriptorBufferInfo::builder()
            .buffer(lane_buf.buffer)
            .offset(0)
            .range(lane_buf.size)
            .build()];

        let writes = [
            vk::WriteDescriptorSet::builder()
                .dst_set(self.road_render_descriptor_set)
                .dst_binding(2)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&road_buf_info)
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(self.road_render_descriptor_set)
                .dst_binding(3)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&ls_buf_info)
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(self.road_render_descriptor_set)
                .dst_binding(4)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&lane_buf_info)
                .build(),
        ];

        unsafe {
            device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
        }
        self.road_render_descriptors_valid = true;
    }

    /// Rebuild tile map data from current road network and upload to GPU.
    fn rebuild_sdf_tiles(&mut self, ctx: &EngineContext) -> anyhow::Result<()> {
        let sdf = match self.sdf_manager.as_mut() {
            Some(s) => s,
            None => return Ok(()),
        };

        // Build segment info from road network
        let mut seg_infos = Vec::new();
        for (road_idx, road_data) in self.network.roads.iter().enumerate() {
            let rl = &road_data.reference_line;
            for (seg_idx, seg) in rl.segments.iter().enumerate() {
                let (seg_type, k_start, k_end) = match seg {
                    road::primitives::Segment::Line { .. } => (0u32, 0.0f32, 0.0f32),
                    road::primitives::Segment::Arc { curvature, .. } => (1, *curvature, *curvature),
                    road::primitives::Segment::Spiral { k_start, k_end, .. } => {
                        (2, *k_start, *k_end)
                    }
                };

                seg_infos.push(RoadSegmentInfo {
                    segment_index: seg_idx as u32
                        + self.network.roads[..road_idx]
                            .iter()
                            .map(|r| r.reference_line.segments.len() as u32)
                            .sum::<u32>(),
                    road_index: road_idx as u32,
                    seg_type,
                    origin_x: rl.origins[seg_idx].x,
                    origin_y: rl.origins[seg_idx].y,
                    heading: rl.headings[seg_idx],
                    length: seg.length(),
                    k_start,
                    k_end,
                });
            }
        }

        // Compute AABBs and rebuild tile map
        let aabbs = compute_segment_aabbs(&seg_infos);
        sdf.tile_map.rebuild(&aabbs);

        // Upload tile data to GPU
        if let Some(seg_buf) = &self.gpu_road_data.segment_buffer {
            sdf.upload_tile_data(
                ctx.device.as_ref(),
                ctx.allocator,
                seg_buf.buffer,
                seg_buf.size,
            )?;
        }

        Ok(())
    }

    /// Rebuild the polyline vertex buffer from current road state.
    fn rebuild_polyline(&mut self, ctx: &EngineContext) -> anyhow::Result<()> {
        // Retire old buffer — keep it alive until in-flight frames are done (FRAME_OVERLAP=2)
        if let Some(b) = self.polyline_buffer.take() {
            self.retired_buffers.push(b);
        }
        // Destroy buffers that are old enough (keep at most 2 pending)
        while self.retired_buffers.len() > 2 {
            self.retired_buffers.remove(0).destroy(ctx.allocator);
        }
        self.road_draws.clear();
        self.pending_draw = None;
        self.cp_draw = None;

        let mut vertices: Vec<LineVertex> = Vec::new();

        let road_color = [0.2f32, 0.9, 0.3, 1.0];
        let pending_color = [1.0f32, 0.9, 0.1, 1.0];
        let cp_color = [1.0f32, 0.2, 0.2, 1.0];

        // Tessellate finalized roads (LINE_STRIP per road)
        for road_data in &self.network.roads {
            let rl = &road_data.reference_line;
            let sample_dist = 1.0; // 1 m between samples
            let n_samples = (rl.total_length / sample_dist).ceil() as usize + 1;
            let first = vertices.len() as u32;

            for i in 0..n_samples {
                let s = (i as f32 * sample_dist).min(rl.total_length);
                let pose = rl.evaluate(s);
                vertices.push(LineVertex {
                    position: [pose.position.x, pose.position.y],
                    color: road_color,
                });
            }

            self.road_draws.push(DrawRange {
                first_vertex: first,
                vertex_count: vertices.len() as u32 - first,
            });
        }

        // Tessellate pending road
        if self.pending_points.len() >= 2 {
            if let Some(rl) = ReferenceLine::fit(&self.pending_points) {
                let sample_dist = 1.0;
                let n_samples = (rl.total_length / sample_dist).ceil() as usize + 1;
                let first = vertices.len() as u32;

                for i in 0..n_samples {
                    let s = (i as f32 * sample_dist).min(rl.total_length);
                    let pose = rl.evaluate(s);
                    vertices.push(LineVertex {
                        position: [pose.position.x, pose.position.y],
                        color: pending_color,
                    });
                }

                self.pending_draw = Some(DrawRange {
                    first_vertex: first,
                    vertex_count: vertices.len() as u32 - first,
                });
            }
        }

        // Control point crosses (drawn as LINE_STRIP pairs of 2 vertices)
        let cross_size = 0.5;
        let cp_first = vertices.len() as u32;

        for cp in &self.pending_points {
            push_cross(&mut vertices, cp.position, cross_size, cp_color);
        }
        for road_data in &self.network.roads {
            for cp in &road_data.control_points {
                push_cross(&mut vertices, cp.position, cross_size, cp_color);
            }
        }

        let cp_count = vertices.len() as u32 - cp_first;
        if cp_count > 0 {
            self.cp_draw = Some(DrawRange {
                first_vertex: cp_first,
                vertex_count: cp_count,
            });
        }

        if vertices.is_empty() {
            return Ok(());
        }

        let data = bytemuck::cast_slice::<_, u8>(&vertices);
        let (buf, ptr) = GpuBuffer::new_mapped(
            ctx.allocator,
            data.len() as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()) };
        self.polyline_buffer = Some(buf);

        Ok(())
    }

    /// Handle input for road editing. Returns true if polyline needs rebuild.
    fn process_input(&mut self, ctx: &EngineContext) -> bool {
        let mut needs_rebuild = false;

        let cursor_world = ctx.camera.screen_to_world(
            Vec2::new(ctx.input.mouse_x as f32, ctx.input.mouse_y as f32),
            Vec2::new(ctx.window_width as f32, ctx.window_height as f32),
        );

        // Left click: place control point or start drag
        if ctx.input.left_mouse_pressed {
            // Hit radius: 10 screen pixels converted to world units
            let pixels_to_world = (2.0 / ctx.camera.zoom) / ctx.window_height as f32;
            let hit_radius = 10.0 * pixels_to_world;
            let mut hit = None;

            for (ri, road_data) in self.network.roads.iter().enumerate() {
                for (pi, cp) in road_data.control_points.iter().enumerate() {
                    if (cp.position - cursor_world).length() < hit_radius {
                        hit = Some((ri, pi));
                        break;
                    }
                }
                if hit.is_some() {
                    break;
                }
            }

            if let Some((ri, pi)) = hit {
                self.dragging = Some((ri, pi));
            } else {
                self.pending_points.push(ControlPoint {
                    position: cursor_world,
                    turn_radius: 20.0,
                    spiral_length: 5.0,
                });
                needs_rebuild = true;
            }
        }

        // Dragging
        if ctx.input.left_mouse {
            if let Some((ri, pi)) = self.dragging {
                if ri < self.network.roads.len() {
                    self.network.roads[ri].control_points[pi].position = cursor_world;
                    if let Some(road_data) =
                        Road::new_with_defaults(self.network.roads[ri].control_points.clone())
                    {
                        self.network.roads[ri] = road_data;
                        self.dirty = true;
                        needs_rebuild = true;
                    }
                }
            }
        }

        // Release drag
        if !ctx.input.left_mouse {
            self.dragging = None;
        }

        // Right click: finalize road
        if ctx.input.right_mouse_pressed && self.pending_points.len() >= 2 {
            let pts = std::mem::take(&mut self.pending_points);
            if let Some(road_data) = Road::new_with_defaults(pts) {
                self.network.add_road(road_data);
                self.dirty = true;
                needs_rebuild = true;
            }
        }

        // Escape: cancel current road
        if ctx.input.escape_pressed && !self.pending_points.is_empty() {
            self.pending_points.clear();
            needs_rebuild = true;
        }

        // Simulation speed controls
        use engine::winit::keyboard::{Key, NamedKey};
        for key in &ctx.input.keys_pressed {
            match key {
                Key::Character(c) if c.as_str() == "1" => self.sim_speed = 0.5,
                Key::Character(c) if c.as_str() == "2" => self.sim_speed = 1.0,
                Key::Character(c) if c.as_str() == "3" => self.sim_speed = 2.0,
                Key::Character(c) if c.as_str() == "4" => self.sim_speed = 4.0,
                Key::Named(NamedKey::Space) => {
                    self.sim_speed = if self.sim_speed == 0.0 { 1.0 } else { 0.0 };
                }
                _ => {}
            }
        }

        needs_rebuild
    }
}

fn push_cross(vertices: &mut Vec<LineVertex>, pos: Vec2, size: f32, color: [f32; 4]) {
    vertices.push(LineVertex {
        position: [pos.x - size, pos.y],
        color,
    });
    vertices.push(LineVertex {
        position: [pos.x + size, pos.y],
        color,
    });
    vertices.push(LineVertex {
        position: [pos.x, pos.y - size],
        color,
    });
    vertices.push(LineVertex {
        position: [pos.x, pos.y + size],
        color,
    });
}

impl App for TrafficApp {
    fn init(&mut self, ctx: &EngineContext) -> anyhow::Result<()> {
        self.create_grid_pipeline(ctx)?;
        self.create_line_pipeline(ctx)?;
        self.create_sdf_system(ctx)?;
        self.traffic.create_pipelines(ctx)?;
        self.gpu_timestamps = Some(GpuTimestamps::new(
            ctx.device.as_ref(),
            ctx.timestamp_period,
        )?);
        Ok(())
    }

    fn resize(&mut self, ctx: &EngineContext) -> anyhow::Result<()> {
        self.update_grid_descriptors(ctx);
        Ok(())
    }

    fn render(&mut self, ctx: &EngineContext, cmd: vk::CommandBuffer) -> anyhow::Result<()> {
        let device = ctx.device.as_ref();

        // Read GPU timestamps from previous frame, then reset for this frame
        if let Some(ts) = &mut self.gpu_timestamps {
            ts.read_results(device);
            ts.reset_and_begin(device, cmd);
        }

        // Process input
        let needs_rebuild = self.process_input(ctx);
        if needs_rebuild {
            self.rebuild_polyline(ctx)?;
        }

        // Upload GPU road data if dirty
        if self.dirty {
            // Wait for in-flight frames before destroying/recreating buffers
            unsafe {
                ctx.device.device_wait_idle().unwrap();
            }

            self.gpu_road_data.upload(ctx.allocator, &self.network)?;
            self.rebuild_sdf_tiles(ctx)?;
            self.update_road_render_descriptors(ctx);

            // Initialize/reinitialize traffic cars on road change
            // Compute density-aware car count: target ~1 car per 60m per lane
            // (ensures IDM gaps of ~55m which is above the desired s* at 30 m/s)
            let cars_per_road = {
                let mut total = 0u32;
                for road_data in &self.network.roads {
                    let road_len = road_data.reference_line.total_length;
                    let lane_count = if road_data.lane_sections.is_empty() {
                        1u32
                    } else {
                        let sec = &road_data.lane_sections[0];
                        (sec.right_lanes.len() + sec.left_lanes.len()) as u32
                    };
                    // Target spacing: 60m per car (car_length 4.5 + IDM gap ~55m)
                    let cars_this_road = ((road_len / 60.0) as u32) * lane_count;
                    total += cars_this_road.max(1);
                }
                total / self.network.roads.len().max(1) as u32
            };
            self.traffic
                .initialize_cars(ctx.allocator, &self.network, cars_per_road)?;
            self.traffic.update_descriptors(
                ctx.device.as_ref(),
                self.gpu_road_data.segment_buffer.as_ref(),
                self.gpu_road_data.road_buffer.as_ref(),
                self.gpu_road_data.lane_section_buffer.as_ref(),
                self.gpu_road_data.lane_buffer.as_ref(),
            );

            self.dirty = false;
        }

        // --- Compute: SDF tile generation (dirty tiles only) ---
        if let Some(sdf) = &mut self.sdf_manager {
            if sdf.has_dirty_tiles() {
                // Transition SDF atlas to GENERAL for compute write
                transition_image(
                    device,
                    cmd,
                    sdf.atlas.image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                );

                sdf.dispatch_dirty_tiles(device, cmd);

                // Memory barrier: compute writes → fragment reads
                unsafe {
                    let barrier = vk::MemoryBarrier2::builder()
                        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_READ);
                    let dep = vk::DependencyInfo::builder()
                        .memory_barriers(std::slice::from_ref(&barrier));
                    device.cmd_pipeline_barrier2(cmd, &dep);
                }

                // Transition atlas to shader read for debug vis
                transition_image(
                    device,
                    cmd,
                    sdf.atlas.image,
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
            }
        }

        // Timestamp: SDF done
        if let Some(ts) = &self.gpu_timestamps {
            ts.write(device, cmd, TS_SDF_END);
        }

        // --- Compute: grid background ---
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.grid_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.grid_pipeline_layout,
                0,
                &[self.grid_descriptor_set],
                &[],
            );

            let aspect = ctx.window_width as f32 / ctx.window_height as f32;
            let inv_vp = ctx.camera.inverse_view_projection(aspect);
            let params = CameraParams {
                inv_vp: inv_vp.to_cols_array_2d(),
            };
            device.cmd_push_constants(
                cmd,
                self.grid_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&params),
            );

            let wg_x = (ctx.window_width + 15) / 16;
            let wg_y = (ctx.window_height + 15) / 16;
            device.cmd_dispatch(cmd, wg_x, wg_y, 1);
        }

        // Timestamp: grid done
        if let Some(ts) = &self.gpu_timestamps {
            ts.write(device, cmd, TS_GRID_END);
        }

        // --- Compute: traffic simulation update (fixed timestep) ---
        // Phase 8: Sort → IDM → MOBIL (replaces simple s += v*dt)
        if self.traffic.initialized && self.traffic.car_count > 0 {
            self.traffic.sim_accumulator += ctx.dt * self.sim_speed;
            // Cap to max ticks per frame (higher for faster sim speeds)
            let max_ticks = (2.0 * self.sim_speed).ceil().max(2.0) as u32;
            let mut ticks_this_frame = 0u32;
            while self.traffic.sim_accumulator >= SIM_DT && ticks_this_frame < max_ticks {
                self.traffic.sim_accumulator -= SIM_DT;
                self.traffic.sim_tick += 1;
                ticks_this_frame += 1;

                let car_count = self.traffic.car_count;
                let num_sort_wg = (car_count + SORT_TILE_SIZE - 1) / SORT_TILE_SIZE;

                unsafe {
                    // === Step 1: Build sort keys ===
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        self.traffic.sort_keys_pipeline,
                    );
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        self.traffic.sort_keys_pipeline_layout,
                        0,
                        &[self.traffic.sort_keys_descriptor_set],
                        &[],
                    );
                    let keys_pc = SortKeysPushConstants {
                        car_count,
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    };
                    device.cmd_push_constants(
                        cmd,
                        self.traffic.sort_keys_pipeline_layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        bytemuck::bytes_of(&keys_pc),
                    );
                    device.cmd_dispatch(cmd, num_sort_wg, 1, 1);

                    // Barrier: keys written → histogram reads
                    let barrier = vk::MemoryBarrier2::builder()
                        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(
                            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                        );
                    let dep = vk::DependencyInfo::builder()
                        .memory_barriers(std::slice::from_ref(&barrier));
                    device.cmd_pipeline_barrier2(cmd, &dep);

                    // === Step 2: Radix sort (4 passes of 8 bits) ===
                    let hist_buf = self.traffic.sort_histogram_buf.as_ref().unwrap();
                    let hist_buf_size = hist_buf.size;
                    let hist_buffer = hist_buf.buffer;

                    for pass in 0u32..4 {
                        // Which descriptor sets to use:
                        // Even passes (0, 2): read from A (set index 0), scatter A→B (set index 0)
                        // Odd passes (1, 3): read from B (set index 1), scatter B→A (set index 1)
                        let set_idx = (pass % 2) as usize;

                        // Zero histogram buffer before each pass
                        device.cmd_fill_buffer(cmd, hist_buffer, 0, hist_buf_size, 0);
                        let fill_barrier = vk::MemoryBarrier2::builder()
                            .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .dst_access_mask(
                                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                            );
                        let fill_dep = vk::DependencyInfo::builder()
                            .memory_barriers(std::slice::from_ref(&fill_barrier));
                        device.cmd_pipeline_barrier2(cmd, &fill_dep);

                        // --- Histogram ---
                        device.cmd_bind_pipeline(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            self.traffic.sort_histogram_pipeline,
                        );
                        device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            self.traffic.sort_histogram_pipeline_layout,
                            0,
                            &[self.traffic.sort_histogram_descriptor_sets[set_idx]],
                            &[],
                        );
                        let hist_pc = SortHistogramPushConstants {
                            pass_id: pass,
                            car_count,
                            num_workgroups: num_sort_wg,
                            _pad: 0,
                        };
                        device.cmd_push_constants(
                            cmd,
                            self.traffic.sort_histogram_pipeline_layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            bytemuck::bytes_of(&hist_pc),
                        );
                        device.cmd_dispatch(cmd, num_sort_wg, 1, 1);

                        // Barrier: histogram written → scan reads
                        device.cmd_pipeline_barrier2(cmd, &dep);

                        // --- Prefix sum (scan) ---
                        device.cmd_bind_pipeline(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            self.traffic.sort_scan_pipeline,
                        );
                        device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            self.traffic.sort_scan_pipeline_layout,
                            0,
                            &[self.traffic.sort_scan_descriptor_set],
                            &[],
                        );
                        let scan_pc = SortScanPushConstants {
                            num_workgroups: num_sort_wg,
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        };
                        device.cmd_push_constants(
                            cmd,
                            self.traffic.sort_scan_pipeline_layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            bytemuck::bytes_of(&scan_pc),
                        );
                        // 1 workgroup: single thread processes all 256 digit rows
                        device.cmd_dispatch(cmd, 1, 1, 1);

                        // Barrier: scan written → scatter reads
                        device.cmd_pipeline_barrier2(cmd, &dep);

                        // --- Scatter ---
                        device.cmd_bind_pipeline(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            self.traffic.sort_scatter_pipeline,
                        );
                        device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            self.traffic.sort_scatter_pipeline_layout,
                            0,
                            &[self.traffic.sort_scatter_descriptor_sets[set_idx]],
                            &[],
                        );
                        let scatter_pc = SortScatterPushConstants {
                            pass_id: pass,
                            car_count,
                            num_workgroups: num_sort_wg,
                            _pad: 0,
                        };
                        device.cmd_push_constants(
                            cmd,
                            self.traffic.sort_scatter_pipeline_layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            bytemuck::bytes_of(&scatter_pc),
                        );
                        device.cmd_dispatch(cmd, num_sort_wg, 1, 1);

                        // Barrier between passes
                        device.cmd_pipeline_barrier2(cmd, &dep);
                    }
                    // After 4 passes: sorted result is in buffer A
                    // (pass 0: A→B, pass 1: B→A, pass 2: A→B, pass 3: B→A → final in A)

                    // === Step 3: IDM car-following ===
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        self.traffic.idm_pipeline,
                    );
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        self.traffic.idm_pipeline_layout,
                        0,
                        &[self.traffic.idm_descriptor_set],
                        &[],
                    );
                    let idm_pc = IdmPushConstants {
                        dt: SIM_DT,
                        car_count,
                        a_max: 1.5,
                        b_comfort: 2.0,
                        s0: 2.0,
                        time_headway: 1.5,
                        car_length: 4.5,
                        _pad: 0,
                    };
                    device.cmd_push_constants(
                        cmd,
                        self.traffic.idm_pipeline_layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        bytemuck::bytes_of(&idm_pc),
                    );
                    device.cmd_dispatch(cmd, (car_count + 255) / 256, 1, 1);

                    // Barrier: IDM writes → MOBIL reads (or vertex reads)
                    device.cmd_pipeline_barrier2(cmd, &dep);

                    // === Step 4: MOBIL lane change (every Nth tick) ===
                    if self.traffic.sim_tick % LANE_CHANGE_INTERVAL == 0 {
                        device.cmd_bind_pipeline(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            self.traffic.mobil_pipeline,
                        );
                        device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            self.traffic.mobil_pipeline_layout,
                            0,
                            &[self.traffic.mobil_descriptor_set],
                            &[],
                        );
                        let mobil_pc = MobilPushConstants {
                            car_count,
                            a_max: 1.5,
                            b_comfort: 2.0,
                            s0: 2.0,
                            time_headway: 1.5,
                            car_length: 4.5,
                            politeness: 0.5,
                            threshold: 0.5,
                            b_safe: 4.0,
                            max_right_lanes: 2,          // 2 right lanes (0, 1)
                            max_left_lanes: 2,           // 2 left lanes (-1, -2)
                            _pad: self.traffic.sim_tick, // stagger phase: car_idx % 4 == tick % 4
                        };
                        device.cmd_push_constants(
                            cmd,
                            self.traffic.mobil_pipeline_layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            bytemuck::bytes_of(&mobil_pc),
                        );
                        device.cmd_dispatch(cmd, (car_count + 255) / 256, 1, 1);

                        // Barrier: MOBIL writes → next iteration reads
                        device.cmd_pipeline_barrier2(cmd, &dep);
                    }

                    // Final barrier: compute writes → vertex shader reads
                    let final_barrier = vk::MemoryBarrier2::builder()
                        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_READ);
                    let final_dep = vk::DependencyInfo::builder()
                        .memory_barriers(std::slice::from_ref(&final_barrier));
                    device.cmd_pipeline_barrier2(cmd, &final_dep);
                }
            }
            // Clamp accumulator to prevent unbounded growth
            if self.traffic.sim_accumulator > SIM_DT * 4.0 {
                self.traffic.sim_accumulator = 0.0;
            }
        }

        // Timestamp: traffic sim done
        if let Some(ts) = &self.gpu_timestamps {
            ts.write(device, cmd, TS_TRAFFIC_END);
        }

        // --- Graphics: SDF debug visualization + road polylines ---
        {
            transition_image(
                device,
                cmd,
                ctx.draw_image.image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );

            let color_attachment = vk::RenderingAttachmentInfo::builder()
                .image_view(ctx.draw_image.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE);

            let render_area = vk::Rect2D::builder().extent(
                vk::Extent2D::builder()
                    .width(ctx.window_width)
                    .height(ctx.window_height)
                    .build(),
            );

            let rendering_info = vk::RenderingInfo::builder()
                .render_area(*render_area)
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color_attachment));

            unsafe {
                device.cmd_begin_rendering(cmd, &rendering_info);

                let viewport = vk::Viewport::builder()
                    .x(0.0)
                    .y(0.0)
                    .width(ctx.window_width as f32)
                    .height(ctx.window_height as f32)
                    .min_depth(0.0)
                    .max_depth(1.0);
                device.cmd_set_viewport(cmd, 0, &[*viewport]);

                let scissor = vk::Rect2D::builder().extent(
                    vk::Extent2D::builder()
                        .width(ctx.window_width)
                        .height(ctx.window_height)
                        .build(),
                );
                device.cmd_set_scissor(cmd, 0, &[*scissor]);

                let aspect = ctx.window_width as f32 / ctx.window_height as f32;
                let vp = ctx.camera.view_projection(aspect);

                // --- Draw road tiles ---
                if let Some(sdf) = &self.sdf_manager {
                    if !sdf.tile_map.tiles.is_empty() && self.road_render_descriptors_valid {
                        device.cmd_bind_pipeline(
                            cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.road_render_pipeline,
                        );
                        device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.road_render_pipeline_layout,
                            0,
                            &[self.road_render_descriptor_set],
                            &[],
                        );

                        let atlas_tiles = ATLAS_TILES;
                        let atlas_size_f = (atlas_tiles * TILE_RESOLUTION) as f32;
                        // Half-texel inset to prevent bilinear bleed across tile boundaries
                        let half_texel = 0.5 / atlas_size_f;

                        for (key, info) in &sdf.tile_map.tiles {
                            let slot_x = info.atlas_slot % atlas_tiles;
                            let slot_y = info.atlas_slot / atlas_tiles;
                            let uv_offset_x =
                                (slot_x * TILE_RESOLUTION) as f32 / atlas_size_f + half_texel;
                            let uv_offset_y =
                                (slot_y * TILE_RESOLUTION) as f32 / atlas_size_f + half_texel;
                            let uv_scale = TILE_RESOLUTION as f32 / atlas_size_f - 2.0 * half_texel;

                            let (wx, wy) = key.world_origin();

                            let pc = SdfDebugPushConstants {
                                view_proj: vp.to_cols_array_2d(),
                                atlas_uv_offset: [uv_offset_x, uv_offset_y],
                                atlas_uv_scale: [uv_scale, uv_scale],
                                tile_world_origin: [wx, wy],
                                tile_world_size: [TILE_SIZE, TILE_SIZE],
                            };

                            device.cmd_push_constants(
                                cmd,
                                self.road_render_pipeline_layout,
                                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                                0,
                                bytemuck::bytes_of(&pc),
                            );

                            device.cmd_draw(cmd, 6, 1, 0, 0);
                        }
                    }
                }

                // --- Draw road polylines ---
                if self.polyline_buffer.is_some() {
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.line_pipeline,
                    );

                    let pc = LinePushConstants {
                        view_proj: vp.to_cols_array_2d(),
                    };
                    device.cmd_push_constants(
                        cmd,
                        self.line_pipeline_layout,
                        vk::ShaderStageFlags::VERTEX,
                        0,
                        bytemuck::bytes_of(&pc),
                    );

                    let vb = self.polyline_buffer.as_ref().unwrap();
                    device.cmd_bind_vertex_buffers(cmd, 0, &[vb.buffer], &[0]);

                    // Draw finalized roads (LINE_STRIP per road)
                    for draw in &self.road_draws {
                        device.cmd_draw(cmd, draw.vertex_count, 1, draw.first_vertex, 0);
                    }

                    // Draw pending road
                    if let Some(ref draw) = self.pending_draw {
                        device.cmd_draw(cmd, draw.vertex_count, 1, draw.first_vertex, 0);
                    }

                    // Draw control point crosses (pairs of 2 vertices)
                    if let Some(ref draw) = self.cp_draw {
                        let mut v = draw.first_vertex;
                        let end = draw.first_vertex + draw.vertex_count;
                        while v + 1 < end {
                            device.cmd_draw(cmd, 2, 1, v, 0);
                            v += 2;
                        }
                    }
                }

                // --- Draw cars (instanced rendering) ---
                if self.traffic.initialized && self.traffic.car_count > 0 {
                    if let Some(renderer) = &self.traffic.car_renderer {
                        if renderer.is_ready() {
                            let pc = CarRenderPushConstants {
                                view_proj: vp.to_cols_array_2d(),
                                car_count: self.traffic.car_count,
                                _pad0: 0,
                                _pad1: 0,
                                _pad2: 0,
                            };
                            renderer.draw(device, cmd, &pc);
                        }
                    }
                }

                device.cmd_end_rendering(cmd);
            }

            transition_image(
                device,
                cmd,
                ctx.draw_image.image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::GENERAL,
            );
        }

        // Timestamp: render done
        if let Some(ts) = &self.gpu_timestamps {
            ts.write(device, cmd, TS_RENDER_END);
        }

        // --- Window title HUD ---
        self.fps_accumulator += ctx.dt;
        self.fps_frame_count += 1;
        if self.fps_accumulator >= 0.5 {
            self.fps_display = self.fps_frame_count as f32 / self.fps_accumulator;
            self.fps_accumulator = 0.0;
            self.fps_frame_count = 0;
        }
        if let Some(window) = ctx.window {
            let speed_label = if self.sim_speed == 0.0 {
                "PAUSED".to_string()
            } else {
                format!("{:.1}x", self.sim_speed)
            };
            let gpu_info = if let Some(ts) = &self.gpu_timestamps {
                format!(
                    " | GPU: {:.1}ms (sdf:{:.1} grid:{:.1} sim:{:.1} draw:{:.1})",
                    ts.total_ms, ts.sdf_ms, ts.grid_ms, ts.traffic_ms, ts.render_ms,
                )
            } else {
                String::new()
            };
            window.set_title(&format!(
                "Traffic Sim | {:.0} FPS | {} cars | {}{} | [Space]=pause [1-4]=speed",
                self.fps_display, self.traffic.car_count, speed_label, gpu_info,
            ));
        }

        Ok(())
    }

    fn shutdown(&mut self, ctx: &EngineContext) {
        unsafe {
            ctx.device.destroy_pipeline(self.grid_pipeline, None);
            ctx.device
                .destroy_pipeline_layout(self.grid_pipeline_layout, None);
            ctx.device
                .destroy_descriptor_pool(self.grid_descriptor_pool, None);
            ctx.device
                .destroy_descriptor_set_layout(self.grid_descriptor_set_layout, None);
            ctx.device.destroy_pipeline(self.line_pipeline, None);
            ctx.device
                .destroy_pipeline_layout(self.line_pipeline_layout, None);
            ctx.device.destroy_pipeline(self.road_render_pipeline, None);
            ctx.device
                .destroy_pipeline_layout(self.road_render_pipeline_layout, None);
            ctx.device
                .destroy_descriptor_pool(self.road_render_descriptor_pool, None);
            ctx.device
                .destroy_descriptor_set_layout(self.road_render_descriptor_set_layout, None);
            ctx.device.destroy_sampler(self.road_render_sampler, None);
        }
        if let Some(b) = self.polyline_buffer.take() {
            b.destroy(ctx.allocator);
        }
        for b in self.retired_buffers.drain(..) {
            b.destroy(ctx.allocator);
        }
        self.gpu_road_data.destroy(ctx.allocator);
        if let Some(mut sdf) = self.sdf_manager.take() {
            sdf.destroy(ctx.device.as_ref(), ctx.allocator);
        }
        self.traffic.destroy(ctx.device.as_ref(), ctx.allocator);
        if let Some(ts) = self.gpu_timestamps.take() {
            ts.destroy(ctx.device.as_ref());
        }
    }
}

fn main() -> anyhow::Result<()> {
    engine::run(TrafficApp::default())
}

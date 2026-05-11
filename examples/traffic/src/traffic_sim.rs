use engine::car_renderer::{CarRenderPushConstants, CarRenderer};
use engine::gpu_resources::GpuBuffer;
use engine::pipeline::write_storage_buffers;
use engine::vk::{self, DeviceV1_0, DeviceV1_3, HasBuilder};
use engine::{ComputePass, EngineContext};
use rand::Rng;
use road::network::RoadNetwork;

// ---------------------------------------------------------------------------
// Shader sources
// ---------------------------------------------------------------------------

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
// Simulation constants
// ---------------------------------------------------------------------------

/// Maximum number of cars (power of 2, > 500k)
pub const MAX_CARS: u32 = 524_288;

/// Fixed simulation timestep (1/60 s)
pub const SIM_DT: f32 = 1.0 / 60.0;

/// Sort workgroup tile size (elements per workgroup)
const SORT_TILE_SIZE: u32 = 256;

/// Number of radix sort workgroups
const NUM_SORT_WG: u32 = MAX_CARS / SORT_TILE_SIZE; // 2048

/// MOBIL lane change frequency (every N simulation ticks)
const LANE_CHANGE_INTERVAL: u32 = 30;

// ---------------------------------------------------------------------------
// IDM parameters
// ---------------------------------------------------------------------------

/// Maximum acceleration (m/s²)
const IDM_A_MAX: f32 = 1.5;
/// Comfortable braking deceleration (m/s²)
const IDM_B_COMFORT: f32 = 2.0;
/// Minimum gap distance (m)
const IDM_S0: f32 = 2.0;
/// Desired time headway (s)
const IDM_TIME_HEADWAY: f32 = 1.5;
/// Car length (m)
const CAR_LENGTH: f32 = 4.5;

// ---------------------------------------------------------------------------
// MOBIL parameters (European model: asymmetric keep-right rule)
// ---------------------------------------------------------------------------

/// Politeness factor
const MOBIL_POLITENESS: f32 = 0.3;
/// Lane change incentive threshold (for overtaking / leftward moves)
const MOBIL_THRESHOLD: f32 = 0.2;
/// Maximum safe braking for follower (m/s²)
const MOBIL_B_SAFE: f32 = 4.0;
/// Keep-right bias: added incentive for returning to the rightmost lane (m/s²)
const MOBIL_KEEP_RIGHT_BIAS: f32 = 0.3;

// ---------------------------------------------------------------------------
// Push constants
// ---------------------------------------------------------------------------

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
    stagger_phase: u32,
    keep_right_bias: f32,
    _pad: u32,
}

// ---------------------------------------------------------------------------
// Traffic simulation state
// ---------------------------------------------------------------------------

pub struct TrafficSim {
    // Car SoA buffers
    car_road_id_buf: Option<GpuBuffer>,
    car_s_buf: Option<GpuBuffer>,
    car_lane_buf: Option<GpuBuffer>,
    car_speed_buf: Option<GpuBuffer>,
    car_desired_speed_buf: Option<GpuBuffer>,
    road_lengths_buf: Option<GpuBuffer>,

    // Sort buffers (ping-pong)
    sort_keys_a_buf: Option<GpuBuffer>,
    sort_keys_b_buf: Option<GpuBuffer>,
    sort_vals_a_buf: Option<GpuBuffer>,
    sort_vals_b_buf: Option<GpuBuffer>,
    sort_histogram_buf: Option<GpuBuffer>,

    // Compute passes
    sort_keys_pass: Option<ComputePass>,
    sort_histogram_pass: Option<ComputePass>,
    sort_scan_pass: Option<ComputePass>,
    sort_scatter_pass: Option<ComputePass>,
    idm_pass: Option<ComputePass>,
    mobil_pass: Option<ComputePass>,

    // Car renderer (instanced rendering)
    car_renderer: Option<CarRenderer>,

    // Simulation state
    pub car_count: u32,
    pub sim_accumulator: f32,
    pub sim_tick: u32,
    pub initialized: bool,
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

            sort_keys_pass: None,
            sort_histogram_pass: None,
            sort_scan_pass: None,
            sort_scatter_pass: None,
            idm_pass: None,
            mobil_pass: None,

            car_renderer: None,

            car_count: 0,
            sim_accumulator: 0.0,
            sim_tick: 0,
            initialized: false,
        }
    }
}

impl TrafficSim {
    pub fn create_pipelines(&mut self, ctx: &EngineContext) -> anyhow::Result<()> {
        let device = ctx.device.as_ref();

        // Car renderer (instanced rendering)
        {
            self.car_renderer = Some(CarRenderer::new(
                device,
                ctx.draw_image.format,
                CAR_RENDER_WGSL,
            )?);
        }

        // Sort: build keys (6 SSBOs)
        self.sort_keys_pass = Some(ComputePass::new(
            device,
            TRAFFIC_SORT_KEYS_WGSL,
            6,
            std::mem::size_of::<SortKeysPushConstants>() as u32,
            1,
        )?);

        // Sort: histogram (2 SSBOs, 2 sets for ping-pong)
        self.sort_histogram_pass = Some(ComputePass::new(
            device,
            TRAFFIC_SORT_HISTOGRAM_WGSL,
            2,
            std::mem::size_of::<SortHistogramPushConstants>() as u32,
            2,
        )?);

        // Sort: prefix sum scan (1 SSBO)
        self.sort_scan_pass = Some(ComputePass::new(
            device,
            TRAFFIC_SORT_SCAN_WGSL,
            1,
            std::mem::size_of::<SortScanPushConstants>() as u32,
            1,
        )?);

        // Sort: scatter (5 SSBOs, 2 sets for ping-pong)
        self.sort_scatter_pass = Some(ComputePass::new(
            device,
            TRAFFIC_SORT_SCATTER_WGSL,
            5,
            std::mem::size_of::<SortScatterPushConstants>() as u32,
            2,
        )?);

        // IDM car-following (7 SSBOs)
        self.idm_pass = Some(ComputePass::new(
            device,
            TRAFFIC_IDM_WGSL,
            7,
            std::mem::size_of::<IdmPushConstants>() as u32,
            1,
        )?);

        // MOBIL lane change (8 SSBOs: car SoA + road_lengths + sorted_indices + road_lane_counts)
        self.mobil_pass = Some(ComputePass::new(
            device,
            TRAFFIC_LANE_CHANGE_WGSL,
            8,
            std::mem::size_of::<MobilPushConstants>() as u32,
            1,
        )?);

        Ok(())
    }

    /// Allocate car SoA buffers and initialize with random cars on the given roads.
    pub fn initialize_cars(
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

        // Allocate car SoA buffers
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

            let total_lanes = right_lane_count + left_lane_count;
            if total_lanes > 0 {
                let pick = rng.random_range(0..total_lanes);
                if pick < right_lane_count {
                    lane_vals[i] = pick;
                } else {
                    lane_vals[i] = -(pick - right_lane_count + 1);
                }
            } else {
                lane_vals[i] = 0;
            }

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
        let hist_size = 256u64 * (NUM_SORT_WG as u64) * 4;
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

    /// Allocate car SoA buffers and initialize with explicit car placements from a scenario.
    pub fn initialize_cars_from_scenario(
        &mut self,
        allocator: &engine::vma::Allocator,
        network: &RoadNetwork,
        cars: &[crate::scenario::ScenarioCar],
    ) -> anyhow::Result<()> {
        self.destroy_buffers(allocator);

        if cars.is_empty() || network.roads.is_empty() {
            self.car_count = 0;
            self.initialized = false;
            return Ok(());
        }

        let total_cars = (cars.len() as u32).min(MAX_CARS);
        self.car_count = total_cars;

        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let n = total_cars as usize;

        let (road_id_buf, road_id_ptr) = GpuBuffer::new_mapped(allocator, (n * 4) as u64, usage)?;
        let (s_buf, s_ptr) = GpuBuffer::new_mapped(allocator, (n * 4) as u64, usage)?;
        let (lane_buf, lane_ptr) = GpuBuffer::new_mapped(allocator, (n * 4) as u64, usage)?;
        let (speed_buf, speed_ptr) = GpuBuffer::new_mapped(allocator, (n * 4) as u64, usage)?;
        let (desired_buf, desired_ptr) = GpuBuffer::new_mapped(allocator, (n * 4) as u64, usage)?;

        let road_ids = unsafe { std::slice::from_raw_parts_mut(road_id_ptr as *mut u32, n) };
        let s_vals = unsafe { std::slice::from_raw_parts_mut(s_ptr as *mut f32, n) };
        let lane_vals = unsafe { std::slice::from_raw_parts_mut(lane_ptr as *mut i32, n) };
        let speed_vals = unsafe { std::slice::from_raw_parts_mut(speed_ptr as *mut f32, n) };
        let desired_vals = unsafe { std::slice::from_raw_parts_mut(desired_ptr as *mut f32, n) };

        for (i, car) in cars.iter().take(n).enumerate() {
            road_ids[i] = car.road_id as u32;
            s_vals[i] = car.s;
            lane_vals[i] = car.lane;
            speed_vals[i] = car.speed;
            desired_vals[i] = car.desired_speed;
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

        let hist_size = 256u64 * (NUM_SORT_WG as u64) * 4;
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
    pub fn update_descriptors(
        &mut self,
        device: &engine::VkDevice,
        segment_buffer: Option<&GpuBuffer>,
        road_buffer: Option<&GpuBuffer>,
        lane_section_buffer: Option<&GpuBuffer>,
        lane_buffer: Option<&GpuBuffer>,
        road_lane_counts_buf: Option<&GpuBuffer>,
    ) {
        if !self.initialized {
            return;
        }

        // Unpack required buffers
        let car_road_id = match &self.car_road_id_buf { Some(b) => b, None => return };
        let car_s = match &self.car_s_buf { Some(b) => b, None => return };
        let car_lane = match &self.car_lane_buf { Some(b) => b, None => return };
        let car_speed = match &self.car_speed_buf { Some(b) => b, None => return };
        let car_desired = match &self.car_desired_speed_buf { Some(b) => b, None => return };
        let road_lengths = match &self.road_lengths_buf { Some(b) => b, None => return };

        // Car renderer descriptors (bindings 0-4: car buffers, 5-8: road data)
        if let (Some(seg_buf), Some(rd_buf), Some(ls_buf), Some(ln_buf)) = (
            segment_buffer, road_buffer, lane_section_buffer, lane_buffer,
        ) && let Some(renderer) = &mut self.car_renderer {
            renderer.update_descriptors(
                device, car_road_id, car_s, car_lane, car_speed, car_desired,
                seg_buf, rd_buf, ls_buf, ln_buf,
            );
        }

        // Sort buffers
        let sort_keys_a = match &self.sort_keys_a_buf { Some(b) => b, None => return };
        let sort_keys_b = match &self.sort_keys_b_buf { Some(b) => b, None => return };
        let sort_vals_a = match &self.sort_vals_a_buf { Some(b) => b, None => return };
        let sort_vals_b = match &self.sort_vals_b_buf { Some(b) => b, None => return };
        let hist_buf = match &self.sort_histogram_buf { Some(b) => b, None => return };

        // Sort keys: bindings 0-5 = car_road_id, car_s, car_lane, road_lengths, keys_a, vals_a
        if let Some(pass) = &self.sort_keys_pass {
            write_storage_buffers(device, pass.set(), 0, &[
                car_road_id, car_s, car_lane, road_lengths, sort_keys_a, sort_vals_a,
            ]);
        }

        // Sort histogram: set 0 reads A, set 1 reads B
        if let Some(pass) = &self.sort_histogram_pass {
            write_storage_buffers(device, pass.set_at(0), 0, &[sort_keys_a, hist_buf]);
            write_storage_buffers(device, pass.set_at(1), 0, &[sort_keys_b, hist_buf]);
        }

        // Sort scan: binding 0 = histogram
        if let Some(pass) = &self.sort_scan_pass {
            write_storage_buffers(device, pass.set(), 0, &[hist_buf]);
        }

        // Sort scatter: set 0 = A→B, set 1 = B→A
        if let Some(pass) = &self.sort_scatter_pass {
            write_storage_buffers(device, pass.set_at(0), 0, &[
                sort_keys_a, sort_vals_a, sort_keys_b, sort_vals_b, hist_buf,
            ]);
            write_storage_buffers(device, pass.set_at(1), 0, &[
                sort_keys_b, sort_vals_b, sort_keys_a, sort_vals_a, hist_buf,
            ]);
        }

        // IDM: bindings 0-6 = car SoA + road_lengths + sorted_indices
        // After 4 sort passes: final sorted indices are in vals_a
        if let Some(pass) = &self.idm_pass {
            write_storage_buffers(device, pass.set(), 0, &[
                car_road_id, car_s, car_lane, car_speed, car_desired, road_lengths, sort_vals_a,
            ]);
        }

        // MOBIL: bindings 0-7 = car SoA + road_lengths + sorted_indices + road_lane_counts
        if let Some(pass) = &self.mobil_pass
            && let Some(lane_counts) = road_lane_counts_buf
        {
            write_storage_buffers(device, pass.set(), 0, &[
                car_road_id, car_s, car_lane, car_speed, car_desired, road_lengths, sort_vals_a,
                lane_counts,
            ]);
        }
    }

    /// Record GPU commands for one simulation tick (sort → IDM → MOBIL).
    pub fn dispatch_tick(&self, device: &engine::VkDevice, cmd: vk::CommandBuffer) {
        let car_count = self.car_count;
        let num_sort_wg = car_count.div_ceil(SORT_TILE_SIZE);

        let sort_keys_pass = match &self.sort_keys_pass { Some(p) => p, None => return };
        let sort_histogram_pass = match &self.sort_histogram_pass { Some(p) => p, None => return };
        let sort_scan_pass = match &self.sort_scan_pass { Some(p) => p, None => return };
        let sort_scatter_pass = match &self.sort_scatter_pass { Some(p) => p, None => return };
        let idm_pass = match &self.idm_pass { Some(p) => p, None => return };

        let barrier = vk::MemoryBarrier2::builder()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE);
        let dep = vk::DependencyInfo::builder()
            .memory_barriers(std::slice::from_ref(&barrier));

        unsafe {
            // === Step 1: Build sort keys ===
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, sort_keys_pass.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, sort_keys_pass.pipeline_layout,
                0, &[sort_keys_pass.set()], &[],
            );
            let keys_pc = SortKeysPushConstants {
                car_count, _pad0: 0, _pad1: 0, _pad2: 0,
            };
            device.cmd_push_constants(
                cmd, sort_keys_pass.pipeline_layout, vk::ShaderStageFlags::COMPUTE,
                0, bytemuck::bytes_of(&keys_pc),
            );
            device.cmd_dispatch(cmd, num_sort_wg, 1, 1);
            device.cmd_pipeline_barrier2(cmd, &dep);

            // === Step 2: Radix sort (4 passes × 8 bits) ===
            let hist_buf = self.sort_histogram_buf.as_ref().unwrap();

            for pass in 0u32..4 {
                let set_idx = (pass % 2) as usize;

                // Zero histogram buffer
                device.cmd_fill_buffer(cmd, hist_buf.buffer, 0, hist_buf.size, 0);
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

                // Histogram
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, sort_histogram_pass.pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd, vk::PipelineBindPoint::COMPUTE, sort_histogram_pass.pipeline_layout,
                    0, &[sort_histogram_pass.set_at(set_idx)], &[],
                );
                let hist_pc = SortHistogramPushConstants {
                    pass_id: pass, car_count, num_workgroups: num_sort_wg, _pad: 0,
                };
                device.cmd_push_constants(
                    cmd, sort_histogram_pass.pipeline_layout, vk::ShaderStageFlags::COMPUTE,
                    0, bytemuck::bytes_of(&hist_pc),
                );
                device.cmd_dispatch(cmd, num_sort_wg, 1, 1);
                device.cmd_pipeline_barrier2(cmd, &dep);

                // Prefix sum (scan)
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, sort_scan_pass.pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd, vk::PipelineBindPoint::COMPUTE, sort_scan_pass.pipeline_layout,
                    0, &[sort_scan_pass.set()], &[],
                );
                let scan_pc = SortScanPushConstants {
                    num_workgroups: num_sort_wg, _pad0: 0, _pad1: 0, _pad2: 0,
                };
                device.cmd_push_constants(
                    cmd, sort_scan_pass.pipeline_layout, vk::ShaderStageFlags::COMPUTE,
                    0, bytemuck::bytes_of(&scan_pc),
                );
                device.cmd_dispatch(cmd, 1, 1, 1);
                device.cmd_pipeline_barrier2(cmd, &dep);

                // Scatter
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, sort_scatter_pass.pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd, vk::PipelineBindPoint::COMPUTE, sort_scatter_pass.pipeline_layout,
                    0, &[sort_scatter_pass.set_at(set_idx)], &[],
                );
                let scatter_pc = SortScatterPushConstants {
                    pass_id: pass, car_count, num_workgroups: num_sort_wg, _pad: 0,
                };
                device.cmd_push_constants(
                    cmd, sort_scatter_pass.pipeline_layout, vk::ShaderStageFlags::COMPUTE,
                    0, bytemuck::bytes_of(&scatter_pc),
                );
                device.cmd_dispatch(cmd, num_sort_wg, 1, 1);
                device.cmd_pipeline_barrier2(cmd, &dep);
            }

            // === Step 3: IDM car-following ===
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, idm_pass.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, idm_pass.pipeline_layout,
                0, &[idm_pass.set()], &[],
            );
            let idm_pc = IdmPushConstants {
                dt: SIM_DT,
                car_count,
                a_max: IDM_A_MAX,
                b_comfort: IDM_B_COMFORT,
                s0: IDM_S0,
                time_headway: IDM_TIME_HEADWAY,
                car_length: CAR_LENGTH,
                _pad: 0,
            };
            device.cmd_push_constants(
                cmd, idm_pass.pipeline_layout, vk::ShaderStageFlags::COMPUTE,
                0, bytemuck::bytes_of(&idm_pc),
            );
            device.cmd_dispatch(cmd, car_count.div_ceil(256), 1, 1);
            device.cmd_pipeline_barrier2(cmd, &dep);
        }
    }

    /// Record GPU commands for MOBIL lane change evaluation.
    pub fn dispatch_lane_change(&self, device: &engine::VkDevice, cmd: vk::CommandBuffer) {
        let mobil_pass = match &self.mobil_pass { Some(p) => p, None => return };

        let barrier = vk::MemoryBarrier2::builder()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE);
        let dep = vk::DependencyInfo::builder()
            .memory_barriers(std::slice::from_ref(&barrier));

        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, mobil_pass.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, mobil_pass.pipeline_layout,
                0, &[mobil_pass.set()], &[],
            );
            let mobil_pc = MobilPushConstants {
                car_count: self.car_count,
                a_max: IDM_A_MAX,
                b_comfort: IDM_B_COMFORT,
                s0: IDM_S0,
                time_headway: IDM_TIME_HEADWAY,
                car_length: CAR_LENGTH,
                politeness: MOBIL_POLITENESS,
                threshold: MOBIL_THRESHOLD,
                b_safe: MOBIL_B_SAFE,
                stagger_phase: self.sim_tick / LANE_CHANGE_INTERVAL,
                keep_right_bias: MOBIL_KEEP_RIGHT_BIAS,
                _pad: 0,
            };
            device.cmd_push_constants(
                cmd, mobil_pass.pipeline_layout, vk::ShaderStageFlags::COMPUTE,
                0, bytemuck::bytes_of(&mobil_pc),
            );
            device.cmd_dispatch(cmd, self.car_count.div_ceil(256), 1, 1);
            device.cmd_pipeline_barrier2(cmd, &dep);
        }
    }

    /// Draw cars using instanced rendering. Caller must have begun dynamic rendering.
    pub fn draw_cars(
        &self,
        device: &engine::VkDevice,
        cmd: vk::CommandBuffer,
        view_proj: [[f32; 4]; 4],
    ) {
        if !self.initialized || self.car_count == 0 {
            return;
        }
        if let Some(renderer) = &self.car_renderer
            && renderer.is_ready()
        {
            let pc = CarRenderPushConstants {
                view_proj,
                car_count: self.car_count,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            renderer.draw(device, cmd, &pc);
        }
    }

    /// Whether a MOBIL lane change should run this tick.
    pub fn should_lane_change(&self) -> bool {
        self.sim_tick.is_multiple_of(LANE_CHANGE_INTERVAL)
    }

    fn destroy_buffers(&mut self, allocator: &engine::vma::Allocator) {
        for buf in [
            &mut self.car_road_id_buf,
            &mut self.car_s_buf,
            &mut self.car_lane_buf,
            &mut self.car_speed_buf,
            &mut self.car_desired_speed_buf,
            &mut self.road_lengths_buf,
            &mut self.sort_keys_a_buf,
            &mut self.sort_keys_b_buf,
            &mut self.sort_vals_a_buf,
            &mut self.sort_vals_b_buf,
            &mut self.sort_histogram_buf,
        ] {
            if let Some(b) = buf.take() {
                b.destroy(allocator);
            }
        }
    }

    pub fn destroy(&mut self, device: &engine::VkDevice, allocator: &engine::vma::Allocator) {
        for pass in [
            &mut self.sort_keys_pass,
            &mut self.sort_histogram_pass,
            &mut self.sort_scan_pass,
            &mut self.sort_scatter_pass,
            &mut self.idm_pass,
            &mut self.mobil_pass,
        ] {
            if let Some(p) = pass.take() {
                p.destroy(device);
            }
        }
        if let Some(renderer) = self.car_renderer.take() {
            renderer.destroy(device);
        }
        self.destroy_buffers(allocator);
    }
}

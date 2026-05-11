mod gpu_road_data;
mod scenario;
mod traffic_sim;

use std::path::PathBuf;

use engine::gpu_pipeline_stats::GpuPipelineStats;
use engine::gpu_resources::GpuBuffer;
use engine::gpu_timestamps::GpuTimestamps;
use engine::pipeline::{
    GraphicsPipelineDesc, allocate_descriptor_set, create_compute_pipeline,
    create_descriptor_set_layout, create_graphics_pipeline, write_storage_buffers,
};
use engine::sdf::{
    ATLAS_TILES, RoadSegmentInfo, SdfTileManager, TILE_RESOLUTION, TILE_SIZE, compute_segment_aabbs,
};
use engine::vk::{DeviceV1_0, DeviceV1_3, Handle, HasBuilder};
use engine::{App, EngineContext, transition_image, vk};
use glam::Vec2;
use road::fitting::{ControlPoint, ReferenceLine};
use road::network::Road;
use road::network::RoadNetwork;

use crate::gpu_road_data::{GpuRoadData, segment_type_info};
use crate::traffic_sim::{SIM_DT, TrafficSim};

// ---------------------------------------------------------------------------
// Pre-compiled SPIR-V shaders (built by spirv-builder in build.rs)
// ---------------------------------------------------------------------------

pub const GRID_SPIRV: &[u8] = include_bytes!(env!("shaders.spv"));

pub fn spirv_bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// Push constants & vertex types
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraParams {
    inv_vp: [[f32; 4]; 4],
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LinePushConstants {
    view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RoadRenderPushConstants {
    view_proj: [[f32; 4]; 4],
    atlas_uv_offset: [f32; 2],
    atlas_uv_scale: [f32; 2],
    tile_world_origin: [f32; 2],
    tile_world_size: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LineVertex {
    position: [f32; 2],
    color: [f32; 4],
}

struct DrawRange {
    first_vertex: u32,
    vertex_count: u32,
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

    // Road rendering pipeline
    road_render_pipeline: vk::Pipeline,
    road_render_pipeline_layout: vk::PipelineLayout,
    road_render_descriptor_set_layout: vk::DescriptorSetLayout,
    road_render_descriptor_pool: vk::DescriptorPool,
    road_render_descriptor_set: vk::DescriptorSet,
    road_render_sampler: vk::Sampler,
    road_render_descriptors_valid: bool,

    // Traffic simulation
    traffic: TrafficSim,

    // Scenario-based car placement (overrides random init when Some)
    scenario_cars: Option<Vec<scenario::ScenarioCar>>,

    // Simulation speed multiplier (0 = paused, 0.5, 1.0, 2.0, 4.0)
    sim_speed: f32,

    // FPS tracking
    fps_accumulator: f32,
    fps_frame_count: u32,
    fps_display: f32,

    // GPU timestamp profiling
    gpu_timestamps: Option<GpuTimestamps>,
    gpu_pipeline_stats: Option<GpuPipelineStats>,
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

            scenario_cars: None,

            sim_speed: 1.0,
            fps_accumulator: 0.0,
            fps_frame_count: 0,
            fps_display: 0.0,
            gpu_timestamps: None,
            gpu_pipeline_stats: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline creation
// ---------------------------------------------------------------------------

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

        let spirv = spirv_bytes_to_words(GRID_SPIRV);

        let push_constant_ranges = [vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<CameraParams>() as u32)
            .build()];

        let (pipeline, layout) = create_compute_pipeline(
            device,
            &spirv,
            "grid::grid_main",
            &[self.grid_descriptor_set_layout],
            &push_constant_ranges,
        )?;

        self.grid_pipeline = pipeline;
        self.grid_pipeline_layout = layout;

        Ok(())
    }

    fn create_line_pipeline(&mut self, ctx: &EngineContext) -> anyhow::Result<()> {
        let device = ctx.device.as_ref();
        let spirv = spirv_bytes_to_words(GRID_SPIRV); // road_line entry points are in same module

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
            vertex_entry: "road_line::vs_main",
            fragment_entry: "road_line::fs_main",
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

    fn create_sdf_system(&mut self, ctx: &EngineContext) -> anyhow::Result<()> {
        let device = ctx.device.as_ref();

        let sdf_spirv = spirv_bytes_to_words(GRID_SPIRV); // sdf_generate entry point
        let sdf_manager = SdfTileManager::new(ctx.device, ctx.allocator, &sdf_spirv)?;

        let spirv = spirv_bytes_to_words(GRID_SPIRV); // road_render entry points

        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        self.road_render_sampler = unsafe { device.create_sampler(&sampler_info, None)? };

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

        // Write atlas image + sampler descriptors (constant)
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

        let push_constant_ranges = [vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<RoadRenderPushConstants>() as u32)
            .build()];

        let desc = GraphicsPipelineDesc {
            vertex_spirv: &spirv,
            fragment_spirv: &spirv,
            vertex_entry: "road_render::vs_main",
            fragment_entry: "road_render::fs_main",
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
}

// ---------------------------------------------------------------------------
// Descriptor updates
// ---------------------------------------------------------------------------

impl TrafficApp {
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

    fn update_road_render_descriptors(&mut self, ctx: &EngineContext) {
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

        write_storage_buffers(
            ctx.device.as_ref(),
            self.road_render_descriptor_set,
            2,
            &[road_buf, ls_buf, lane_buf],
        );
        self.road_render_descriptors_valid = true;
    }
}

// ---------------------------------------------------------------------------
// Road data management
// ---------------------------------------------------------------------------

impl TrafficApp {
    fn rebuild_sdf_tiles(&mut self, ctx: &EngineContext) -> anyhow::Result<()> {
        let sdf = match self.sdf_manager.as_mut() {
            Some(s) => s,
            None => return Ok(()),
        };

        let mut seg_infos = Vec::new();
        for (road_idx, road_data) in self.network.roads.iter().enumerate() {
            let rl = &road_data.reference_line;
            for (seg_idx, seg) in rl.segments.iter().enumerate() {
                let (seg_type, k_start, k_end) = segment_type_info(seg);

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

        let aabbs = compute_segment_aabbs(&seg_infos);
        sdf.tile_map.rebuild(&aabbs);

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

    fn handle_dirty_roads(&mut self, ctx: &EngineContext) -> anyhow::Result<()> {
        if !self.dirty {
            return Ok(());
        }

        unsafe {
            ctx.device.device_wait_idle().unwrap();
        }

        self.gpu_road_data.upload(ctx.allocator, &self.network)?;
        self.rebuild_sdf_tiles(ctx)?;
        self.update_road_render_descriptors(ctx);

        if let Some(cars) = &self.scenario_cars {
            // Scenario mode: use explicit car placements
            self.traffic
                .initialize_cars_from_scenario(ctx.allocator, &self.network, cars)?;
        } else {
            // Interactive mode: density-aware random car placement
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
                    let cars_this_road = ((road_len / 60.0) as u32) * lane_count;
                    total += cars_this_road.max(1);
                }
                total / self.network.roads.len().max(1) as u32
            };
            self.traffic
                .initialize_cars(ctx.allocator, &self.network, cars_per_road)?;
        }
        self.traffic.update_descriptors(
            ctx.device.as_ref(),
            self.gpu_road_data.segment_buffer.as_ref(),
            self.gpu_road_data.road_buffer.as_ref(),
            self.gpu_road_data.lane_section_buffer.as_ref(),
            self.gpu_road_data.lane_buffer.as_ref(),
            self.gpu_road_data.road_lane_counts_buf.as_ref(),
        );

        self.dirty = false;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Polyline rendering
// ---------------------------------------------------------------------------

impl TrafficApp {
    fn rebuild_polyline(&mut self, ctx: &EngineContext) -> anyhow::Result<()> {
        if let Some(b) = self.polyline_buffer.take() {
            self.retired_buffers.push(b);
        }
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

        for road_data in &self.network.roads {
            let rl = &road_data.reference_line;
            let sample_dist = 1.0;
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

        if self.pending_points.len() >= 2
            && let Some(rl) = ReferenceLine::fit(&self.pending_points)
        {
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
}

// ---------------------------------------------------------------------------
// Input handling
// ---------------------------------------------------------------------------

impl TrafficApp {
    fn process_input(&mut self, ctx: &EngineContext) -> bool {
        let mut needs_rebuild = false;

        let cursor_world = ctx.camera.screen_to_world(
            Vec2::new(ctx.input.mouse_x as f32, ctx.input.mouse_y as f32),
            Vec2::new(ctx.window_width as f32, ctx.window_height as f32),
        );

        if ctx.input.left_mouse_pressed {
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

        if ctx.input.left_mouse
            && let Some((ri, pi)) = self.dragging
            && ri < self.network.roads.len()
        {
            self.network.roads[ri].control_points[pi].position = cursor_world;
            if let Some(road_data) =
                Road::new_with_defaults(self.network.roads[ri].control_points.clone())
            {
                self.network.roads[ri] = road_data;
                self.dirty = true;
                needs_rebuild = true;
            }
        }

        if !ctx.input.left_mouse {
            self.dragging = None;
        }

        if ctx.input.right_mouse_pressed && self.pending_points.len() >= 2 {
            let pts = std::mem::take(&mut self.pending_points);
            if let Some(road_data) = Road::new_with_defaults(pts) {
                self.network.add_road(road_data);
                self.dirty = true;
                needs_rebuild = true;
            }
        }

        if ctx.input.escape_pressed && !self.pending_points.is_empty() {
            self.pending_points.clear();
            needs_rebuild = true;
        }

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

// ---------------------------------------------------------------------------
// Render sub-phases
// ---------------------------------------------------------------------------

/// GPU timestamp phase names (order must match write_phase calls in render()).
const GPU_PHASE_NAMES: &[&str] = &["sdf", "grid", "sort", "idm", "mobil", "draw"];

impl TrafficApp {
    fn dispatch_sdf(&mut self, device: &engine::VkDevice, cmd: vk::CommandBuffer) {
        let sdf = match self.sdf_manager.as_mut() {
            Some(s) => s,
            None => return,
        };
        if !sdf.has_dirty_tiles() {
            return;
        }

        transition_image(
            device,
            cmd,
            sdf.atlas.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );

        sdf.dispatch_dirty_tiles(device, cmd);

        unsafe {
            let barrier = vk::MemoryBarrier2::builder()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ);
            let dep = vk::DependencyInfo::builder().memory_barriers(std::slice::from_ref(&barrier));
            device.cmd_pipeline_barrier2(cmd, &dep);
        }

        transition_image(
            device,
            cmd,
            sdf.atlas.image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
    }

    fn dispatch_grid(&self, ctx: &EngineContext, cmd: vk::CommandBuffer) {
        let device = ctx.device.as_ref();
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
                width: ctx.window_width,
                height: ctx.window_height,
                _pad0: 0,
                _pad1: 0,
            };
            device.cmd_push_constants(
                cmd,
                self.grid_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&params),
            );

            let wg_x = ctx.window_width.div_ceil(16);
            let wg_y = ctx.window_height.div_ceil(16);
            device.cmd_dispatch(cmd, wg_x, wg_y, 1);
        }
    }

    fn dispatch_traffic(&mut self, device: &engine::VkDevice, cmd: vk::CommandBuffer, dt: f32) {
        if !self.traffic.initialized || self.traffic.car_count == 0 {
            return;
        }

        self.traffic.sim_accumulator += dt * self.sim_speed;
        let max_ticks = (2.0 * self.sim_speed).ceil().max(2.0) as u32;
        let mut ticks_this_frame = 0u32;

        while self.traffic.sim_accumulator >= SIM_DT && ticks_this_frame < max_ticks {
            self.traffic.sim_accumulator -= SIM_DT;
            self.traffic.sim_tick += 1;
            ticks_this_frame += 1;

            // Only timestamp the first tick per frame to stay within query pool bounds
            let stamp = ticks_this_frame == 1;

            // Sort (keys + radix sort)
            self.traffic.dispatch_sort(device, cmd);
            if stamp {
                if let Some(ts) = &mut self.gpu_timestamps {
                    ts.write_phase(device, cmd);
                }
            }

            // IDM car-following
            self.traffic.dispatch_idm(device, cmd);
            if stamp {
                if let Some(ts) = &mut self.gpu_timestamps {
                    ts.write_phase(device, cmd);
                }
            }

            // MOBIL lane change
            if self.traffic.should_lane_change() {
                self.traffic.dispatch_lane_change(device, cmd);
            }
            if stamp {
                if let Some(ts) = &mut self.gpu_timestamps {
                    ts.write_phase(device, cmd);
                }
            }
        }

        // Clamp accumulator to prevent unbounded growth
        if self.traffic.sim_accumulator > SIM_DT * 4.0 {
            self.traffic.sim_accumulator = 0.0;
        }

        // Final barrier: compute writes → vertex shader reads
        if ticks_this_frame > 0 {
            unsafe {
                let barrier = vk::MemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ);
                let dep =
                    vk::DependencyInfo::builder().memory_barriers(std::slice::from_ref(&barrier));
                device.cmd_pipeline_barrier2(cmd, &dep);
            }
        }
    }

    fn draw_scene(&self, ctx: &EngineContext, cmd: vk::CommandBuffer) {
        let device = ctx.device.as_ref();

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

            self.draw_road_tiles(device, cmd, &vp);
            self.draw_polylines(device, cmd, &vp);
            self.traffic.draw_cars(device, cmd, vp.to_cols_array_2d());

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

    fn draw_road_tiles(&self, device: &engine::VkDevice, cmd: vk::CommandBuffer, vp: &glam::Mat4) {
        let sdf = match &self.sdf_manager {
            Some(s) => s,
            None => return,
        };
        if sdf.tile_map.tiles.is_empty() || !self.road_render_descriptors_valid {
            return;
        }

        unsafe {
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
        }

        let atlas_tiles = ATLAS_TILES;
        let atlas_size_f = (atlas_tiles * TILE_RESOLUTION) as f32;
        let half_texel = 0.5 / atlas_size_f;

        for (key, info) in &sdf.tile_map.tiles {
            let slot_x = info.atlas_slot % atlas_tiles;
            let slot_y = info.atlas_slot / atlas_tiles;
            let uv_offset_x = (slot_x * TILE_RESOLUTION) as f32 / atlas_size_f + half_texel;
            let uv_offset_y = (slot_y * TILE_RESOLUTION) as f32 / atlas_size_f + half_texel;
            let uv_scale = TILE_RESOLUTION as f32 / atlas_size_f - 2.0 * half_texel;

            let (wx, wy) = key.world_origin();

            let pc = RoadRenderPushConstants {
                view_proj: vp.to_cols_array_2d(),
                atlas_uv_offset: [uv_offset_x, uv_offset_y],
                atlas_uv_scale: [uv_scale, uv_scale],
                tile_world_origin: [wx, wy],
                tile_world_size: [TILE_SIZE, TILE_SIZE],
            };

            unsafe {
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

    fn draw_polylines(&self, device: &engine::VkDevice, cmd: vk::CommandBuffer, vp: &glam::Mat4) {
        let vb = match &self.polyline_buffer {
            Some(b) => b,
            None => return,
        };

        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.line_pipeline);

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

            device.cmd_bind_vertex_buffers(cmd, 0, &[vb.buffer], &[0]);

            for draw in &self.road_draws {
                device.cmd_draw(cmd, draw.vertex_count, 1, draw.first_vertex, 0);
            }

            if let Some(ref draw) = self.pending_draw {
                device.cmd_draw(cmd, draw.vertex_count, 1, draw.first_vertex, 0);
            }

            if let Some(ref draw) = self.cp_draw {
                let mut v = draw.first_vertex;
                let end = draw.first_vertex + draw.vertex_count;
                while v + 1 < end {
                    device.cmd_draw(cmd, 2, 1, v, 0);
                    v += 2;
                }
            }
        }
    }

    fn update_hud(&mut self, ctx: &EngineContext) {
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
                format!(" | GPU: {}", ts.format_phases())
            } else {
                String::new()
            };
            let stats_info = if let Some(ps) = &self.gpu_pipeline_stats {
                format!(" | {}", ps.format())
            } else {
                String::new()
            };
            window.set_title(&format!(
                "Traffic Sim | {:.0} FPS | {} cars | {}{}{} | [Space]=pause [1-4]=speed",
                self.fps_display, self.traffic.car_count, speed_label, gpu_info, stats_info,
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// App trait implementation
// ---------------------------------------------------------------------------

impl App for TrafficApp {
    fn init(&mut self, ctx: &EngineContext) -> anyhow::Result<()> {
        // Initialize Vulkan pipeline cache (loads previous cache from disk)
        engine::init_pipeline_cache(ctx.device.as_ref())?;

        self.create_grid_pipeline(ctx)?;
        self.create_line_pipeline(ctx)?;
        self.create_sdf_system(ctx)?;
        self.traffic.create_pipelines(ctx)?;
        self.gpu_timestamps = Some(GpuTimestamps::new(
            ctx.device.as_ref(),
            ctx.timestamp_period,
            GPU_PHASE_NAMES,
        )?);
        self.gpu_pipeline_stats = Some(GpuPipelineStats::new(ctx.device.as_ref())?);

        // If a scenario was loaded before init, mark dirty to trigger road upload + car init
        if !self.network.roads.is_empty() {
            self.dirty = true;
        }

        Ok(())
    }

    fn resize(&mut self, ctx: &EngineContext) -> anyhow::Result<()> {
        self.update_grid_descriptors(ctx);
        Ok(())
    }

    fn render(&mut self, ctx: &EngineContext, cmd: vk::CommandBuffer) -> anyhow::Result<()> {
        let device = ctx.device.as_ref();

        // Read previous frame's GPU timestamps, begin new frame
        if let Some(ts) = &mut self.gpu_timestamps {
            ts.read_results(device);
            ts.reset_and_begin(device, cmd);
        }
        if let Some(ps) = &mut self.gpu_pipeline_stats {
            ps.read_results(device);
            ps.begin(device, cmd);
        }

        // Input & dirty road handling
        let needs_rebuild = self.process_input(ctx);
        if needs_rebuild {
            self.rebuild_polyline(ctx)?;
        }
        self.handle_dirty_roads(ctx)?;

        // Phase: SDF tile generation
        self.dispatch_sdf(device, cmd);
        if let Some(ts) = &mut self.gpu_timestamps {
            ts.write_phase(device, cmd);
        }

        // Phase: Grid background
        self.dispatch_grid(ctx, cmd);
        if let Some(ts) = &mut self.gpu_timestamps {
            ts.write_phase(device, cmd);
        }

        // Phase: Traffic simulation (sort → IDM → MOBIL) — writes sub-phase timestamps internally
        self.dispatch_traffic(device, cmd, ctx.dt);

        // Phase: Scene rendering (roads, polylines, cars)
        self.draw_scene(ctx, cmd);
        if let Some(ts) = &mut self.gpu_timestamps {
            ts.write_phase(device, cmd);
        }

        // End pipeline stats query before HUD
        if let Some(ps) = &self.gpu_pipeline_stats {
            ps.end(device, cmd);
        }

        // HUD
        self.update_hud(ctx);

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
        if let Some(ps) = self.gpu_pipeline_stats.take() {
            ps.destroy(ctx.device.as_ref());
        }
        // Save and destroy Vulkan pipeline cache
        engine::destroy_pipeline_cache(ctx.device.as_ref());
    }
}

fn main() -> anyhow::Result<()> {
    let mut app = TrafficApp::default();

    // Parse --scenario <path> argument
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--scenario") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--scenario requires a file path"))?;
        let path = PathBuf::from(path);
        let scenario = scenario::Scenario::load(&path)?;
        let network = scenario.build_network()?;
        scenario.validate(&network)?;
        log::info!(
            "Loaded scenario: {} roads, {} cars",
            network.roads.len(),
            scenario.cars.len()
        );
        app.network = network;
        app.scenario_cars = Some(scenario.cars);
        app.dirty = true;
    }

    engine::run(app)
}

use engine::gpu_resources::GpuBuffer;
use engine::pipeline::{
    GraphicsPipelineDesc, allocate_descriptor_set, compile_wgsl, create_compute_pipeline,
    create_descriptor_set_layout, create_graphics_pipeline,
};
use engine::vk::{DeviceV1_0, DeviceV1_3, Handle, HasBuilder};
use engine::{App, EngineContext, transition_image, vk};
use glam::Vec2;
use road::fitting::{ControlPoint, ReferenceLine};
use road::network::Road;
use road::network::RoadNetwork;

const GRID_SHADER: &str = include_str!("../../../assets/shaders/grid.wgsl");
const ROAD_LINE_SHADER: &str = include_str!("../../../assets/shaders/road_line.wgsl");

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
            let hit_radius = 2.0 / ctx.camera.zoom;
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
        Ok(())
    }

    fn resize(&mut self, ctx: &EngineContext) -> anyhow::Result<()> {
        self.update_grid_descriptors(ctx);
        Ok(())
    }

    fn render(&mut self, ctx: &EngineContext, cmd: vk::CommandBuffer) -> anyhow::Result<()> {
        let device = ctx.device.as_ref();

        // Process input
        let needs_rebuild = self.process_input(ctx);
        if needs_rebuild {
            self.rebuild_polyline(ctx)?;
        }

        // Upload GPU road data if dirty
        if self.dirty {
            self.gpu_road_data.upload(ctx.allocator, &self.network)?;
            self.dirty = false;
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

        // --- Graphics: road polylines ---
        if self.polyline_buffer.is_some() {
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

                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.line_pipeline);

                let aspect = ctx.window_width as f32 / ctx.window_height as f32;
                let vp = ctx.camera.view_projection(aspect);
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
        }
        if let Some(b) = self.polyline_buffer.take() {
            b.destroy(ctx.allocator);
        }
        for b in self.retired_buffers.drain(..) {
            b.destroy(ctx.allocator);
        }
        self.gpu_road_data.destroy(ctx.allocator);
    }
}

fn main() -> anyhow::Result<()> {
    engine::run(TrafficApp::default())
}

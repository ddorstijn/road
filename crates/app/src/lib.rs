use crate::road_constructor::{GeometryType, generate_road_geometry};
use engine::{component::GameComponent, engine::VulkanContext};
use glam::Vec2;
use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo,
        SubpassContents,
    },
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    format::Format,
    image::{
        Image, ImageCreateInfo, ImageType, ImageUsage,
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        ComputePipeline, DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint,
        PipelineLayout, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Scissor, Viewport, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
};

mod road_constructor;

// --- Shaders ---
mod cs {
    vulkano_shaders::shader! { ty: "compute", path: "assets/shaders/sdf.comp" }
}
mod vs {
    vulkano_shaders::shader! { ty: "vertex", path: "assets/shaders/road.vert" }
}
mod fs {
    vulkano_shaders::shader! { ty: "fragment", path: "assets/shaders/road.frag" }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct QuadVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
    #[format(R32G32_SFLOAT)]
    uv: [f32; 2],
}

struct WorldQuad {
    texture: Arc<ImageView>,
    origin: Vec2,
    size: Vec2,
    screen_pos: Vec2,
}

pub struct App {
    compute_pipeline: Option<Arc<ComputePipeline>>,
    graphics_pipeline: Option<Arc<GraphicsPipeline>>,
    render_pass: Option<Arc<RenderPass>>,

    road_buffer: Option<Subbuffer<[cs::RoadSegment]>>,
    vertex_buffer: Option<Subbuffer<[QuadVertex]>>,

    quads: Vec<WorldQuad>,
    sampler: Option<Arc<Sampler>>,

    // Flag to ensure we only compute once
    sdf_generated: bool,
}

impl Default for App {
    fn default() -> Self {
        Self {
            compute_pipeline: None,
            graphics_pipeline: None,
            render_pass: None,
            road_buffer: None,
            vertex_buffer: None,
            quads: Vec::new(),
            sampler: None,
            sdf_generated: false,
        }
    }
}

impl GameComponent for App {
    fn start(&mut self, ctx: &VulkanContext) {
        // 1. SETUP COMPUTE PIPELINE
        let cs = cs::load(ctx.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let stage_c = PipelineShaderStageCreateInfo::new(cs);
        let layout_c = PipelineLayout::new(
            ctx.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage_c])
                .into_pipeline_layout_create_info(ctx.device.clone())
                .unwrap(),
        )
        .unwrap();
        self.compute_pipeline = Some(
            ComputePipeline::new(
                ctx.device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage_c, layout_c),
            )
            .unwrap(),
        );

        // 2. SETUP DATA
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(100.0, 0.0),
            Vec2::new(100.0, 100.0),
            Vec2::new(200.0, 100.0),
        ];
        let raw_road = generate_road_geometry(&points, 30.0, 20.0);
        let mut total_s = 0.0;

        self.road_buffer = Some(
            Buffer::from_iter(
                ctx.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                raw_road.iter().enumerate().map(|(i, seg)| {
                    let (k0, k1, len) = match seg.geometry {
                        GeometryType::Line { length } => (0.0, 0.0, length),
                        GeometryType::Arc { length, curvature } => (curvature, curvature, length),
                        GeometryType::Spiral {
                            length,
                            curv_start,
                            curv_end,
                        } => (curv_start, curv_end, length),
                    };
                    let s = cs::RoadSegment {
                        start: [seg.start_x, seg.start_y],
                        hdg: seg.start_hdg,
                        len,
                        k0,
                        k1,
                        s_offset: total_s,
                        road_id: i as i32,
                    };
                    total_s += len;
                    s
                }),
            )
            .unwrap(),
        );

        // 3. SETUP QUADS (Textures)
        let texture_size = 512;

        let create_quad = |origin: Vec2, size: Vec2, screen_x: f32| -> WorldQuad {
            let image = Image::new(
                ctx.memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    // Use R32G32B32A32_SFLOAT for high precision distance/s-coords
                    format: Format::R32G32B32A32_SFLOAT,
                    extent: [texture_size, texture_size, 1],
                    usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap();

            WorldQuad {
                texture: ImageView::new_default(image).unwrap(),
                origin,
                size,
                screen_pos: Vec2::new(screen_x, 0.0),
            }
        };

        self.quads.push(create_quad(
            Vec2::new(0.0, -50.0),
            Vec2::new(120.0, 200.0),
            -0.5,
        ));
        self.quads.push(create_quad(
            Vec2::new(120.0, -50.0),
            Vec2::new(120.0, 200.0),
            0.5,
        ));

        // 4. SETUP GRAPHICS
        let vs = vs::load(ctx.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = fs::load(ctx.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let render_pass = vulkano::single_pass_renderpass!(
            ctx.device.clone(),
            attachments: {
                color: {
                    format: ctx.draw_image.format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap();
        self.render_pass = Some(render_pass.clone());

        self.sampler = Some(
            Sampler::new(
                ctx.device.clone(),
                SamplerCreateInfo {
                    mag_filter: Filter::Linear,
                    min_filter: Filter::Linear,
                    address_mode: [SamplerAddressMode::ClampToEdge; 3],
                    ..Default::default()
                },
            )
            .unwrap(),
        );

        self.vertex_buffer = Some(
            Buffer::from_iter(
                ctx.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                vec![
                    QuadVertex {
                        position: [-1.0, -1.0],
                        uv: [0.0, 0.0],
                    },
                    QuadVertex {
                        position: [-1.0, 1.0],
                        uv: [0.0, 1.0],
                    },
                    QuadVertex {
                        position: [1.0, -1.0],
                        uv: [1.0, 0.0],
                    },
                    QuadVertex {
                        position: [1.0, 1.0],
                        uv: [1.0, 1.0],
                    },
                    QuadVertex {
                        position: [-1.0, 1.0],
                        uv: [0.0, 1.0],
                    },
                    QuadVertex {
                        position: [1.0, -1.0],
                        uv: [1.0, 0.0],
                    },
                ],
            )
            .unwrap(),
        );

        let stage_vs = PipelineShaderStageCreateInfo::new(vs.clone());
        let stage_fs = PipelineShaderStageCreateInfo::new(fs.clone());
        let layout_g = PipelineLayout::new(
            ctx.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage_vs, &stage_fs])
                .into_pipeline_layout_create_info(ctx.device.clone())
                .unwrap(),
        )
        .unwrap();

        self.graphics_pipeline = Some(
            GraphicsPipeline::new(
                ctx.device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: vec![stage_vs, stage_fs].into_iter().collect(),
                    vertex_input_state: Some(
                        QuadVertex::per_vertex().definition(&vs.clone()).unwrap(),
                    ),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState {
                        viewports: [Viewport::default()].into_iter().collect(),
                        scissors: [Scissor::default()].into_iter().collect(),
                        ..Default::default()
                    }),
                    dynamic_state: [DynamicState::Viewport, DynamicState::Scissor]
                        .into_iter()
                        .collect(),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        render_pass.clone().first_subpass().num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    subpass: Some(Subpass::from(render_pass, 0).unwrap().into()),
                    ..GraphicsPipelineCreateInfo::layout(layout_g)
                },
            )
            .unwrap(),
        );
    }

    fn ui(&mut self, _ctx: &VulkanContext, _ui: &mut imgui::Ui) {}

    fn render(
        &mut self,
        ctx: &VulkanContext,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        // --- PASS 1: COMPUTE (Run ONCE) ---
        if !self.sdf_generated {
            if let (Some(cp), Some(buf)) = (&self.compute_pipeline, &self.road_buffer) {
                for quad in &self.quads {
                    let layout = &cp.layout().set_layouts()[0];
                    let set = DescriptorSet::new(
                        ctx.descriptor_set_allocator.clone(),
                        layout.clone(),
                        [
                            WriteDescriptorSet::image_view(0, quad.texture.clone()),
                            WriteDescriptorSet::buffer(1, buf.clone()),
                        ],
                        [],
                    )
                    .unwrap();

                    let push_consts = cs::PushConstants {
                        quad_origin: [quad.origin.x, quad.origin.y],
                        quad_size: [quad.size.x, quad.size.y],
                    };

                    let [w, h, _] = quad.texture.image().extent();

                    unsafe {
                        builder
                            .bind_pipeline_compute(cp.clone())
                            .unwrap()
                            .bind_descriptor_sets(
                                PipelineBindPoint::Compute,
                                cp.layout().clone(),
                                0,
                                set,
                            )
                            .unwrap()
                            .push_constants(cp.layout().clone(), 0, push_consts)
                            .unwrap()
                            .dispatch([(w + 15) / 16, (h + 15) / 16, 1])
                            .unwrap();
                    }
                }
            }
            // Mark as generated so we don't run this again
            self.sdf_generated = true;
        }

        // --- PASS 2: GRAPHICS (Every Frame) ---
        if let (Some(gp), Some(rp), Some(vb), Some(samp)) = (
            &self.graphics_pipeline,
            &self.render_pass,
            &self.vertex_buffer,
            &self.sampler,
        ) {
            let [w, h, _] = ctx.draw_image.image().extent();

            let fb = Framebuffer::new(
                rp.clone(),
                FramebufferCreateInfo {
                    attachments: vec![ctx.draw_image.clone()],
                    ..Default::default()
                },
            )
            .unwrap();

            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
                        ..RenderPassBeginInfo::framebuffer(fb)
                    },
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                )
                .unwrap()
                .bind_pipeline_graphics(gp.clone())
                .unwrap()
                .bind_vertex_buffers(0, vb.clone())
                .unwrap();

            for quad in &self.quads {
                // Bind the TEXTURE we generated in Pass 1
                let layout = &gp.layout().set_layouts()[0];
                let set = DescriptorSet::new(
                    ctx.descriptor_set_allocator.clone(),
                    layout.clone(),
                    [WriteDescriptorSet::image_view_sampler(
                        0,
                        quad.texture.clone(),
                        samp.clone(),
                    )],
                    [],
                )
                .unwrap();

                // Simple Viewport positioning for demo
                let vp_x = (quad.screen_pos.x * 0.5 + 0.5) * (w as f32);
                let vp_w = (w as f32) / 2.0;

                unsafe {
                    builder
                        .set_viewport(
                            0,
                            [Viewport {
                                offset: [vp_x, 0.0],
                                extent: [vp_w, h as f32],
                                depth_range: 0.0..=1.0,
                            }]
                            .into_iter()
                            .collect(),
                        )
                        .unwrap()
                        .set_scissor(
                            0,
                            [Scissor {
                                offset: [vp_x as u32, 0],
                                extent: [vp_w as u32, h],
                            }]
                            .into_iter()
                            .collect(),
                        )
                        .unwrap()
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            gp.layout().clone(),
                            0,
                            set,
                        )
                        .unwrap()
                        .draw(6, 1, 0, 0)
                        .unwrap();
                }
            }

            builder.end_render_pass(Default::default()).unwrap();
        }
    }
}

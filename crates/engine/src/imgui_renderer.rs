use std::collections::HashMap;
use std::sync::Arc;

use vulkano::{
    buffer::{
        Buffer, BufferContents, BufferCreateInfo, BufferUsage,
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
    },
    command_buffer::{
        AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::{Device, Queue},
    format::Format,
    image::{
        Image, ImageCreateInfo, ImageUsage,
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{
                AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
            },
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::CullMode,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Scissor, Viewport, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    sync::GpuFuture,
};

// --- Vertex Definition ---
#[derive(BufferContents, Vertex, Clone, Copy, Debug)]
#[repr(C)]
struct ImGuiVertex {
    #[format(R32G32_SFLOAT)]
    pos: [f32; 2],
    #[format(R32G32_SFLOAT)]
    uv: [f32; 2],
    #[format(R8G8B8A8_UNORM)]
    col: u32,
}

// --- Shaders (Unchanged) ---
mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450
            layout(push_constant) uniform PushConsts {
                vec2 uScale;
                vec2 uTranslate;
            } pc;
            
            layout(location = 0) in vec2 pos;
            layout(location = 1) in vec2 uv;
            layout(location = 2) in vec4 col;
            
            layout(location = 0) out vec2 out_uv;
            layout(location = 1) out vec4 out_col;
            
            void main() {
                out_uv = uv;
                out_col = col;
                gl_Position = vec4(pos * pc.uScale + pc.uTranslate, 0.0, 1.0);
            }
        "
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450
            layout(set = 0, binding = 0) uniform sampler2D tex;
            
            layout(location = 0) in vec2 in_uv;
            layout(location = 1) in vec4 in_col;
            
            layout(location = 0) out vec4 out_color;
            
            void main() {
                out_color = in_col * texture(tex, in_uv);
            }
        "
    }
}

pub struct ImguiVulkanoRenderer {
    pipeline: Arc<GraphicsPipeline>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,

    vertex_allocator: SubbufferAllocator,
    index_allocator: SubbufferAllocator,

    textures: HashMap<imgui::TextureId, Arc<DescriptorSet>>,
    font_texture_id: imgui::TextureId,

    vertex_buffer_host: Vec<ImGuiVertex>,
    index_buffer_host: Vec<u16>,
}

impl ImguiVulkanoRenderer {
    pub fn new(
        imgui: &mut imgui::Context,
        device: Arc<Device>,
        queue: Arc<Queue>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        output_format: Format,
    ) -> Self {
        let fonts = imgui.fonts();
        let font_texture_id = fonts.tex_id;
        let texture_data = fonts.build_rgba32_texture();

        let font_image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: vulkano::image::ImageType::Dim2d,
                format: Format::R8G8B8A8_UNORM,
                extent: [texture_data.width, texture_data.height, 1],
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();

        let upload_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            texture_data.data.iter().cloned(),
        )
        .unwrap();

        let mut cbb = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            queue.queue_family_index(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        cbb.copy_buffer_to_image(
            vulkano::command_buffer::CopyBufferToImageInfo::buffer_image(
                upload_buffer,
                font_image.clone(),
            ),
        )
        .unwrap();

        cbb.build()
            .unwrap()
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        let font_view = ImageView::new_default(font_image).unwrap();
        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )
        .unwrap();

        let vs = vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let pipeline = {
            let stage_vs = PipelineShaderStageCreateInfo::new(vs.clone());
            let stage_fs = PipelineShaderStageCreateInfo::new(fs.clone());

            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage_vs, &stage_fs])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            let subpass = vulkano::pipeline::graphics::subpass::PipelineRenderingCreateInfo {
                color_attachment_formats: vec![Some(output_format)],
                ..Default::default()
            };

            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: vec![stage_vs, stage_fs].into_iter().collect(),
                    dynamic_state: [DynamicState::Viewport, DynamicState::Scissor]
                        .into_iter()
                        .collect(),
                    vertex_input_state: Some(ImGuiVertex::per_vertex().definition(&vs).unwrap()),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(
                        vulkano::pipeline::graphics::rasterization::RasterizationState {
                            cull_mode: CullMode::None,
                            ..Default::default()
                        },
                    ),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.color_attachment_formats.len() as u32,
                        ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend {
                                src_color_blend_factor: BlendFactor::SrcAlpha,
                                dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
                                color_blend_op: BlendOp::Add,
                                src_alpha_blend_factor: BlendFactor::OneMinusSrcAlpha,
                                dst_alpha_blend_factor: BlendFactor::Zero,
                                alpha_blend_op: BlendOp::Add,
                            }),
                            ..Default::default()
                        },
                    )),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        let font_descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            pipeline.layout().set_layouts().get(0).unwrap().clone(),
            [WriteDescriptorSet::image_view_sampler(
                0, font_view, sampler,
            )],
            [],
        )
        .unwrap();

        let mut textures = HashMap::new();
        textures.insert(font_texture_id, font_descriptor_set);

        let vertex_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::VERTEX_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let index_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::INDEX_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        Self {
            pipeline,
            descriptor_set_allocator,
            vertex_allocator,
            index_allocator,
            textures,
            font_texture_id,
            vertex_buffer_host: Vec::new(),
            index_buffer_host: Vec::new(),
        }
    }

    #[allow(unused)]
    pub fn register_texture(
        &mut self,
        view: Arc<ImageView>,
        sampler: Arc<Sampler>,
        id: imgui::TextureId,
    ) {
        let set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts().get(0).unwrap().clone(),
            [WriteDescriptorSet::image_view_sampler(0, view, sampler)],
            [],
        )
        .unwrap();

        self.textures.insert(id, set);
    }

    pub fn draw(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        draw_data: &imgui::DrawData,
    ) {
        if draw_data.total_vtx_count == 0 {
            return;
        }

        self.vertex_buffer_host.clear();
        self.index_buffer_host.clear();
        self.vertex_buffer_host
            .reserve(draw_data.total_vtx_count as usize);
        self.index_buffer_host
            .reserve(draw_data.total_idx_count as usize);

        for draw_list in draw_data.draw_lists() {
            for v in draw_list.vtx_buffer() {
                self.vertex_buffer_host.push(ImGuiVertex {
                    pos: v.pos,
                    uv: v.uv,
                    col: u32::from_ne_bytes(v.col),
                });
            }
            self.index_buffer_host
                .extend_from_slice(draw_list.idx_buffer());
        }

        let vertex_subbuffer = self
            .vertex_allocator
            .allocate_slice(self.vertex_buffer_host.len() as u64)
            .expect("Failed to allocate vertex buffer");

        {
            let mut buffer_content = vertex_subbuffer.write().unwrap();
            buffer_content.copy_from_slice(&self.vertex_buffer_host);
        }

        let index_subbuffer = self
            .index_allocator
            .allocate_slice(self.index_buffer_host.len() as u64)
            .expect("Failed to allocate index buffer");

        {
            let mut buffer_content = index_subbuffer.write().unwrap();
            buffer_content.copy_from_slice(&self.index_buffer_host);
        }

        let [width, height] = draw_data.display_size;
        let [scale_w, scale_h] = draw_data.framebuffer_scale;

        let push_consts = vs::PushConsts {
            uScale: [2.0 / width, 2.0 / height],
            uTranslate: [-1.0, -1.0],
        };

        builder
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap();
        builder.bind_vertex_buffers(0, vertex_subbuffer).unwrap();
        builder.bind_index_buffer(index_subbuffer).unwrap();
        builder
            .push_constants(self.pipeline.layout().clone(), 0, push_consts)
            .unwrap();

        let fb_width = width * scale_w;
        let fb_height = height * scale_h;

        builder
            .set_viewport(
                0,
                [Viewport {
                    offset: [0.0, 0.0],
                    extent: [fb_width, fb_height],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )
            .unwrap();

        // 4. Render Loop
        let mut index_offset = 0;
        let mut vertex_offset = 0;
        let clip_off = draw_data.display_pos;
        let mut current_texture_id = None;

        for draw_list in draw_data.draw_lists() {
            for cmd in draw_list.commands() {
                match cmd {
                    imgui::DrawCmd::Elements { count, cmd_params } => {
                        let clip_rect = cmd_params.clip_rect;

                        // Optimize Texture Binding
                        if current_texture_id != Some(cmd_params.texture_id) {
                            let set = self
                                .textures
                                .get(&cmd_params.texture_id)
                                .or_else(|| self.textures.get(&self.font_texture_id))
                                .expect("Texture not registered");

                            builder
                                .bind_descriptor_sets(
                                    PipelineBindPoint::Graphics,
                                    self.pipeline.layout().clone(),
                                    0,
                                    set.clone(),
                                )
                                .unwrap();

                            current_texture_id = Some(cmd_params.texture_id);
                        }

                        // Calculate Scissor Rect
                        let clip_min_x = (clip_rect[0] - clip_off[0]) * scale_w;
                        let clip_min_y = (clip_rect[1] - clip_off[1]) * scale_h;
                        let clip_max_x = (clip_rect[2] - clip_off[0]) * scale_w;
                        let clip_max_y = (clip_rect[3] - clip_off[1]) * scale_h;

                        // Clamp
                        let clip_min_x = clip_min_x.max(0.0).min(fb_width);
                        let clip_min_y = clip_min_y.max(0.0).min(fb_height);
                        let clip_max_x = clip_max_x.max(0.0).min(fb_width);
                        let clip_max_y = clip_max_y.max(0.0).min(fb_height);

                        if clip_max_x <= clip_min_x || clip_max_y <= clip_min_y {
                            continue;
                        }

                        builder
                            .set_scissor(
                                0,
                                [Scissor {
                                    offset: [clip_min_x as u32, clip_min_y as u32],
                                    extent: [
                                        (clip_max_x - clip_min_x) as u32,
                                        (clip_max_y - clip_min_y) as u32,
                                    ],
                                }]
                                .into_iter()
                                .collect(),
                            )
                            .unwrap();

                        unsafe {
                            builder
                                .draw_indexed(
                                    count as u32,
                                    1,
                                    index_offset as u32,
                                    vertex_offset as i32,
                                    0,
                                )
                                .unwrap()
                        };

                        index_offset += count;
                    }
                    _ => {}
                }
            }
            // Update vertex offset relative to the global buffer
            vertex_offset += draw_list.vtx_buffer().len();
        }
    }
}

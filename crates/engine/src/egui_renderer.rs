use std::collections::HashMap;

use egui::epaint::{ImageDelta, Primitive};
use egui::{ClippedPrimitive, TextureId, TexturesDelta};
use vulkanalia::bytecode::Bytecode;
use vulkanalia::prelude::v1_4::*;
use vulkanalia_bootstrap::Device;
use vulkanalia_vma::{self as vma, Alloc};

use crate::core::transition_image;
use crate::gpu_resources::{GpuBuffer, GpuImage};
use crate::pipeline::{allocate_descriptor_set, create_descriptor_set_layout};

// ---------------------------------------------------------------------------
// Embedded SPIR-V (compiled by build.rs via slangc)
// ---------------------------------------------------------------------------

const EGUI_VS_SPV: &[u8] = include_bytes!(env!("SHADER_EGUI_VS"));
const EGUI_FS_SPV: &[u8] = include_bytes!(env!("SHADER_EGUI_FS"));

fn spirv_bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const FRAME_OVERLAP: usize = 2;
const MAX_TEXTURES: u32 = 16;
const INITIAL_VERTEX_BUFFER_SIZE: vk::DeviceSize = 4 * 1024 * 1024; // 4 MB
const INITIAL_INDEX_BUFFER_SIZE: vk::DeviceSize = 2 * 1024 * 1024; // 2 MB

// ---------------------------------------------------------------------------
// Push constants — must match egui.slang PushConstants
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EguiPushConstants {
    screen_size: [f32; 2],
}

// ---------------------------------------------------------------------------
// Per-texture GPU resources
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct EguiTexture {
    image: vk::Image,
    view: vk::ImageView,
    allocation: vma::Allocation,
    descriptor_set: vk::DescriptorSet,
    width: u32,
    height: u32,
}

// ---------------------------------------------------------------------------
// Per-frame dynamic buffers
// ---------------------------------------------------------------------------

struct FrameBuffers {
    vertex: GpuBuffer,
    vertex_ptr: *mut u8,
    index: GpuBuffer,
    index_ptr: *mut u8,
}

// ---------------------------------------------------------------------------
// EguiRenderer
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub struct EguiRenderer {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    sampler: vk::Sampler,
    color_format: vk::Format,

    frames: Vec<FrameBuffers>,
    textures: HashMap<TextureId, EguiTexture>,

    /// Reusable staging buffer for texture uploads (grown as needed).
    staging_buffer: Option<GpuBuffer>,
    staging_ptr: *mut u8,
    staging_size: vk::DeviceSize,
}

impl EguiRenderer {
    /// Create the egui Vulkan renderer.
    /// `color_format` should match the draw-image format (e.g. R16G16B16A16_SFLOAT).
    pub fn new(
        device: &Device,
        allocator: &vma::Allocator,
        color_format: vk::Format,
    ) -> anyhow::Result<Self> {
        // --- Sampler ---
        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        let sampler = unsafe { device.create_sampler(&sampler_info, None)? };

        // --- Descriptor set layout ---
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
        ];
        let descriptor_set_layout = create_descriptor_set_layout(device, &bindings)?;

        // --- Descriptor pool ---
        let pool_sizes = [
            vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(MAX_TEXTURES)
                .build(),
            vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::SAMPLER)
                .descriptor_count(MAX_TEXTURES)
                .build(),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(MAX_TEXTURES)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        // --- Pipeline ---
        let (pipeline, pipeline_layout) =
            Self::create_pipeline(device, color_format, descriptor_set_layout)?;

        // --- Per-frame buffers ---
        let mut frames = Vec::with_capacity(FRAME_OVERLAP);
        for _ in 0..FRAME_OVERLAP {
            let (vertex, vertex_ptr) = GpuBuffer::new_mapped(
                allocator,
                INITIAL_VERTEX_BUFFER_SIZE,
                vk::BufferUsageFlags::VERTEX_BUFFER,
            )?;
            let (index, index_ptr) = GpuBuffer::new_mapped(
                allocator,
                INITIAL_INDEX_BUFFER_SIZE,
                vk::BufferUsageFlags::INDEX_BUFFER,
            )?;
            frames.push(FrameBuffers {
                vertex,
                vertex_ptr,
                index,
                index_ptr,
            });
        }

        Ok(Self {
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            sampler,
            color_format,
            frames,
            textures: HashMap::new(),
            staging_buffer: None,
            staging_ptr: std::ptr::null_mut(),
            staging_size: 0,
        })
    }

    // -----------------------------------------------------------------------
    // Texture management
    // -----------------------------------------------------------------------

    /// Process texture updates from egui (creates, updates, and frees).
    /// Must be called before `paint()`, within the same command buffer.
    pub fn update_textures(
        &mut self,
        device: &Device,
        allocator: &vma::Allocator,
        cmd: vk::CommandBuffer,
        textures_delta: &TexturesDelta,
    ) -> anyhow::Result<()> {
        for &id in &textures_delta.free {
            if let Some(tex) = self.textures.remove(&id) {
                unsafe {
                    device.destroy_image_view(tex.view, None);
                    allocator.destroy_image(tex.image, tex.allocation);
                }
            }
        }

        for (id, delta) in &textures_delta.set {
            self.set_texture(device, allocator, cmd, *id, delta)?;
        }

        Ok(())
    }

    fn set_texture(
        &mut self,
        device: &Device,
        allocator: &vma::Allocator,
        cmd: vk::CommandBuffer,
        id: TextureId,
        delta: &ImageDelta,
    ) -> anyhow::Result<()> {
        let pixels = Self::image_delta_to_rgba(delta);
        let [w, h] = delta.image.size();
        let w = w as u32;
        let h = h as u32;
        let data_size = (w * h * 4) as vk::DeviceSize;

        let is_partial = delta.pos.is_some();

        if !is_partial {
            // Full texture (re)create
            if let Some(old) = self.textures.remove(&id) {
                unsafe {
                    device.destroy_image_view(old.view, None);
                    allocator.destroy_image(old.image, old.allocation);
                }
            }

            let extent = vk::Extent3D::builder().width(w).height(h).depth(1).build();

            let img_info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .extent(extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST);

            let alloc_opts = vma::AllocationOptions::default();
            let (image, allocation) = unsafe { allocator.create_image(img_info, &alloc_opts) }?;

            let view = unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::builder()
                        .view_type(vk::ImageViewType::_2D)
                        .image(image)
                        .format(vk::Format::R8G8B8A8_UNORM)
                        .subresource_range(
                            vk::ImageSubresourceRange::builder()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1),
                        ),
                    None,
                )?
            };

            // Allocate descriptor set
            let descriptor_set =
                allocate_descriptor_set(device, self.descriptor_pool, self.descriptor_set_layout)?;

            // Write descriptor
            let image_info = [vk::DescriptorImageInfo::builder()
                .image_view(view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .build()];
            let sampler_info = [vk::DescriptorImageInfo::builder()
                .sampler(self.sampler)
                .build()];

            let writes = [
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&image_info)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&sampler_info)
                    .build(),
            ];

            unsafe {
                device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
            }

            // Upload via staging
            self.upload_texture_data(
                device, allocator, cmd, image, w, h, 0, 0, &pixels, data_size,
            )?;

            self.textures.insert(
                id,
                EguiTexture {
                    image,
                    view,
                    allocation,
                    descriptor_set,
                    width: w,
                    height: h,
                },
            );
        } else {
            // Partial update
            let [ox, oy] = delta.pos.unwrap();
            let tex = self.textures.get(&id).ok_or_else(|| {
                anyhow::anyhow!("Partial texture update for unknown texture {:?}", id)
            })?;

            self.upload_texture_data(
                device, allocator, cmd, tex.image, w, h, ox as u32, oy as u32, &pixels, data_size,
            )?;
        }

        Ok(())
    }

    fn upload_texture_data(
        &mut self,
        device: &Device,
        allocator: &vma::Allocator,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        width: u32,
        height: u32,
        offset_x: u32,
        offset_y: u32,
        pixels: &[u8],
        data_size: vk::DeviceSize,
    ) -> anyhow::Result<()> {
        // Ensure staging buffer is large enough
        if data_size > self.staging_size {
            if let Some(old) = self.staging_buffer.take() {
                old.destroy(allocator);
            }
            let new_size = data_size.max(256 * 1024); // minimum 256 KB
            let (buf, ptr) =
                GpuBuffer::new_mapped(allocator, new_size, vk::BufferUsageFlags::TRANSFER_SRC)?;
            self.staging_buffer = Some(buf);
            self.staging_ptr = ptr;
            self.staging_size = new_size;
        }

        // Copy pixel data to staging buffer
        unsafe {
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), self.staging_ptr, data_size as usize);
        }

        // Transition image to TRANSFER_DST
        transition_image(
            device,
            cmd,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        // Copy staging buffer → image
        let region = vk::BufferImageCopy::builder()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            )
            .image_offset(
                vk::Offset3D::builder()
                    .x(offset_x as i32)
                    .y(offset_y as i32)
                    .z(0)
                    .build(),
            )
            .image_extent(
                vk::Extent3D::builder()
                    .width(width)
                    .height(height)
                    .depth(1)
                    .build(),
            );

        unsafe {
            device.cmd_copy_buffer_to_image(
                cmd,
                self.staging_buffer.as_ref().unwrap().buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[*region],
            );
        }

        // Transition image to SHADER_READ_ONLY
        transition_image(
            device,
            cmd,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );

        Ok(())
    }

    /// Convert an `ImageDelta` to RGBA8 pixel data.
    fn image_delta_to_rgba(delta: &ImageDelta) -> Vec<u8> {
        match &delta.image {
            egui::ImageData::Color(color_image) => {
                assert_eq!(
                    color_image.width() * color_image.height(),
                    color_image.pixels.len()
                );
                let mut rgba = Vec::with_capacity(color_image.pixels.len() * 4);
                for pixel in &color_image.pixels {
                    rgba.push(pixel.r());
                    rgba.push(pixel.g());
                    rgba.push(pixel.b());
                    rgba.push(pixel.a());
                }
                rgba
            }
            egui::ImageData::Font(font_image) => {
                let mut rgba = Vec::with_capacity(font_image.pixels.len() * 4);
                for &coverage in &font_image.pixels {
                    let alpha = (coverage * 255.0 + 0.5) as u8;
                    rgba.push(255); // R
                    rgba.push(255); // G
                    rgba.push(255); // B
                    rgba.push(alpha); // A
                }
                rgba
            }
        }
    }

    // -----------------------------------------------------------------------
    // Rendering
    // -----------------------------------------------------------------------

    /// Record commands to render egui primitives into the draw image.
    /// The draw image must be in GENERAL layout on entry and will be in GENERAL
    /// layout on exit.
    pub fn paint(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        draw_image: &GpuImage,
        clipped_primitives: &[ClippedPrimitive],
        screen_size_px: [u32; 2],
        pixels_per_point: f32,
        frame_index: usize,
    ) {
        if clipped_primitives.is_empty() {
            return;
        }

        let fb = &self.frames[frame_index % FRAME_OVERLAP];

        // Collect all mesh data and upload to mapped buffers
        let mut total_vertices = 0usize;
        let mut total_indices = 0usize;
        for prim in clipped_primitives {
            if let Primitive::Mesh(mesh) = &prim.primitive {
                total_vertices += mesh.vertices.len();
                total_indices += mesh.indices.len();
            }
        }

        if total_vertices == 0 {
            return;
        }

        let vertex_byte_size = total_vertices * std::mem::size_of::<egui::epaint::Vertex>();
        let index_byte_size = total_indices * std::mem::size_of::<u32>();

        // Safety: we waited for this frame's fence before calling render,
        // so the GPU is done with the previous use of these buffers.
        assert!(
            vertex_byte_size as vk::DeviceSize <= fb.vertex.size,
            "egui vertex data ({vertex_byte_size} B) exceeds buffer ({} B) — need grow logic",
            fb.vertex.size
        );
        assert!(
            index_byte_size as vk::DeviceSize <= fb.index.size,
            "egui index data ({index_byte_size} B) exceeds buffer ({} B) — need grow logic",
            fb.index.size
        );

        // Copy mesh data into mapped buffers
        let mut v_offset = 0usize;
        let mut i_offset = 0usize;
        for prim in clipped_primitives {
            if let Primitive::Mesh(mesh) = &prim.primitive {
                let v_bytes = mesh.vertices.len() * std::mem::size_of::<egui::epaint::Vertex>();
                let i_bytes = mesh.indices.len() * std::mem::size_of::<u32>();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        mesh.vertices.as_ptr() as *const u8,
                        fb.vertex_ptr.add(v_offset),
                        v_bytes,
                    );
                    std::ptr::copy_nonoverlapping(
                        mesh.indices.as_ptr() as *const u8,
                        fb.index_ptr.add(i_offset),
                        i_bytes,
                    );
                }
                v_offset += v_bytes;
                i_offset += i_bytes;
            }
        }

        // --- Begin dynamic rendering ---
        transition_image(
            device,
            cmd,
            draw_image.image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

        let color_attachment = vk::RenderingAttachmentInfo::builder()
            .image_view(draw_image.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE);

        let render_area = vk::Rect2D::builder().extent(
            vk::Extent2D::builder()
                .width(screen_size_px[0])
                .height(screen_size_px[1])
                .build(),
        );

        let rendering_info = vk::RenderingInfo::builder()
            .render_area(*render_area)
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment));

        unsafe {
            device.cmd_begin_rendering(cmd, &rendering_info);

            // Bind pipeline
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

            // Viewport
            let viewport = vk::Viewport::builder()
                .x(0.0)
                .y(0.0)
                .width(screen_size_px[0] as f32)
                .height(screen_size_px[1] as f32)
                .min_depth(0.0)
                .max_depth(1.0);
            device.cmd_set_viewport(cmd, 0, &[*viewport]);

            // Push constants (screen size in physical pixels)
            let pc = EguiPushConstants {
                screen_size: [screen_size_px[0] as f32, screen_size_px[1] as f32],
            };
            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(&pc),
            );

            // Bind vertex + index buffers
            device.cmd_bind_vertex_buffers(cmd, 0, &[fb.vertex.buffer], &[0]);
            device.cmd_bind_index_buffer(cmd, fb.index.buffer, 0, vk::IndexType::UINT32);

            // Draw each mesh
            let mut vertex_base = 0i32;
            let mut index_base = 0u32;

            for prim in clipped_primitives {
                let mesh = match &prim.primitive {
                    Primitive::Mesh(m) => m,
                    Primitive::Callback(_) => continue,
                };

                if mesh.indices.is_empty() || mesh.vertices.is_empty() {
                    continue;
                }

                // Look up texture
                let Some(tex) = self.textures.get(&mesh.texture_id) else {
                    vertex_base += mesh.vertices.len() as i32;
                    index_base += mesh.indices.len() as u32;
                    continue;
                };

                // Clip rect (egui gives logical points → convert to physical pixels)
                let clip = prim.clip_rect;
                let x = (clip.min.x * pixels_per_point).round() as i32;
                let y = (clip.min.y * pixels_per_point).round() as i32;
                let w = ((clip.max.x - clip.min.x) * pixels_per_point).round() as u32;
                let h = ((clip.max.y - clip.min.y) * pixels_per_point).round() as u32;

                // Clamp to framebuffer
                let x = x.max(0);
                let y = y.max(0);
                let w = w.min(screen_size_px[0].saturating_sub(x as u32));
                let h = h.min(screen_size_px[1].saturating_sub(y as u32));

                if w == 0 || h == 0 {
                    vertex_base += mesh.vertices.len() as i32;
                    index_base += mesh.indices.len() as u32;
                    continue;
                }

                let scissor = vk::Rect2D::builder()
                    .offset(vk::Offset2D::builder().x(x).y(y).build())
                    .extent(vk::Extent2D::builder().width(w).height(h).build());
                device.cmd_set_scissor(cmd, 0, &[*scissor]);

                // Bind texture descriptor
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    &[tex.descriptor_set],
                    &[],
                );

                device.cmd_draw_indexed(
                    cmd,
                    mesh.indices.len() as u32,
                    1,
                    index_base,
                    vertex_base,
                    0,
                );

                vertex_base += mesh.vertices.len() as i32;
                index_base += mesh.indices.len() as u32;
            }

            device.cmd_end_rendering(cmd);
        }

        transition_image(
            device,
            cmd,
            draw_image.image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::GENERAL,
        );
    }

    // -----------------------------------------------------------------------
    // Pipeline creation
    // -----------------------------------------------------------------------

    fn create_pipeline(
        device: &Device,
        color_format: vk::Format,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> anyhow::Result<(vk::Pipeline, vk::PipelineLayout)> {
        let vs_words = spirv_bytes_to_words(EGUI_VS_SPV);
        let fs_words = spirv_bytes_to_words(EGUI_FS_SPV);

        let vs_bytecode = Bytecode::new(bytemuck::cast_slice(&vs_words))?;
        let vs_module_info = vk::ShaderModuleCreateInfo::builder()
            .code(vs_bytecode.code())
            .code_size(vs_bytecode.code_size());
        let vs_module = unsafe { device.create_shader_module(&vs_module_info, None)? };

        let fs_bytecode = Bytecode::new(bytemuck::cast_slice(&fs_words))?;
        let fs_module_info = vk::ShaderModuleCreateInfo::builder()
            .code(fs_bytecode.code())
            .code_size(fs_bytecode.code_size());
        let fs_module = unsafe { device.create_shader_module(&fs_module_info, None)? };

        let vert_entry = b"main\0";
        let frag_entry = b"main\0";

        let stages = [
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vs_module)
                .name(vert_entry)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fs_module)
                .name(frag_entry)
                .build(),
        ];

        // Vertex input — matches egui::epaint::Vertex (20 bytes):
        //   pos:   float2   offset 0
        //   uv:    float2   offset 8
        //   color: rgba8    offset 16
        let vertex_bindings = [vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(20) // sizeof(egui::epaint::Vertex)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()];

        let vertex_attrs = [
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT) // pos
                .offset(0)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32_SFLOAT) // uv
                .offset(8)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(2)
                .format(vk::Format::R8G8B8A8_UNORM) // color
                .offset(16)
                .build(),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&vertex_bindings)
            .vertex_attribute_descriptions(&vertex_attrs);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)
            .scissor_count(1);

        let rasterization = vk::PipelineRasterizationStateCreateInfo::builder()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);

        let multisample = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::_1);

        // Premultiplied alpha blending: src=ONE, dst=ONE_MINUS_SRC_ALPHA
        let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::all())
            .build()];

        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::builder().attachments(&color_blend_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

        // Dynamic rendering info (no render pass)
        let color_formats = [color_format];
        let mut rendering_info =
            vk::PipelineRenderingCreateInfo::builder().color_attachment_formats(&color_formats);

        // Push constants
        let push_constant_ranges = [vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<EguiPushConstants>() as u32)
            .build()];

        let set_layouts = [descriptor_set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_constant_ranges);
        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .push_next(&mut rendering_info);

        let pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[*pipeline_info], None)?
                .0
                .into_iter()
                .next()
                .ok_or(anyhow::anyhow!("Failed to create egui graphics pipeline"))?
        };

        unsafe {
            device.destroy_shader_module(vs_module, None);
            device.destroy_shader_module(fs_module, None);
        }

        Ok((pipeline, pipeline_layout))
    }

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------

    pub fn destroy(&mut self, device: &Device, allocator: &vma::Allocator) {
        unsafe {
            device.device_wait_idle().ok();
        }

        for fb in &self.frames {
            fb.vertex.destroy(allocator);
            fb.index.destroy(allocator);
        }
        self.frames.clear();

        for (_, tex) in self.textures.drain() {
            unsafe {
                device.destroy_image_view(tex.view, None);
                allocator.destroy_image(tex.image, tex.allocation);
            }
        }

        if let Some(staging) = self.staging_buffer.take() {
            staging.destroy(allocator);
        }

        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_sampler(self.sampler, None);
        }
    }
}

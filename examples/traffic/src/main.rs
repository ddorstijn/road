use engine::pipeline::{
    allocate_descriptor_set, compile_wgsl, create_compute_pipeline, create_descriptor_set_layout,
};
use engine::vk::{DeviceV1_0, Handle, HasBuilder};
use engine::{App, EngineContext, vk};

const GRID_SHADER: &str = include_str!("../../../assets/shaders/grid.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraParams {
    inv_vp: [[f32; 4]; 4],
}

struct TrafficApp {
    grid_pipeline: vk::Pipeline,
    grid_pipeline_layout: vk::PipelineLayout,
    grid_descriptor_set_layout: vk::DescriptorSetLayout,
    grid_descriptor_pool: vk::DescriptorPool,
    grid_descriptor_set: vk::DescriptorSet,
}

impl Default for TrafficApp {
    fn default() -> Self {
        Self {
            grid_pipeline: vk::Pipeline::null(),
            grid_pipeline_layout: vk::PipelineLayout::null(),
            grid_descriptor_set_layout: vk::DescriptorSetLayout::null(),
            grid_descriptor_pool: vk::DescriptorPool::null(),
            grid_descriptor_set: vk::DescriptorSet::null(),
        }
    }
}

impl TrafficApp {
    fn create_grid_pipeline(&mut self, ctx: &EngineContext) -> anyhow::Result<()> {
        let device = ctx.device.as_ref();

        // Descriptor set layout: binding 0 = storage image
        let bindings = [vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build()];

        self.grid_descriptor_set_layout = create_descriptor_set_layout(device, &bindings)?;

        // Descriptor pool
        let pool_sizes = [vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .build()];
        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        self.grid_descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        // Descriptor set
        self.grid_descriptor_set = allocate_descriptor_set(
            device,
            self.grid_descriptor_pool,
            self.grid_descriptor_set_layout,
        )?;

        // Write descriptors
        self.update_grid_descriptors(ctx);

        // Compile shader and create pipeline with push constant for camera inverse VP
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
}

impl App for TrafficApp {
    fn init(&mut self, ctx: &EngineContext) -> anyhow::Result<()> {
        self.create_grid_pipeline(ctx)?;
        Ok(())
    }

    fn resize(&mut self, ctx: &EngineContext) -> anyhow::Result<()> {
        self.update_grid_descriptors(ctx);
        Ok(())
    }

    fn render(&mut self, ctx: &EngineContext, cmd: vk::CommandBuffer) -> anyhow::Result<()> {
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

            // Push the inverse view-projection matrix so the shader can map pixels → world
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
        }
    }
}

fn main() -> anyhow::Result<()> {
    engine::run(TrafficApp::default())
}

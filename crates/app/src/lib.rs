use std::sync::Arc;

use engine::{component::GameComponent, engine::VulkanContext};
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
};

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/gradient.comp"
    }
}

pub struct App {
    push_constants: cs::constants,
    pipeline: Option<Arc<ComputePipeline>>,
}

impl Default for App {
    fn default() -> Self {
        let push_constants = cs::constants {
            start: [1., 0.],
            end: [0., 0.],
            radius: 0.1,
        };

        Self {
            push_constants,
            pipeline: None,
        }
    }
}

impl GameComponent for App {
    fn start(&mut self, ctx: &VulkanContext) {
        self.pipeline = {
            let cs = cs::load(ctx.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                ctx.device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(ctx.device.clone())
                    .unwrap(),
            )
            .unwrap();

            Some(
                ComputePipeline::new(
                    ctx.device.clone(),
                    None,
                    ComputePipelineCreateInfo::stage_layout(stage, layout),
                )
                .unwrap(),
            )
        };
    }

    fn ui(&mut self, _ctx: &VulkanContext, ui: &mut imgui::Ui) {
        ui.window("Hello world")
            .size([300.0, 100.0], imgui::Condition::FirstUseEver)
            .build(|| {
                ui.text("Hello world!");
                ui.input_float2("data1", &mut self.push_constants.start)
                    .build();
                ui.input_float2("data2", &mut self.push_constants.end)
                    .build();
            });
    }

    fn render(
        &mut self,
        ctx: &VulkanContext,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        let Some(ref pipeline) = self.pipeline else {
            return;
        };

        let layout = &pipeline.layout().set_layouts()[0];
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            layout.clone(),
            [WriteDescriptorSet::image_view(0, ctx.draw_image.clone())],
            [],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                set,
            )
            .unwrap()
            .push_constants(pipeline.layout().clone(), 0, self.push_constants)
            .unwrap();

        // Get the current dimensions of the drawing image
        let [width, height, _] = ctx.draw_image.image().extent();

        // Calculate how many workgroups are needed to cover the image.
        // We assume the shader uses a local_size of 16x16.
        // (width + 15) / 16 is integer math for ceil(width / 16.0)
        let dispatch_x = (width + 15) / 16;
        let dispatch_y = (height + 15) / 16;

        unsafe { builder.dispatch([dispatch_x, dispatch_y, 1]) }.unwrap();
    }
}

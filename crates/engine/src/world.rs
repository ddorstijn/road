use std::sync::Arc;

use vulkano::{
    device::Device,
    pipeline::{ComputePipeline, GraphicsPipeline},
};
use vulkano_taskgraph::{Task, TaskContext};

// This is passed to every task during execution
pub struct AppWorld {
    pub device: Arc<Device>,
    pub road_pipeline: Arc<GraphicsPipeline>,
    pub sdf_pipeline: Arc<ComputePipeline>,
}

impl Task for DrawTask {
    type World = AppWorld;

    fn execute(&self, context: &mut TaskContext<'_>, world: &Self::World) -> Result<(), TaskError> {
        let mut builder = context.main_command_buffer_builder()?;

        // Resolve the virtual ID to a real subbuffer for this frame
        let buffer = context.get_buffer(self.buffer)?;

        builder
            .bind_pipeline_graphics(world.road_pipeline.clone())
            .bind_vertex_buffers(0, buffer); // Use the shared data

        Ok(())
    }
}

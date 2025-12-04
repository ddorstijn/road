use crate::engine::VulkanContext;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};

/// This mimics Unity's MonoBehaviour
pub trait GameComponent {
    /// Called once when the Engine initializes (like Unity's Start)
    /// Use this to load pipelines, shaders, and buffers.
    fn start(&mut self, ctx: &VulkanContext);

    /// Called every frame for logic/UI (like Unity's OnGUI + Update)
    fn ui(&mut self, ctx: &VulkanContext, ui: &mut imgui::Ui);

    /// Called every frame to record Vulkan commands (like Unity's OnRenderObject)
    fn render(
        &mut self,
        ctx: &VulkanContext,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    );
}

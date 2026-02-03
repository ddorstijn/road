pub use imgui::Ui;
pub use vulkano_taskgraph::graph::TaskNodeBuilder;
pub trait GameComponent: Send + Sync {
    /// Called once to add this component's tasks to the graph.
    /// Use this to define which Virtual IDs this task reads from or writes to.
    fn add_to_graph(&self, builder: &mut TaskNodeBuilder, world: &TaskWorld);
}

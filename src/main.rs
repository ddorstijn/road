use bevy::prelude::*;

mod simulation;
use crate::simulation::GpuSimulationPlugin;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, GpuSimulationPlugin))
        .insert_resource(ClearColor(Color::BLACK))
        .run();
}

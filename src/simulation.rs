use bevy::{
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        extract_component::ExtractComponent,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{
            AsBindGroup, BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry,
            BindingType, Buffer, BufferBindingType, BufferInitDescriptor, BufferUsages,
            CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
            PipelineCache, ShaderRef, ShaderStages,
        },
        renderer::{RenderContext, RenderDevice},
    },
};
use rand::Rng;

/// This example uses a shader source file from the assets subdirectory
const SHADER_ASSET_PATH: &str = "shaders/compute.wgsl";

// The length of the buffer sent to the gpu
pub const BUFFER_LEN: usize = 16;

// We need a plugin to organize all the systems and render node required for this example
pub struct GpuSimulationPlugin;
impl Plugin for GpuSimulationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimulationParams>();
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<SimulationTimer>()
            .add_systems(
                RenderStartup,
                (init_compute_pipeline, add_compute_render_graph_node),
            )
            .add_systems(
                Render,
                simulation_tick_system.run_if(resource_exists::<SimulationBuffers>),
            )
            .add_systems(
                Render,
                prepare
                    .in_set(RenderSystems::PrepareBindGroups)
                    // We don't need to recreate the bind group every frame
                    .run_if(not(resource_exists::<SimulationBuffers>)),
            );
    }
}

fn add_compute_render_graph_node(mut render_graph: ResMut<RenderGraph>) {
    // Add the compute node as a top-level node to the render graph. This means it will only execute
    // once per frame. Normally, adding a node would use the `RenderGraphApp::add_render_graph_node`
    // method, but it does not allow adding as a top-level node.
    render_graph.add_node(ComputeNodeLabel, ComputeNode::default());
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Particle {
    pos: Vec2,
    vel: Vec2,
}

fn prepare(
    mut commands: Commands,
    pipeline: Res<ComputePipeline>,
    render_device: Res<RenderDevice>,
    world: &World,
) {
    let params = world.get_resource::<SimulationParams>();
    if params.is_none() {
        return;
    }

    let params = params.unwrap();

    let mut initial_particles = Vec::with_capacity(params.num_particles as usize);
    let mut rng = rand::rng();
    for _ in 0..params.num_particles {
        let angle = rng.random::<f32>() * 2.0 * std::f32::consts::PI;
        let radius = rng.random::<f32>().sqrt() * 0.6 + 0.1;
        initial_particles.push(Particle {
            pos: Vec2::from_angle(angle) * radius,
            vel: Vec2::ZERO,
        });
    }

    let buffers: [Buffer; 3] = std::array::from_fn(|i| {
        render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some(&format!("Simulation Buffer {}", i)),
            contents: bytemuck::cast_slice(&initial_particles),
            usage: BufferUsages::STORAGE,
        })
    });

    let bindgroups: [BindGroup; 3] = std::array::from_fn(|i| {
        // Each bind group will correspond to a buffer at the same index.
        let buffer = &buffers[i];
        render_device.create_bind_group(
            Some(format!("Simulation BG {}", i).as_str()),
            &pipeline.layout, // Assumes pipeline.layout is accessible
            &[BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        )
    });

    commands.insert_resource(SimulationBuffers {
        buffers,
        bindgroups,
        active_idx: 0,
    });
}

#[derive(Resource)]
struct ComputePipeline {
    layout: BindGroupLayout,
    pipeline: CachedComputePipelineId,
}

fn init_compute_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = render_device.create_bind_group_layout(
        None,
        &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    );
    let shader = asset_server.load(SHADER_ASSET_PATH);
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("GPU readback compute shader".into()),
        layout: vec![layout.clone()],
        shader: shader.clone(),
        ..default()
    });
    commands.insert_resource(ComputePipeline { layout, pipeline });
}

/// Label to identify the node in the render graph
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct ComputeNodeLabel;

/// The node that will execute the compute shader
#[derive(Default)]
struct ComputeNode {}

impl render_graph::Node for ComputeNode {
    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<ComputePipeline>();
        let Some(sim_buffers) = world.get_resource::<SimulationBuffers>() else {
            return Ok(());
        };

        if let Some(init_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) {
            let mut pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("GPU readback compute pass"),
                        ..default()
                    });

            // Read-only previous bindgroup
            pass.set_bind_group(0, sim_buffers.previous_bindgroup(), &[]);
            // Active bindgroup
            pass.set_bind_group(1, sim_buffers.active_bindgroup(), &[]);
            pass.set_pipeline(init_pipeline);
            pass.dispatch_workgroups(BUFFER_LEN as u32, 1, 1);
        }
        Ok(())
    }
}

//==============================================================================
// Main World Resources & Systems
//==============================================================================

#[derive(Resource)]
struct SimulationParams {
    num_particles: u32,
    tick_duration: std::time::Duration,
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            num_particles: 100_000,
            tick_duration: std::time::Duration::from_millis(500),
        }
    }
}

#[derive(Resource)]
struct SimulationTimer(Timer);

impl Default for SimulationTimer {
    fn default() -> Self {
        Self(Timer::from_seconds(0.5, TimerMode::Repeating))
    }
}

#[derive(Resource)]
pub struct SimulationBuffers {
    #[allow(unused)]
    pub buffers: [Buffer; 3],
    pub bindgroups: [BindGroup; 3],
    pub active_idx: usize,
}

impl SimulationBuffers {
    /// Returns the index of the previous buffer, wrapping around if the current active_idx is 0.
    pub fn previous_idx(&self) -> usize {
        // This is a common pattern for "previous" with wrapping.
        // (current_index + total_length - 1) % total_length
        (self.active_idx + self.bindgroups.len() - 1) % self.bindgroups.len()
    }

    /// Returns a reference to the previous buffer.
    pub fn previous_bindgroup(&self) -> &BindGroup {
        let prev_idx = self.previous_idx();
        &self.bindgroups[prev_idx]
    }

    /// Returns a reference to the previous buffer.
    pub fn active_bindgroup(&self) -> &BindGroup {
        &self.bindgroups[self.active_idx]
    }

    /// Advances the active_idx to the next buffer, wrapping around.
    pub fn advance_to_next(&mut self) {
        self.active_idx = (self.active_idx + 1) % self.bindgroups.len();
    }
}

fn simulation_tick_system(
    time: Res<Time>,
    params: Res<SimulationParams>,
    mut timer: ResMut<SimulationTimer>,
    mut tracker: ResMut<SimulationBuffers>,
) {
    timer.0.set_duration(params.tick_duration);
    timer.0.tick(time.delta());

    if timer.0.just_finished() {
        tracker.advance_to_next();
    }
}

#[derive(AsBindGroup, TypePath, Debug, Clone, Asset)]
pub struct ParticleMaterial {
    #[uniform(0)]
    color: Color,
}

impl Material for ParticleMaterial {
    fn vertex_shader() -> ShaderRef {
        PARTICLE_RENDER_SHADER.into()
    }
    fn fragment_shader() -> ShaderRef {
        PARTICLE_RENDER_SHADER.into()
    }
    fn specialized_pipeline(
        &self,
        layout: &bevy::render::render_resource::PipelineLayout,
        format: bevy::render::render_resource::TextureFormat,
        msaa: &Msaa,
    ) -> SpecializedRenderPipeline {
        SpecializedRenderPipeline {
            layout,
            vertex: SpecializedRenderPipeline::vertex_from_source_and_attributes(
                PARTICLE_RENDER_SHADER,
                &[
                    // Position and velocity from compute buffer
                    VertexAttribute {
                        format: VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                        name: "Position",
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32x2,
                        offset: 8,
                        shader_location: 1,
                        name: "Velocity",
                    },
                ],
                vec![
                    // For interpolation, we need two buffers
                    VertexStepMode::Instance,
                    VertexStepMode::Instance,
                ],
            ),
            fragment: Some(SpecializedRenderPipeline::fragment_from_source(
                PARTICLE_RENDER_SHADER,
            )),
            primitive: bevy::render::render_resource::PrimitiveState::default(),
            depth_stencil: Some(bevy::render::render_resource::DepthStencilState::default()),
            multisample: msaa.get_wgpu_multisample_state(),
        }
    }
}

// The mesh for a single particle quad
#[derive(Resource, Clone, ExtractComponentt)]
struct ParticleMesh(Handle<Mesh>);

fn queue_particles(
    mut commands: Commands,
    mut particle_phase: ResMut<RenderPhase<ParticleRenderPhase>>,
    particle_mesh: Res<ParticleMesh>,
    render_device: Res<RenderDevice>,
    render_assets: Res<RenderAssets<Mesh>>,
    sim_buffers: Res<SimulationBuffers>,
    views: Query<Entity, With<ManualTextureView>>,
) {
    let Some(mesh_handle) = render_assets.get(&particle_mesh.0) else {
        return;
    };
    for view_entity in &views {
        // We'll queue a draw command for each view, referencing the single mesh
        // This command will use the particle buffers for instancing
        let item = EntityPhaseItem {
            entity: view_entity,
            draw_function_handle: DrawFunctions::get_handle::<DrawParticles>(),
            sort_key: 0.0,
            distance: 0.0,
        };
        particle_phase.add(item);
    }
    // We also need to bind the particle buffers for rendering
    let (old_buffer, new_buffer) = sim_buffers.render_buffers();
    let old_buffer_binding = old_buffer.as_entire_binding();
    let new_buffer_binding = new_buffer.as_entire_binding();

    // Now, create a render pass and bind the buffers before drawing
    // This part would be more complex in a real implementation, but for simplicity,
    // we'll assume a single pass with both buffers bound
    commands.add_system(move |render_context: &mut RenderContext| {
        // Setup render pass, bind pipelines and buffers, and call draw_indexed
        let mut render_pass = render_context.command_encoder().begin_render_pass(
            // ...
        );
        render_pass.set_vertex_buffer(0, old_buffer_binding);
        render_pass.set_vertex_buffer(1, new_buffer_binding);
        render_pass.draw_indexed(0..mesh_handle.indices.len(), 0, 0..BUFFER_LEN as u32);
    });
}

// The custom render phase label
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct ParticleRenderPhase;

// The custom render command for drawing particles
#[derive(Debug, Clone, Copy)]
pub struct DrawParticles;
impl<P: PhaseItem> render_phase::Draw<P> for DrawParticles {
    fn draw<'w>(
        &mut self,
        _item: &'_ P,
        _view_data: &'_ ViewUniforms,
        _pipeline: &'_ SpecializedRenderPipeline,
        _bind_groups: &'_ [BindGroup],
        _render_context: &'_ mut RenderContext,
        _command_encoder: &'_ mut wgpu::CommandEncoder,
    ) {
        // The actual draw call will be handled by the queuing system
        // This is a simplified example
    }
}

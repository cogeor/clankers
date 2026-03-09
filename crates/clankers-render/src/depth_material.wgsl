#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_view_bindings::view

struct DepthMaterialUniform {
    max_depth: f32,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> material: DepthMaterialUniform;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Transform world position to view space; view Z is negative for
    // objects in front of the camera.
    let view_pos = view.view_from_world * in.world_position;
    let linear_depth = -view_pos.z;
    let normalized = clamp(linear_depth / material.max_depth, 0.0, 1.0);
    return vec4<f32>(normalized, normalized, normalized, 1.0);
}

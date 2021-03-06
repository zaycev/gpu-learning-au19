#version 450
#define MAX_MODELS 10
#define MAX_LIGHTS 10

//
// Inputs.
//
layout (location = 0) in vec3 in_pose;
layout (location = 1) in vec3 in_norm;

//
// Outputs.
//
layout (location = 0) out     gl_PerVertex {
    vec4 gl_Position;
};
layout (location = 1) out vec3 out_norm;
layout (location = 2) out vec3 out_pose;
layout (location = 3) out int  out_idx;

//
// Uniforms.
//
layout (push_constant) uniform PushConsts {
    mat4 p;
    mat4 v;
} push;

layout(binding = 1) uniform TransformsUniformBlock {
    mat4 m[MAX_MODELS];
} transform_uniform;


void main()
{
    // Compute mvp.
    mat4 m   = transform_uniform.m[gl_InstanceIndex];
    mat4 mvp = push.p * push.v * m;

    // Copute vertex pose.
    gl_Position = mvp * vec4(in_pose, 1.0);

    // Write outputs.
    out_norm    = in_norm;
    out_pose    = in_pose;
    out_idx     = gl_InstanceIndex;
}
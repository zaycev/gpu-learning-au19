#version 450
#define MAX_MODELS 10
#define MAX_LIGHTS 10
#define PI 3.14159265
#define GAMMA 2.2

// LightSource contains position and color of a single ligt source.
struct LightSource
{
    vec4 pose;
    vec4 color;
};

//
// Inputs.
//
layout (location = 1) in vec3      in_norm;
layout (location = 2) in vec3      in_pose;
layout (location = 3) in flat int  in_idx;

//
// Outputs.
//
layout (location = 0) out vec4     color;

//
// Uniforms.
//
layout (push_constant) uniform PushConsts {
    mat4 p;
    mat4 v;
} push;
layout(binding = 0) uniform LightsUniformBlock {
    uvec4       sources_num;
    uvec4       junk;
    LightSource sources[MAX_LIGHTS];
} light_uniform;
layout(binding = 1) uniform TransformsUniformBlock {
    mat4 m[MAX_MODELS];
} transform_uniform;

//
// Utility functions for lighting model.
// See: https://github.com/stackgl/glsl-lighting-walkthrough
//

vec3 to_linear(vec3 v) {
    return pow(v, vec3(GAMMA));
}

vec3 to_gamma(vec3 v) {
    return pow(v, vec3(1.0 / GAMMA));
}

float attenuation(float r, float f, float d) {
    float denom = d / r + 1.0;
    float attenuation = 1.0 / (denom*denom);
    float t = (attenuation - f) / (1.0 - f);
    return max(t, 0.0);
}

float oren_nayar_diffuse(
    vec3 light_direction,
    vec3 view_direction,
    vec3 surface_normal,
    float roughness,
    float albedo) {

    float l_dot_v = dot(light_direction, view_direction);
    float n_dot_l = dot(light_direction, surface_normal);
    float n_dot_v = dot(surface_normal, view_direction);

    float s       = l_dot_v - n_dot_l * n_dot_v;
    float t       = mix(1.0, max(n_dot_l, n_dot_v), step(0.0, s));

    float sigma2  = roughness * roughness;
    float a       = 1.0 + sigma2 * (albedo / (sigma2 + 0.13) + 0.5 / (sigma2 + 0.33));
    float b       = 0.45 * sigma2 / (sigma2 + 0.09);

    return albedo * max(0.0, n_dot_l) * (a + b * s / t) / PI;
}

void main() {

    vec3 result         = vec3(0, 0, 0);
    vec3 in_color       = vec3(1.0, 1.0, 1.0);
    uint sources_num    = light_uniform.sources_num.x;

    for (int i = 0; i < sources_num; ++i) {

        ///
        mat4 model_view      = push.v * transform_uniform.m[in_idx];
        mat3 normal_matrix   = transpose(inverse(mat3(model_view)));
        vec3 norm            = normalize(normal_matrix * in_norm);
        vec3 diffuse_color   = to_gamma(to_linear(in_color));
        vec4 view_pose       = model_view * vec4(in_pose, 1.0);

        ///
        vec4 light_pose      = push.v * vec4(light_uniform.sources[i].pose.xyz, 1.0);
        vec3 light_color     = light_uniform.sources[i].color.xyz;
        vec3 light_vec       = light_pose.xyz - view_pose.xyz;
        float light_dist     = length(light_vec);
        float falloff        = attenuation(1.1, 0.03, light_dist);

        ///
        vec3 l               = normalize(light_vec);
        vec3 v               = normalize(view_pose.xyz);
        vec3 n               = norm;
        vec3 diffuse         = light_color * oren_nayar_diffuse(l, v, n, 1.2, 1.2) * falloff;

        result               += diffuse_color * diffuse;
    }

    vec3 final = 0.02 * in_color + 0.98 * result;
    color      = vec4(final, 1.0);
}
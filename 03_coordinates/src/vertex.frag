#version 450

#define MAX_MODELS 10
#define MAX_LIGHTS 10
#define PI 3.14159265

struct LightSource
{
    vec4 pose;
    vec4 color;
};

layout (push_constant) uniform PushConsts {
    mat4 p;
    mat4 v;
} push;
layout (location = 1) in vec3      in_norm;
layout (location = 2) in vec3      in_pose;
layout (location = 3) in vec3      in_color;
layout (location = 4) in flat int  in_idx;
layout (location = 0) out vec4     color;
layout(binding = 0) uniform LightsUniformBlock {
    LightSource sources[MAX_LIGHTS];
} light_uniform;
layout(binding = 1) uniform TransformsUniformBlock {
    mat4 m[MAX_MODELS];
} transform_uniform;

const float gamma = 2.2;

vec3 toLinear(vec3 v) {
    return pow(v, vec3(gamma));
}

vec3 toGamma(vec3 v) {
    return pow(v, vec3(1.0 / gamma));
}

float attenuation(float r, float f, float d) {
    float denom = d / r + 1.0;
    float attenuation = 1.0 / (denom*denom);
    float t = (attenuation - f) / (1.0 - f);
    return max(t, 0.0);
}

float orenNayarDiffuse(
    vec3 lightDirection,
    vec3 viewDirection,
    vec3 surfaceNormal,
    float roughness,
    float albedo) {

    float LdotV = dot(lightDirection, viewDirection);
    float NdotL = dot(lightDirection, surfaceNormal);
    float NdotV = dot(surfaceNormal, viewDirection);

    float s = LdotV - NdotL * NdotV;
    float t = mix(1.0, max(NdotL, NdotV), step(0.0, s));

    float sigma2 = roughness * roughness;
    float A = 1.0 + sigma2 * (albedo / (sigma2 + 0.13) + 0.5 / (sigma2 + 0.33));
    float B = 0.45 * sigma2 / (sigma2 + 0.09);

    return albedo * max(0.0, NdotL) * (A + B * s / t) / PI;
}

void main()
{
    vec3 result          = vec3(0, 0, 0);

    // vec3 eye_position = vec3(0.0, 1.0, -2.0);

    for (int i=0; i<MAX_LIGHTS; ++i) {

        mat4 m               = transform_uniform.m[in_idx];
        mat4 mvp             = push.p * push.v * m;
        mat4 model_view      = push.v * transform_uniform.m[in_idx];
        mat3 normal_matrix   = transpose(inverse(mat3(model_view)));
        vec3 norm            = normalize(normal_matrix * in_norm);
        vec3 diffuse_color   = toGamma(toLinear(in_color));
        vec4 view_pose       = model_view * vec4(in_pose, 1.0);

        ///
        vec4 light_pose      = push.v * vec4(light_uniform.sources[i].pose.xyz, 1.0);
        vec3 light_color     = light_uniform.sources[i].color.xyz;
        vec3 light_vec       = light_pose.xyz - view_pose.xyz;
        float light_dist     = length(light_vec);
        float falloff        = attenuation(1.1, 0.1, light_dist);


        vec3 L = normalize(light_vec);
        vec3 V = normalize(view_pose.xyz);
        vec3 N = norm;

        vec3 diffuse = light_color * orenNayarDiffuse(L, V, N, 1.0, 1.0) * falloff;

        result += diffuse_color * diffuse;
    }

    ///

    color                = vec4(result, 1.0);


    /**
    vec3 base_color      = in_color;
    vec3 result          = vec3(0, 0, 0);


    // Basic phong model.
    for (int i=0; i<MAX_LIGHTS; ++i) {

        // Get light source data from uniform.
        vec3 light_pose  = light_uniform.sources[i].pose.xyz;
        vec3 light_color = light_uniform.sources[i].color.xyz;

        // Compute normale and direction.
        vec3 light_dir = normalize(light_pose - in_pose);
        vec3 norm_rot  = (transform_uniform.m[in_idx] * vec4(in_norm, 1.0)).xyz;
        vec3 norm      = normalize(norm_rot);

        // Compute diffuse component.
        float diff    = max(dot(norm, light_dir), 0.0);
        vec3 diffuse  = diff * light_color;

        // Accumulate result.
        result += diffuse * base_color;
    }

    color = vec4(result, 1.0);
    **/
}
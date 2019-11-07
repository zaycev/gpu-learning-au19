#version 450

layout (push_constant) uniform PushConsts {
    uint frame_counter;
} push;

layout (location = 1) in vec3 in_norm;
layout (location = 2) in vec3 in_pose;
layout (location = 3) in vec3 in_color;
layout (location = 0) out vec4 color;

void main()
{
    ///
    /// basic implementation of phong lighting model.
    ///

    vec3 light_color = vec3(1.0, 1.0, 1.0);
    vec3 base_color = in_color;

    vec3 light_pose = vec3(0.0, 1.0, -2.0);

    vec3 light_dir = normalize(light_pose - in_pose);

    vec3 norm = normalize(in_norm);

    float diff = max(dot(norm, light_dir), 0.0);

    vec3 diffuse = diff * light_color;

    vec3 result = 0.05 * base_color + 0.95 * diffuse * base_color;

    color = vec4(result * 0.5 + 0.5 * norm, 1.0);
}
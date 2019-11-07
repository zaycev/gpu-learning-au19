#version 450

layout (push_constant) uniform PushConsts {
    mat4 vp;
} push;
layout (location = 0) in vec3 in_pose;
layout (location = 1) in vec3 in_norm;

layout (location = 0) out gl_PerVertex {
    vec4 gl_Position;
};
layout (location = 1) out smooth vec3 out_norm;
layout (location = 2) out smooth vec3 out_pose;
layout (location = 3) out smooth vec3 out_color;

void main()
{
    gl_Position = push.vp * vec4(in_pose.xyz, 1.0);
    out_norm = in_norm;
    out_pose = (push.vp * vec4(in_pose.xyz, 1.0)).xyz;
    out_color = vec3(1.0, 1.0, 1.0);
}
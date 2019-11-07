#version 450

#define PI 3.1415926538

layout (push_constant) uniform PushConsts {
  uint frame_counter;
} push;

layout (location = 1) in  vec3 vert_color;
layout (location = 0) out vec4 color;

void main()
{	
	float value = float(push.frame_counter) / 180 * PI;
	float mapped = abs(sin(value));

	color = vec4(vert_color * mapped, 1.0);
}
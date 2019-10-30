#version 450

layout (location = 1) in vec3 vert_color;
layout (location = 0) out vec4 color;

void main()
{
	color = vec4(
		vert_color.x,
		vert_color.y,
		vert_color.z,
		1.0
	);
}
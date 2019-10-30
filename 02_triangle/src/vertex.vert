#version 450

layout (location = 0) in vec2 in_position;
layout (location = 0) out  gl_PerVertex {
  vec4 gl_Position;
};
layout (location = 1) out vec3 vert_color;

void main()
{
  gl_Position = vec4(in_position, 0.0, 1.0);
  float len = sqrt(
    in_position.x * in_position.x + 
    in_position.y * in_position.y
  );
  vert_color = vec3(0.0, 0.0, 0.0);
  vert_color[gl_VertexIndex] = len;  
}
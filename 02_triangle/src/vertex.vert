#version 450

layout (location = 0) in vec2 in_position;
layout (location = 0) out     gl_PerVertex {
  vec4 gl_Position;
};
layout (location = 1) out vec3 vert_color;

void main()
{
  // Set position.
  gl_Position = vec4(in_position, 0.0, 1.0);

  // Compute distance from center to the current vertex.  
  float len = sqrt(
    in_position.x * in_position.x + 
    in_position.y * in_position.y
  );

  // Compute color using vertex index.
  vert_color = vec3(0.0, 0.0, 0.0);
  vert_color[gl_VertexIndex] = len;  
}
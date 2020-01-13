extern crate gfx_hal as hal;
extern crate nalgebra as na;
extern crate nalgebra_glm as glm;

use std::collections::HashMap;
use std::io;
use std::mem;

use hal::format::Format;
use hal::pso::{
    AttributeDesc,
    Element,
    VertexBufferDesc,
    VertexInputRate};

use crate::buffer_object::{BlocksSource, FlatBlock, VertexBlock};

/// Vertex with position in 3D space.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Vertex {
    /// x, y, z coordinates of the vertex.
    pub xyz: [f32; 3],
    /// normal vector of the vertex.
    pub vn: [f32; 3],
}

impl VertexBlock for Vertex {
    fn vertex_buffer_attributes() -> Vec<VertexBufferDesc> {
        vec![VertexBufferDesc {
            binding: 0,
            stride: mem::size_of::<Self>() as u32,
            rate: VertexInputRate::Vertex,
        }]
    }

    fn vertex_attributes() -> Vec<AttributeDesc> {
        let xyz_offset = 0;
        let vn_offset = mem::size_of::<[f32; 3]>() as u32;
        vec![
            AttributeDesc {
                location: 0,
                binding: 0,
                element: Element {
                    format: Format::Rgb32Sfloat,
                    offset: xyz_offset,
                },
            },
            AttributeDesc {
                location: 1,
                binding: 0,
                element: Element {
                    format: Format::Rgb32Sfloat,
                    offset: vn_offset,
                },
            },
        ]
    }
}

/// Triangle in 3D space.
#[derive(Debug, Clone)]
#[repr(C)]
pub struct Triangle {
    pub vertices: [Vertex; 3],
}

impl FlatBlock for Triangle {
    fn stride_size() -> u32 {
        return mem::size_of::<Self>() as u32;
    }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct Mesh {
    pub triangles: Vec<Triangle>,
    pub transform: glm::Mat4,
}

impl FlatBlock for Mesh {
    fn stride_size() -> u32 {
        return mem::size_of::<glm::Mat4>() as u32;
    }
}

/// Implementation of mesh.
impl Mesh {

    /// Load model from obj stream reader.
    /// Perhaps move it to separate type.
    pub fn load_from_obj<R: io::BufRead>(reader: R, scale: Option<f32>) -> io::Result<Mesh> {

        // Create mappings from vertices and faces into their infos.
        let mut vertex_pose: HashMap<usize, [f32; 3]> = HashMap::with_capacity(2_000_000);
        let mut vertex_face: HashMap<usize, Vec<usize>> = HashMap::with_capacity(2_000_000);
        let mut face_vertex: HashMap<usize, [usize; 3]> = HashMap::with_capacity(2_000_000);

        // Parse .obj lines.
        for line in reader.lines() {
            let line = line.unwrap();
            let segments: Vec<&str> = line.split_ascii_whitespace().collect();
            if segments[0] == "v" && segments.len() == 4 {
                let x = segments[1].parse::<f32>().unwrap();
                let y = segments[2].parse::<f32>().unwrap();
                let z = segments[3].parse::<f32>().unwrap();
                let idx = vertex_pose.len() + 1;
                vertex_pose.insert(idx, [x, y, z]);
            } else if segments[0] == "f" && segments.len() == 4 {
                let v1_idx = segments[1].parse::<usize>().unwrap();
                let v2_idx = segments[2].parse::<usize>().unwrap();
                let v3_idx = segments[3].parse::<usize>().unwrap();
                let face_idx = face_vertex.len() + 1;
                face_vertex.insert(face_idx, [v1_idx, v2_idx, v3_idx]);
                vertex_face.entry(v1_idx).and_modify(|e| e.push(face_idx)).or_insert(vec![face_idx]);
                vertex_face.entry(v2_idx).and_modify(|e| e.push(face_idx)).or_insert(vec![face_idx]);
                vertex_face.entry(v3_idx).and_modify(|e| e.push(face_idx)).or_insert(vec![face_idx]);
            }
        }

        // Calculate normales for each face (triangle).
        let face_norms: HashMap<usize, glm::Vec3> = face_vertex.iter().map(|(face_idx, face_vertices)| {
            let v1_idx = face_vertices[0];
            let v2_idx = face_vertices[1];
            let v3_idx = face_vertices[2];
            let v1 = vertex_pose.get(&v1_idx).unwrap();
            let v2 = vertex_pose.get(&v2_idx).unwrap();
            let v3 = vertex_pose.get(&v3_idx).unwrap();
            let face_norm = glm::triangle_normal(
                &glm::Vec3::new(v1[0], v1[1], v1[2]),
                &glm::Vec3::new(v2[0], v2[1], v2[2]),
                &glm::Vec3::new(v3[0], v3[1], v3[2]),
            );
            (*face_idx, face_norm)
        }).collect();

        // Calculate normales for each vertex by averaging all it's triangle normales.
        let vertex_norms: HashMap<usize, [f32; 3]> = vertex_face.iter().map(|(vertex_id, vertex_faces)| {
            let mut norm = glm::Vec3::new(0.0, 0.0, 0.0);
            for f_id in vertex_faces {
                norm += face_norms.get(f_id).unwrap();
            }
            norm = glm::normalize(&norm);
            (*vertex_id, [norm.x, norm.y, norm.z])
        }).collect();


        let triangles: Vec<Triangle> = face_vertex.iter().map(|(_face_id, face_vertices)| {
            let v1_idx = face_vertices[0];
            let v2_idx = face_vertices[1];
            let v3_idx = face_vertices[2];

            let v1 = vertex_pose.get(&v1_idx).unwrap();
            let v2 = vertex_pose.get(&v2_idx).unwrap();
            let v3 = vertex_pose.get(&v3_idx).unwrap();

            let n1 = vertex_norms.get(&v1_idx).unwrap();
            let n2 = vertex_norms.get(&v2_idx).unwrap();
            let n3 = vertex_norms.get(&v3_idx).unwrap();

            Triangle {
                vertices: [
                    Vertex { xyz: *v1, vn: *n1 },
                    Vertex { xyz: *v2, vn: *n2 },
                    Vertex { xyz: *v3, vn: *n3 },
                ],
            }
        }).collect();

        let mut m = glm::Mat4::identity();
        if let Some(factor) = scale {
            let m_scale = glm::Vec3::new(factor, factor, factor);
            m = glm::scale(&m, &m_scale);
        }

        Ok(Mesh { transform: m, triangles })
    }
}

/// Implementation of FlatObjectContainer for Model.
impl BlocksSource<Triangle> for Mesh {
    fn len(&self) -> u32 {
        return self.triangles.len() as u32;
    }
    fn ptr(&self) -> *const u8 {
        return &self.triangles[0] as *const _ as *const u8;
    }
}
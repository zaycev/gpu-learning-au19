extern crate gfx_hal as hal;
extern crate nalgebra as na;
extern crate nalgebra_glm as glm;

use hal::format::Format;
use hal::pso::{AttributeDesc, Element, VertexBufferDesc, VertexInputRate};

use std::mem;

use std::collections::HashMap;
use std::io;

use log;

use crate::flat_object::{FlatObject, FlatObjectContainer};

/// Vertex with position in 3D space.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Vertex {
    /// x, y, z coordinates of the vertex.
    pub xyz: [f32; 3],
    /// normal vector of the vertex.
    pub vn: [f32; 3],
}

pub trait VertexBufferPrimitive {
    /// Returns vertex buffer description.
    fn vertex_buffer_attributes() -> Vec<VertexBufferDesc>;
    /// Returns vertex buffer attribute descriptions.
    fn vertex_attributes() -> Vec<AttributeDesc>;
}

impl FlatObject for Vertex {
    fn stride_size() -> u32 {
        mem::size_of::<Self>() as u32
    }
}

/// Implementation for vertex.
impl VertexBufferPrimitive for Vertex {

    fn vertex_buffer_attributes() -> Vec<VertexBufferDesc> {
        vec![VertexBufferDesc {
            binding: 0,
            stride: Self::stride_size(),
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
#[derive(Debug)]
#[repr(C)]
pub struct Triangle {
    pub vertices: [Vertex; 3],
}

impl FlatObject for Triangle {
    fn stride_size() -> u32 {
        mem::size_of::<Self>() as u32
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct Model {
    pub triangles: Vec<Triangle>,
    pub m: glm::Mat4,
}

/// Implementation of Model.
impl Model {

    /// Load model from obj stream reader.
    /// Perhaps move it to separate type.
    pub fn load_from_obj<R: io::BufRead>(reader: R, scale: f32) -> io::Result<Model> {

        let mut triangles: Vec<Triangle> = Vec::new();
        let mut vertices: HashMap<usize, Vertex> = HashMap::new();

        for line in reader.lines() {
            let line = line.unwrap();
            let msg = format!("cannot parse: {}", line);
            let segments: Vec<&str> = line.split_ascii_whitespace().collect();
            if segments[0] == "v" && segments.len() == 4 {
                let x = segments[1].parse::<f32>().expect(msg.as_str());
                let y = segments[2].parse::<f32>().expect(msg.as_str());
                let z = segments[3].parse::<f32>().expect(msg.as_str());
                let idx = vertices.len() + 1;
                vertices.insert(
                    idx,
                    Vertex {
                        xyz: [x, y, z],
                        vn: [0.0, 0.0, 0.0],
                    },
                );
            } else if segments[0] == "f" && segments.len() == 4 {
                let v1_idx = segments[1].parse::<usize>().expect(msg.as_str());
                let v2_idx = segments[2].parse::<usize>().expect(msg.as_str());
                let v3_idx = segments[3].parse::<usize>().expect(msg.as_str());
                let mut v1 = vertices.get(&v1_idx).unwrap().clone();
                let mut v2 = vertices.get(&v2_idx).unwrap().clone();
                let mut v3 = vertices.get(&v3_idx).unwrap().clone();

                let vn = glm::triangle_normal(
                    &glm::Vec3::new(v1.xyz[0], v1.xyz[1], v1.xyz[2]),
                    &glm::Vec3::new(v2.xyz[0], v2.xyz[1], v2.xyz[2]),
                    &glm::Vec3::new(v3.xyz[0], v3.xyz[1], v3.xyz[2]),
                );

                v1.vn = [vn.x, vn.y, vn.z];
                v2.vn = [vn.x, vn.y, vn.z];
                v3.vn = [vn.x, vn.y, vn.z];

                triangles.push(Triangle {
                    vertices: [v1, v2, v3],
                })
            }
        }

        log::debug!("total triangles: {}", triangles.len());
        log::debug!("total vertices: {}", triangles.len() * 3);

        let mut m = glm::Mat4::identity();
        let m_scale = glm::Vec3::new(scale, scale, scale);

        m = glm::scale(&m, &m_scale);
        triangles.shrink_to_fit();

        Ok(Model { m, triangles })
    }
}

/// Implementation of FlatObjectContainer for Model.
impl FlatObjectContainer<Triangle> for Model {

    fn flat_size(&self) -> u32 {
        return self.triangles.len() as u32 * Triangle::stride_size();
    }

    fn flat_ptr(&self) -> *const u8 {
        return &self.triangles[0] as *const _ as *const u8;
    }

}
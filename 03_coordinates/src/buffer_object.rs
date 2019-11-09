extern crate gfx_hal as hal;
extern crate nalgebra as na;
extern crate nalgebra_glm as glm;

use hal::pso::{AttributeDesc, VertexBufferDesc};


pub trait FlatObject {
    fn stride_size() -> u32;
}

pub trait FlatObjectContainer<O:FlatObject> {
    fn flat_size(&self) -> u32;
    fn flat_ptr(&self) -> *const u8;
}

pub trait VertexObject {
    const BUFFER_DESCRIPTORS: Vec<VertexBufferDesc>;
    const ATTRIBUTE_DESCRIPTORS: Vec<AttributeDesc>;
}

pub trait UniformObject {

}
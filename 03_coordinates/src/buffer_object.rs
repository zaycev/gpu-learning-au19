extern crate gfx_hal as hal;
extern crate nalgebra as na;
extern crate nalgebra_glm as glm;

use hal::pso::{AttributeDesc, VertexBufferDesc};

pub trait FlatBlock {
    fn stride_size() -> u32;
}

pub trait BlocksSource<O: FlatBlock> {
    /// Returns a total number of elements.
    fn len(&self) -> u32;
    /// Pointer to the source of elements.
    fn ptr(&self) -> *const u8;
    /// Returns a total number of bytes occupied by all elements.
    fn src_size(&self) -> u32 {
        return self.len() * O::stride_size();
    }
}

pub trait VertexBlock {
    /// Returns attributes of vertex buffer.
    fn vertex_buffer_attributes() -> Vec<VertexBufferDesc>;
    /// Returns vertex element attributes.
    fn vertex_attributes() -> Vec<AttributeDesc>;
}
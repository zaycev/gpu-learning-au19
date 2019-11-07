extern crate gfx_backend_metal as back;
extern crate gfx_hal as hal;
extern crate nalgebra as na;
extern crate nalgebra_glm as glm;

use hal::adapter::{Adapter, Gpu, PhysicalDevice};
use hal::buffer;
use hal::device::Device;
use hal::memory;
use hal::Backend;

use std::iter;
use std::marker::PhantomData;
use std::ptr;

use crate::geometry::VertexBufferPrimitive;
use crate::Model;

/// Holds vertex buffer object and associated memory.
pub struct VertexBuffer<B: Backend, V: VertexBufferPrimitive> {
    pub vertex_buffer: B::Buffer,
    pub vertex_memory: B::Memory,
    pub vertex_memory_size: u64,
    _phantom: PhantomData<V>,
}

impl<B: Backend, V: VertexBufferPrimitive> VertexBuffer<B, V> {
    const BUFFER_USAGE: buffer::Usage = buffer::Usage::VERTEX;
    const BUFFER_MEM_TYPE: memory::Properties = memory::Properties::CPU_VISIBLE;

    pub fn new(adapter: &Adapter<B>, gpu: &Gpu<B>, capacity: u64) -> Self {
        let buffer_stride_size = V::stride_size() * 3;
        let buffer_total_size = capacity * buffer_stride_size as u64;
        let mut vertex_buffer = unsafe {
            gpu.device
                .create_buffer(buffer_total_size, Self::BUFFER_USAGE)
                .unwrap()
        };
        let buffer_requirements = unsafe { gpu.device.get_buffer_requirements(&vertex_buffer) };
        let vertex_memory_size = buffer_requirements.size;
        let gpu_memory_types = adapter.physical_device.memory_properties().memory_types;
        let buffer_memory_type_id = gpu_memory_types
            .iter()
            .enumerate()
            .position(|(mem_id, mem_type)| {
                let mem_id_u64 = mem_id as u64;
                let mem_supports_buf_req = buffer_requirements.type_mask & (mem_id_u64 << 1) != 0;
                let mem_supports_mem_type = mem_type.properties.contains(Self::BUFFER_MEM_TYPE);
                mem_supports_buf_req && mem_supports_mem_type
            })
            .unwrap()
            .into();
        let vertex_memory = unsafe {
            gpu.device
                .allocate_memory(buffer_memory_type_id, vertex_memory_size)
                .unwrap()
        };
        unsafe {
            gpu.device
                .bind_buffer_memory(&vertex_memory, 0, &mut vertex_buffer)
                .unwrap();
        }

        Self {
            vertex_buffer,
            vertex_memory,
            vertex_memory_size,
            _phantom: PhantomData,
        }
    }

    /// Writes model data to vertex buffer and returns number of written vertices.
    pub fn write(&mut self, gpu: &Gpu<B>, model: &Model) -> u32 {
        if model.triangles.len() == 0 {
            return 0;
        }

        // Calculate the size of data to copy and flush.
        let stride_size = V::stride_size() * 3;
        let total_copy_size = model.triangles.len() * stride_size as usize;
        let total_copy_size_u64 = total_copy_size as u64;

        // Copy whole triangle vector and flush.
        unsafe {
            let dst_range = 0..total_copy_size_u64;
            let dst_ptr = gpu.device.map_memory(&self.vertex_memory, dst_range).unwrap();
            let src_ptr: *const u8 = &model.triangles[0] as *const _ as *const _;
            let flush_range = iter::once((&self.vertex_memory, 0..total_copy_size_u64));
            ptr::copy_nonoverlapping(src_ptr, dst_ptr, total_copy_size);
            gpu.device.flush_mapped_memory_ranges(flush_range).unwrap();
        }

        (model.triangles.len() * 3) as u32
    }
}

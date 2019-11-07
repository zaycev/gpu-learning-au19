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

use crate::flat_object::{FlatObject, FlatObjectContainer};


/// Holds buffer and its memory for storing buffer objects.
pub struct Buffer<B: Backend, O: FlatObject> {
    pub buffer: B::Buffer,
    pub memory: B::Memory,
    pub memory_size: u64,

    _len: u32,
    _phantom: PhantomData<O>,
}

impl<B: Backend, O: FlatObject> Buffer<B, O> {

    pub fn new(
        adapter: &Adapter<B>,
        gpu: &Gpu<B>,
        capacity: u64,
        memory_usage: buffer::Usage,
        memory_type: memory::Properties) -> Self {

        // Create buffer.
        let stride_size = O::stride_size();
        let total_size = capacity * stride_size as u64;
        let mut buffer = unsafe { gpu.device.create_buffer(total_size, memory_usage).unwrap() };

        // Find memory type id supporting requirements and requested memory type.
        let buffer_requirements = unsafe { gpu.device.get_buffer_requirements(&buffer) };
        let available_types = adapter.physical_device.memory_properties().memory_types;
        let memory_type_id = available_types
            .iter()
            .enumerate()
            .position(|(mem_id, mem_type)| {
                let mem_id_u64 = mem_id as u64;
                let mem_supports_buf_req = buffer_requirements.type_mask & (mem_id_u64 << 1) != 0;
                let mem_supports_mem_type = mem_type.properties.contains(memory_type);
                mem_supports_buf_req && mem_supports_mem_type
            })
            .unwrap()
            .into();

        // Allocate memory and bind memory with buffer.
        let memory_size = buffer_requirements.size;
        let memory = unsafe { gpu.device.allocate_memory(memory_type_id, memory_size) .unwrap() };
        unsafe {
            gpu.device.bind_buffer_memory(&memory, 0, &mut buffer).unwrap();
        }

        Self {
            buffer,
            memory,
            memory_size,
            _len: 0,
            _phantom: PhantomData,
        }
    }

    /// Shortcut for making a new vertex buffer.
    pub fn create_vertex_buffer(adapter: &Adapter<B>, gpu: &Gpu<B>, capacity: u64,) -> Self {
        Self::new(adapter, gpu, capacity, buffer::Usage::VERTEX, memory::Properties::DEVICE_LOCAL)
    }

    /// Returns number elements currently stored in the buffer.
    pub fn len(&self) -> u32 {
        return self._len;
    }

    /// Writes flat data from source to buffer and flushes it.
    pub fn write<C>(&mut self, gpu: &Gpu<B>, src: &C)
    where C: FlatObjectContainer<O>
    {
        if src.flat_size() == 0 {
            return;
        }

        // Calculate the size of data to copy.
        let total_copy_size = src.flat_size() as usize;
        let total_copy_size_u64 = src.flat_size() as u64;
        let dst_range = 0..total_copy_size_u64;
        let src_ptr: *const u8 = src.flat_ptr();
        let flush_range = iter::once((&self.memory, 0..total_copy_size_u64));

        // Copy container flat data to buffer and flush.
        unsafe {
            let dst_ptr = gpu.device.map_memory(&self.memory, dst_range).unwrap();
            ptr::copy_nonoverlapping(src_ptr, dst_ptr, total_copy_size);
            gpu.device.flush_mapped_memory_ranges(flush_range).unwrap();
        }

        self._len = src.flat_size() / O::stride_size();
    }
}

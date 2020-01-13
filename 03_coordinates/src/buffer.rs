extern crate gfx_backend_metal as back;
extern crate gfx_hal as hal;
extern crate nalgebra as na;
extern crate nalgebra_glm as glm;

use std::iter;
use std::marker::PhantomData;
use std::ptr;

use hal::adapter::{Adapter, Gpu, PhysicalDevice};
use hal::Backend;
use hal::buffer;
use hal::device::Device;
use hal::memory;

use crate::buffer_object::{BlocksSource, FlatBlock};

/// Holds buffer and its memory for storing flat objects.
pub struct Buffer<B: Backend, O: FlatBlock> {
    pub buffer: B::Buffer,
    pub memory: B::Memory,
    pub memory_size: u64,
    _len: u32,
    _phantom: PhantomData<O>,
}

impl<B: Backend, O: FlatBlock> Buffer<B, O> {
    pub fn new(
        adapter: &Adapter<B>,
        gpu: &Gpu<B>,
        capacity: u64,
        memory_usage: buffer::Usage,
        memory_type: memory::Properties,
        name: String) -> Self {

        // Create buffer.
        let stride_size = O::stride_size() as u64;
        let total_size = capacity * stride_size;
        let mut buffer = unsafe {
            gpu.device.create_buffer(total_size, memory_usage).unwrap()
        };

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
        let memory = unsafe { gpu.device.allocate_memory(memory_type_id, memory_size).unwrap() };
        unsafe {
            gpu.device.set_buffer_name(&mut buffer, name.as_str());
            gpu.device.bind_buffer_memory(&memory, 0, &mut buffer).unwrap();
        }

        unsafe {
            let dst_range = 0..memory_size;
            let flush_range = iter::once((&memory, 0..memory_size));
            let dst_ptr = gpu.device.map_memory(&memory, dst_range).unwrap();
            ptr::write_bytes(dst_ptr, 0, memory_size as usize);
            gpu.device.flush_mapped_memory_ranges(flush_range).unwrap();
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
    pub fn new_vertex_buffer(adapter: &Adapter<B>, gpu: &Gpu<B>, capacity: u64, name: String) -> Self {
        Self::new(adapter, gpu, capacity, buffer::Usage::VERTEX, memory::Properties::DEVICE_LOCAL, name)
    }

    /// Shortcut for making a new uniform buffer.
    pub fn new_uniform(adapter: &Adapter<B>, gpu: &Gpu<B>, capacity: u64, name: String) -> Self {
        Self::new(adapter, gpu, capacity, buffer::Usage::UNIFORM, memory::Properties::DEVICE_LOCAL, name)
    }

    /// Returns number elements currently stored in the buffer.
    pub fn len(&self) -> u32 {
        return self._len;
    }

    /// Copies copy_size number of bytes starting from src pointer.
    pub fn copy_from_ptr(&mut self, gpu: &Gpu<B>, src: *const u8, copy_size: usize) {

        if copy_size == 0 {
            panic!("nothing to copy");
        }

        let copy_size_u64 = copy_size as u64;
        let dst_start_u64 = (self._len * O::stride_size()) as u64;
        let dst_end_u64 = dst_start_u64 + copy_size_u64;
        let dst_range = dst_start_u64..dst_end_u64;
        let flush_range = iter::once((&self.memory, dst_start_u64..dst_end_u64));

        unsafe {
            let dst_ptr = gpu.device.map_memory(&self.memory, dst_range).unwrap();
            ptr::copy_nonoverlapping(src, dst_ptr, copy_size);
            gpu.device.flush_mapped_memory_ranges(flush_range).unwrap();
        }

        self._len += copy_size as u32 / O::stride_size();
    }

    pub fn copy_from_vec(&mut self, gpu: &Gpu<B>, src: &Vec<O>) {
        if src.len() == 0 {
            return;
        }
        let ptr = &src[0] as *const _ as *const u8;
        self.copy_from_ptr(gpu, ptr, src.len());
    }

    /// Copies data from source to buffer.
    pub fn copy_from_src<C>(&mut self, gpu: &Gpu<B>, src: &C)
        where C: BlocksSource<O>
    {
        self.copy_from_ptr(gpu, src.ptr(), src.src_size() as usize);
    }

    pub fn reset(&mut self) {
        self._len = 0;
    }
}

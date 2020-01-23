extern crate gfx_backend_metal as back;
extern crate gfx_hal as hal;

use std::iter;
use std::marker::PhantomData;
use std::ptr;
use std::mem;

use hal::adapter::{Adapter, Gpu, PhysicalDevice};
use hal::Backend;
use hal::buffer;
use hal::device::Device;
use hal::memory;

/// Holds buffer and its memory.
///
pub struct Buffer<B: Backend, T:Sized> {
    pub buffer: B::Buffer,
    pub memory: B::Memory,
    pub memory_size: usize,

    size: usize,
    capacity: usize,

    phantom: PhantomData<T>,
}

/// Implementation.
///
impl<B: Backend, T:Sized> Buffer<B, T> {

    /// Creates a new buffer and allocates memory for it.
    ///
    pub fn new(
        adapter: &Adapter<B>,
        gpu: &Gpu<B>,
        capacity: usize,
        memory_usage: buffer::Usage,
        memory_type: memory::Properties,
    ) -> Self {

        let stride = mem::size_of::<T>();

        // Create buffer.
        let total_size = (capacity * stride) as u64;
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
            gpu.device.bind_buffer_memory(&memory, 0, &mut buffer).unwrap();
        }

        Self {
            buffer,
            memory,
            capacity,
            size: 0,
            memory_size: memory_size as usize,
            phantom: PhantomData,
        }
    }

    /// Sets buffer name for a given device.
    ///
    pub fn set_name(&mut self, gpu: &Gpu<B>, name: &str) {
        unsafe {
            gpu.device.set_buffer_name(&mut self.buffer, name);
        }
    }

    /// Shortcut for making a new vertex buffer.
    pub fn new_vertex_buffer(adapter: &Adapter<B>, gpu: &Gpu<B>, capacity: usize) -> Self {
        Self::new(
            adapter,
            gpu,
            capacity,
            buffer::Usage::VERTEX,
            memory::Properties::DEVICE_LOCAL,
        )
    }

    /// Shortcut for making a new uniform buffer.
    pub fn new_uniform_buffer(adapter: &Adapter<B>, gpu: &Gpu<B>, capacity: usize) -> Self {
        Self::new(
            adapter,
            gpu,
            capacity,
            buffer::Usage::UNIFORM,
            memory::Properties::DEVICE_LOCAL,
        )
    }

    /// Returns current number of elements in the buffer.
    ///
    pub fn size(&self) -> usize {
        return self.size;
    }

    /// Resets buffer size and memory.
    ///
    pub fn reset(&mut self) {
        self.size = 0;
    }

    /// Copy num objects of size `mem::size_of::<T>()` from src pointer.
    ///
    pub fn copy(&mut self, gpu: &Gpu<B>, src: *const u8, num: usize) {

        if num == 0 {
            panic!("nothing to copy");
        }
        if self.size + num > self.capacity {
            panic!("not enough capacity");
        }

        // Calculate copy range.
        let stride = mem::size_of::<T>();
        let size = stride * num;
        let offset = stride * self.size;

        let dts_start = offset as u64;
        let dst_end = (offset + size) as u64;
        let flush_range = iter::once((&self.memory, dts_start..dst_end));

        unsafe {
            let dst_ptr = gpu.device
                .map_memory(&self.memory, dts_start..dst_end)
                .unwrap();
            ptr::copy_nonoverlapping(src, dst_ptr, size);
            gpu.device.flush_mapped_memory_ranges(flush_range).unwrap();
        }

        self.size += num;
    }
}



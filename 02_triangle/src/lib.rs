extern crate gfx_backend_metal as back;
extern crate gfx_hal as hal;

use hal::adapter::{Gpu, Adapter, PhysicalDevice};
use hal::device::Device;
use hal::pso;
use hal::pso::DescriptorPool;
use hal::{Backend};
use hal::buffer;
use hal::memory;
use hal::pool;
use hal::queue::family::{QueueFamily, QueueFamilyId, QueueGroup};

use shaderc;
use std::mem;
use std::ptr;

pub struct ComputeExample<B:Backend> {
    pub pipeline: B::ComputePipeline,
    pub pipeline_layout: B::PipelineLayout,
    pub descriptor_set_layout: B::DescriptorSetLayout,
    pub descriptor_pool: B::DescriptorPool,

    pub staging_memory: B::Memory,
    pub staging_buffer: B::Buffer,
    pub staging_size: u64,

    pub device_memory: B::Memory,
    pub device_buffer: B::Buffer,
    pub device_size: u64,

    pub family: QueueFamilyId,
}


impl <B:Backend> ComputeExample<B> {


    pub fn new(gpu: &Gpu<B>, adapter: &Adapter<B>, queue_group: &QueueGroup<B>) -> Self {

        let mut shader_compiler = shaderc::Compiler::new().unwrap();
        let shader_entry = "main";
        let shader_file = "vertex.comp";
        let shader_spirv = shader_compiler.compile_into_spirv(
            include_str!("vertex.comp"),
            shaderc::ShaderKind::Compute,
            shader_file,
            shader_entry,
            None,
        ).unwrap();

        let shader_module = unsafe {
            gpu.device
                .create_shader_module(shader_spirv.as_binary())
                .unwrap()
        };

        let descriptor_set_layout = unsafe {
            gpu.device.create_descriptor_set_layout(&[pso::DescriptorSetLayoutBinding{
                binding: 0,
                ty: pso::DescriptorType::StorageBuffer,
                count: 1,
                stage_flags: pso::ShaderStageFlags::COMPUTE,
                immutable_samplers: false,
            }], &[]).unwrap()
        };

        let pipeline_layout = unsafe {
            gpu.device.create_pipeline_layout(Some(&descriptor_set_layout), &[]).unwrap()
        };

        let entry_point = pso::EntryPoint{
            entry: "main",
            module: &shader_module,
            specialization: pso::Specialization::default(),
        };

        let pipeline = unsafe {
            gpu.device.create_compute_pipeline(
                &pso::ComputePipelineDesc::new(entry_point, &pipeline_layout),
                None,
            ).unwrap()
        };

        let descriptor_pool = unsafe {
            gpu.device.create_descriptor_pool(
                1,
                &[pso::DescriptorRangeDesc {
                    ty: pso::DescriptorType::StorageBuffer,
                    count: 1,
                }],
                pso::DescriptorPoolCreateFlags::empty(),
            ).unwrap()
        };

        let memory_types = adapter.physical_device.memory_properties().memory_types;
        let memory_stride = (mem::size_of::<f32>() * 2) as u64;
        let memory_len = 2;

        /// Create staging memory.

        let mut staging_buffer = unsafe {
            let usage = buffer::Usage::TRANSFER_SRC | buffer::Usage::TRANSFER_DST;
            gpu.device.create_buffer(memory_stride * memory_len, usage).unwrap()
        };
        let staging_buffer_requirements = unsafe {
            gpu.device.get_buffer_requirements(&staging_buffer)
        };

        let staging_memory_type = memory_types.clone().into_iter()
            .enumerate()
            .position(|(id, memory_type)| {
                let is_cpu = memory_type.properties.contains(memory::Properties::CPU_VISIBLE);
                let is_supported = staging_buffer_requirements.type_mask & (1 << id) != 0;
                is_cpu && is_supported
            })
            .unwrap()
            .into();

        let staging_memory = unsafe {
            gpu.device.allocate_memory(
                staging_memory_type,
                staging_buffer_requirements.size,
            ).unwrap()
        };

        unsafe {
            gpu.device.bind_buffer_memory(&staging_memory, 0, &mut staging_buffer).unwrap();
        }


        /// Create device memory.

        let mut device_buffer = unsafe {
            let usage = buffer::Usage::TRANSFER_SRC | buffer::Usage::TRANSFER_DST | buffer::Usage::STORAGE;
            gpu.device.create_buffer(memory_stride * memory_len, usage).unwrap()
        };

        let device_buffer_requirements = unsafe {
            gpu.device.get_buffer_requirements(&device_buffer)
        };

        let device_memory_type = memory_types.clone().into_iter()
            .enumerate()
            .position(|(id, memory_type)| {
                let is_local = memory_type.properties.contains(memory::Properties::DEVICE_LOCAL);
                let is_supported = device_buffer_requirements.type_mask & (1 << id) != 0;
                is_local && is_supported
            })
            .unwrap()
            .into();

        let device_memory = unsafe {
            gpu.device.allocate_memory(
                device_memory_type,
                device_buffer_requirements.size,
            ).unwrap()
        };

        unsafe {
            gpu.device.bind_buffer_memory(&device_memory, 0, &mut device_buffer).unwrap();
        }

        Self{
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            staging_memory,
            staging_buffer,
            staging_size: staging_buffer_requirements.size,
            device_memory,
            device_buffer,
            device_size: device_buffer_requirements.size,
            family: queue_group.family,
        }
    }

    pub fn compute(&mut self, gpu: &Gpu<B>, xy: [f32; 2]) -> [f32; 4] {

        let memory_stride = mem::size_of::<f32>() * 2;
        let memory_len = 2;
        let xyxy = [xy[0], xy[1], xy[0], xy[1]];

        unsafe {
            let mapping = gpu.device.map_memory(&self.staging_memory, 0 .. self.staging_size).unwrap();
            ptr::copy_nonoverlapping(xyxy.as_ptr() as *const u8, mapping, memory_stride * memory_len);
        }


        let desc_set;
        unsafe {
            desc_set = self.descriptor_pool.allocate_set(&self.descriptor_set_layout).unwrap();
            gpu.device.write_descriptor_sets(Some(pso::DescriptorSetWrite {
                set: &desc_set,
                binding: 0,
                array_offset: 0,
                descriptors: Some(pso::Descriptor::Buffer(&self.device_buffer, None .. None)),
            }));
        };


//        let mut command_pool = unsafe {
//            gpu.device.create_command_pool(self.family, pool::CommandPoolCreateFlags::empty()).unwrap()
//        };


//        let fence = device.create_fence(false).unwrap();
//        unsafe {
//            let mut command_buffer = command_pool.allocate_one(command::Level::Primary);
//            command_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);
//            command_buffer.copy_buffer(
//                &staging_buffer,
//                &device_buffer,
//                &[command::BufferCopy {
//                    src: 0,
//                    dst: 0,
//                    size: stride * numbers.len() as u64,
//                }],
//            );
//            command_buffer.pipeline_barrier(
//                pso::PipelineStage::TRANSFER .. pso::PipelineStage::COMPUTE_SHADER,
//                memory::Dependencies::empty(),
//                Some(memory::Barrier::Buffer {
//                    states: buffer::Access::TRANSFER_WRITE
//                        .. buffer::Access::SHADER_READ | buffer::Access::SHADER_WRITE,
//                    families: None,
//                    target: &device_buffer,
//                    range: None .. None,
//                }),
//            );
//            command_buffer.bind_compute_pipeline(&pipeline);
//            command_buffer.bind_compute_descriptor_sets(&pipeline_layout, 0, &[desc_set], &[]);
//            command_buffer.dispatch([numbers.len() as u32, 1, 1]);
//            command_buffer.pipeline_barrier(
//                pso::PipelineStage::COMPUTE_SHADER .. pso::PipelineStage::TRANSFER,
//                memory::Dependencies::empty(),
//                Some(memory::Barrier::Buffer {
//                    states: buffer::Access::SHADER_READ | buffer::Access::SHADER_WRITE
//                        .. buffer::Access::TRANSFER_READ,
//                    families: None,
//                    target: &device_buffer,
//                    range: None .. None,
//                }),
//            );
//            command_buffer.copy_buffer(
//                &device_buffer,
//                &staging_buffer,
//                &[command::BufferCopy {
//                    src: 0,
//                    dst: 0,
//                    size: stride * numbers.len() as u64,
//                }],
//            );
//            command_buffer.finish();
//
//            queue_group.queues[0].submit_without_semaphores(Some(&command_buffer), Some(&fence));
//
//            device.wait_for_fence(&fence, !0).unwrap();
//            command_pool.free(Some(command_buffer));
//        }
//
//        unsafe {
//            let mapping = device.map_memory(&staging_memory, 0 .. staging_size).unwrap();
//            println!(
//                "Times: {:?}",
//                slice::from_raw_parts::<u32>(mapping as *const u8 as *const u32, numbers.len()),
//            );
//            device.unmap_memory(&staging_memory);
//        }

        return [0.9, 0.7, -0.9, 0.7];
    }

}

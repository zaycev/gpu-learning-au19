extern crate gfx_backend_metal as back;
extern crate gfx_hal as hal;

use hal::adapter::{Adapter, Gpu, PhysicalDevice};
use hal::buffer;
use hal::command;
use hal::command::CommandBuffer;
use hal::device::Device;
use hal::format;
use hal::image;
use hal::memory;
use hal::pass;
use hal::pool;
use hal::pool::CommandPool;
use hal::pso;
use hal::queue;
use hal::queue::family::{QueueFamily, QueueGroup};
use hal::queue::CommandQueue;
use hal::window::{Extent2D, PresentMode, Surface, Swapchain, SwapchainConfig};
use hal::{Backend, Instance};

use shaderc;

use winit::dpi::{LogicalPosition, LogicalSize};
use winit::{Event, EventsLoop, Window, WindowBuilder, WindowEvent};

use std::iter;
use std::mem;
use std::ptr;
use std::time;

use core::ops::Range;

use log;
use simple_logger;

use class2::ComputeExample;

fn main() {

    // Setup logger.
    simple_logger::init_with_level(log::Level::Error).unwrap();

    let title = "GPU Learning Class: 02 Single Triangle pipeline".to_string();
    let name = title.clone();
    let mut events_loop = EventsLoop::new();
    let mut size = LogicalSize {
        width: 512.0,
        height: 512.0,
    };
    let window = WindowBuilder::new()
        .with_dimensions(size)
        .with_resizable(true)
        .build(&events_loop)
        .unwrap();

    // Create GPU API instance and a surface where we will draw pixels.
    let instance = back::Instance::create(name.as_str(), 1).unwrap();
    let surface = unsafe { instance.create_surface(&window).unwrap() };

    // Select adapter (a logical GPU device). Here we just use the first adapter from the list.
    let adapter = instance.enumerate_adapters().remove(0);

    // Define a default pixel format. We use standard RGBA – 8 bit RGB + Alpha.
    let default_pixel_format = Some(format::Format::Rgba8Srgb);
    let default_channel_format = format::ChannelType::Srgb;

    // Check that device supports surface – Check that surface formats contain sRGB
    // or default to sRGBA.
    let formats = surface.supported_formats(&adapter.physical_device);
    let pixel_format = formats
        .map_or(default_pixel_format, |formats| {
            formats
                .iter()
                .find(|surface_pixel_format| {
                    let channel_type = surface_pixel_format.base_format().1;
                    channel_type == default_channel_format
                })
                .map(|format| *format)
        })
        .unwrap();

    // Open a physical device and its queues (we need just 1 queue for now) for graphics commands.
    // The selector function will check if the queue can support output to our surface.
    let family = adapter
        .queue_families
        .iter()
        .find(|family| {
            let supports_surface = surface.supports_queue_family(family);
            let supports_graphics = family.queue_type().supports_graphics();
            let supports_compute = family.queue_type().supports_compute();
            supports_surface && supports_graphics && supports_compute
        })
        .unwrap();
    let gpu = unsafe {
        adapter
            .physical_device
            .open(&[(family, &[1.0])], hal::Features::empty())
            .unwrap()
    };

    // Create our engine for drawing state.
    let mut engine = GraphicsEngine::init(gpu, surface, adapter, size, pixel_format, window, title);

    // Main loop:
    //
    //  1. Get interaction event for window.
    //  2. Update state.
    //  3. Draw state.
    //
    let mut cursor_pose = LogicalPosition::new(0.0, 0.0);
    let mut end_requested = false;
    let mut resize_requested = false;
    loop {
        // If end requested, exit the loop.
        if end_requested {
            break;
        }

        // Get window event.
        events_loop.poll_events(|e| match e {
            // Catch cursor event.
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => cursor_pose = position,
            // Catch close event.
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => end_requested = true,
            // Window resize.
            Event::WindowEvent {
                event: WindowEvent::Resized(new_size),
                ..
            } => {
                resize_requested = true;
                size = new_size;
            }
            // Default.
            _ => (),
        });

        if resize_requested {
            engine.update_window_size(size);
            resize_requested = false;
        }

        // Map cursor position from screen coordinate space: [0..size.width][0..size.height] into
        //  device coordinate space: [-1.0, 1.0][-1.0, 1.0].
        let vec_0_x = (cursor_pose.x / size.width * 2.0 - 1.0) as f32;
        let vec_0_y = (cursor_pose.y / size.height * 2.0 - 1.0) as f32;

        let angle_1: f32 = 2.0944; // 120 degrees.
        let angle_2: f32 = angle_1 * 2.0;

        let vec_1_x = angle_1.cos() * vec_0_x - angle_1.sin() * vec_0_y;
        let vec_1_y = angle_1.sin() * vec_0_x + angle_1.cos() * vec_0_y;

        let vec_2_x = angle_2.cos() * vec_0_x - angle_2.sin() * vec_0_y;
        let vec_2_y = angle_2.sin() * vec_0_x + angle_2.cos() * vec_0_y;

        // Define a triangle which we will. Each vertex position defined using device
        // space for coordinates. In the next classes we are going to use world space
        // and then project them into device space.
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let triangle = Triangle {
            points: [
                vec_0_x, vec_0_y,
                vec_1_x, vec_1_y,
                vec_2_x, vec_2_y,
            ],
        };

        // Draw state.
        engine.draw(&triangle);
    }
}

/// Triangle is 3 2D point. In the next classes we will define a more
/// handy type.
pub struct Triangle {
    pub points: [f32; 6],
}

impl Triangle {
    /// Returns the size needed to store a single triangle vertex.
    fn vertex_stride() -> u32 {
        mem::size_of::<f32>() as u32 * 2
    }

    /// Returns the attributes describing the format used to store triangle data
    /// in the vertex buffer.
    fn vertex_attributes() -> Vec<pso::AttributeDesc> {
        // Essentially, each vertex in the buffer will occupy `size` number of bytes
        // And consist of a single element – x,y coordinates with position 0 and memory offset 0.
        // Since the coordinates require only two f32 numbers, we use RG (red, green) float
        // format to describe it.
        let element_location = 0;
        let element_binding = 0;
        let element_format = format::Format::Rg32Sfloat;
        let element_offset = 0;
        vec![pso::AttributeDesc {
            location: element_location,
            binding: element_binding,
            element: pso::Element {
                format: element_format,
                offset: element_offset,
            },
        }]
    }
}

/// GraphicsEngine will encapsulate the logic of drawing to surface with one gpu
/// and re-creating the swap-chain when it gets outdated (for example, when the window size gets
/// changed).
pub struct GraphicsEngine<B: Backend> {
    pub surface: B::Surface,
    pub adapter: Adapter<B>,
    pub gpu: Gpu<B>,
    pub queue_group: QueueGroup<B>,
    pub pixel_format: format::Format,
    pub clear_color: [f32; 4],
    pub window_size: LogicalSize,

    state: Option<EngineState<B>>,
    // Window stuff.
    window: Window,
    title: String,
    title_format: String,

    // Variables for counting number of frames and time spent on rendering.
    frame_i: u16,
    total_frames: u16,
    total_time: u64,
}

/// GraphicsEngine implementation.
impl<B: Backend> GraphicsEngine<B> {
    /// Creates a new graphics engine.
    pub fn init(
        mut gpu: Gpu<B>,
        mut surface: B::Surface,
        adapter: Adapter<B>,
        window_size: LogicalSize,
        pixel_format: format::Format,
        window: Window,
        title: String,
    ) -> Self {
        log::info!("using {}", adapter.info.name);
        let queue_group = gpu.queue_groups.pop().unwrap();
        let state = EngineState::new(
            &mut gpu,
            &queue_group,
            &mut surface,
            &adapter,
            window_size,
            pixel_format,
        );
        let clear_color = [0.03, 0.03, 0.03, 1.0];
        Self {
            gpu,
            queue_group,
            surface,
            pixel_format,
            adapter,
            clear_color,
            window_size,
            state: Some(state),
            title_format: title.clone(),
            title,
            window,
            frame_i: 0,
            total_frames: 0,
            total_time: 0,
        }
    }

    /// Update window color.
    pub fn update_clear_color(&mut self, color: [f32; 4]) {
        self.clear_color = color;
    }

    /// Update window size.
    pub fn update_window_size(&mut self, window_size: LogicalSize) {
        let factor = self.window.get_hidpi_factor();
        self.window_size = LogicalSize {
            width: window_size.width * factor,
            height: window_size.height * factor,
        };
        self.recreate_state();
    }

    /// Draw state draws the state and re-creates the swap-chain when it gets outdated.
    pub fn draw(&mut self, triangle: &Triangle) {
        //
        self.frame_i += 1;
        let begin_time = time::SystemTime::now();

        // Render state.
        let result = self.state.as_mut().unwrap().draw(
            &mut self.gpu,
            &mut self.queue_group.queues[0],
            self.clear_color,
            triangle,
        );

        // If draw returned error, try to recreate swap-chain and render state.
        if result.is_err() {
            log::warn!("recreating state");
            self.recreate_state();
        }

        // Measure time spent in draw function.
        let elapsed = begin_time.elapsed().unwrap().as_micros() as u64;
        self.total_frames += 1;
        self.total_time += elapsed;

        // Every 64 frame, calculate average FPS and display it the window title.
        if self.frame_i % 64 == 1 {
            let time_per_frame = (self.total_time as f64) / (self.total_frames as f64);
            let frames_per_second = 1_000_000.0 / time_per_frame;
            self.title = format!(
                "{} | FPS: {:.2} DRAW {:.2}ms",
                self.title_format,
                frames_per_second,
                time_per_frame / 1000.0
            );
            self.window.set_title(self.title.as_str());
            self.total_frames = 0;
            self.total_time = 0;
        }
    }

    /// Destroy old state and create new state.
    fn recreate_state(&mut self) {
        self.destroy_state();
        self.state = Some(EngineState::new(
            &mut self.gpu,
            &self.queue_group,
            &mut self.surface,
            &self.adapter,
            self.window_size,
            self.pixel_format,
        ));
    }

    /// Destroy all resources allocated on GPU in a reversed order.
    fn destroy_state(&mut self) {
        if let Some(mut state) = self.state.take() {
            unsafe {
                self.gpu.device.destroy_graphics_pipeline(state.pipeline);
                self.gpu
                    .device
                    .destroy_pipeline_layout(state.pipeline_layout);
                for layout in state.descriptor_set_layouts.drain(..) {
                    self.gpu.device.destroy_descriptor_set_layout(layout);
                }
                self.gpu.device.destroy_render_pass(state.render_pass);
                self.gpu.device.destroy_swapchain(state.swapchain);
                for fence in state.frame_buffer_fences.drain(..) {
                    self.gpu.device.wait_for_fence(&fence, !0).unwrap();
                    self.gpu.device.destroy_fence(fence);
                }
                let pools = state.command_buffer_pools.drain(..);
                let pools_list = state.command_buffer_pools_lists.drain(..);
                for (mut pool, pool_list) in pools.zip(pools_list) {
                    pool.free(pool_list);
                    self.gpu.device.destroy_command_pool(pool);
                }
                for semaphore in state.acquire_semaphores.drain(..) {
                    self.gpu.device.destroy_semaphore(semaphore);
                }
                for semaphore in state.present_semaphores.drain(..) {
                    self.gpu.device.destroy_semaphore(semaphore);
                }
                for frame_buffer in state.frame_buffer_list.drain(..) {
                    self.gpu.device.destroy_framebuffer(frame_buffer);
                }
                for (_, image_view) in state.frame_buffer_images.drain(..) {
                    self.gpu.device.destroy_image_view(image_view);
                }
            }
        }
    }
}

/// Manually destroy all resources allocated on GPU when graphics engine is dropped.
impl<B: Backend> Drop for GraphicsEngine<B> {
    fn drop(&mut self) {
        self.destroy_state();
    }
}

/// Engine will be encapsulating the state management and the drawing logic.
struct EngineState<B: Backend> {
    // Swap-chain and a back-buffer. The back-buffer is the secondary buffer which is not currently
    // being displayed. The back buffer then becomes the front buffer (and vice versa).
    pub size: Extent2D,
    pub swapchain: B::Swapchain,

    // Render pass describing output of graphics pipeline, for example, color attachments and their
    // pixel format.
    pub render_pass: B::RenderPass,

    // Frame buffers and corresponding stuff for each image in the swap-chain.
    // Each frame in the swap-chain will have a corresponding frame-buffer, a fence for
    // synchronizing CPU and GPU, the actual image and it's view, semaphores, and command pools
    // with currently allocated command buffers.
    pub frame_buffer_list: Vec<B::Framebuffer>,
    pub frame_buffer_fences: Vec<B::Fence>,
    pub frame_buffer_images: Vec<(B::Image, B::ImageView)>,
    pub acquire_semaphores: Vec<B::Semaphore>,
    pub present_semaphores: Vec<B::Semaphore>,
    pub command_buffer_pools: Vec<B::CommandPool>,
    pub command_buffer_pools_lists: Vec<Vec<B::CommandBuffer>>,

    // Graphic pipeline state objects and vertex buffer.
    //
    //
    pub pipeline: B::GraphicsPipeline,
    pub pipeline_layout: B::PipelineLayout,
    pub descriptor_set_layouts: Vec<B::DescriptorSetLayout>,
    pub vertex_memory: B::Memory,
    pub vertex_buffer: B::Buffer,
    pub vertex_memory_size: u64,

    // Compute pipeline
    pub compute: ComputeExample<B>,

    // Index of the currently displayed image in the swap-chain.
    // Needed to acquire buffers, semaphores and fences corresponding to the current image
    // in the back buffer of the swap-chain.
    sem_index: usize,
    frame_counter: u32,
}

/// Engine implementation.
impl<B: Backend> EngineState<B> {
    /// Creates a new graphics engine.
    pub fn new(
        gpu: &mut Gpu<B>,
        queues: &QueueGroup<B>,
        surface: &mut B::Surface,
        adapter: &Adapter<B>,
        window_size: LogicalSize,
        pixel_format: format::Format,
    ) -> Self {
        // 1. Create a swap-chain with back-buffer for surface and GPU. Back-buffer will contain the
        //    images which are currently not in front-buffer e.g. the image presented to the
        //    screen).
        let size = Extent2D {
            width: window_size.width as u32,
            height: window_size.height as u32,
        };
        let caps = surface.capabilities(&adapter.physical_device);
        let mut config = SwapchainConfig::from_caps(&caps, pixel_format, size);
        config.present_mode = PresentMode::FIFO;
        // Pretty much all the methods on GPU require unsafe scope.
        // You have to manually destroy all resources allocated on GPU.
        let (swapchain, backbuffer) =
            unsafe { gpu.device.create_swapchain(surface, config, None).unwrap() };

        // Specify the layout the image will have before the render pass begins. Here we don't care
        // about the previous image since we're going to clear it anyway.
        let initial_layout = image::Layout::Undefined;
        // Specify the layout to transition when the render pass finishes. The image after this
        // pass will be presented to the swap-chain.
        let final_layout = image::Layout::Present;

        // Create an attachment describing the output of our render pass.
        // Here we will use and output the same pixel format as in the swap-chain.
        // The number of samples per pixel is 1 since we are not doing any multi-sampling yet. We
        // are going to just clear the values in the attachment on every pass and store a new
        // value.
        let attachment = pass::Attachment {
            format: Some(pixel_format),
            samples: 1,
            ops: pass::AttachmentOps::new(
                pass::AttachmentLoadOp::Clear,
                pass::AttachmentStoreOp::Store,
            ),
            // We are not using a stencil buffer, so we aren't adding any operations for it.
            stencil_ops: pass::AttachmentOps::new(
                pass::AttachmentLoadOp::DontCare,
                pass::AttachmentStoreOp::DontCare,
            ),
            layouts: initial_layout..final_layout,
        };

        // Here we reference our color attachment with index 0 defined above.
        let attachment_ref: pass::AttachmentRef = (0, image::Layout::ColorAttachmentOptimal);

        // Sub-passes are the subsequent rendering operations. For example, post-processing effects.
        let subpass = pass::SubpassDesc {
            colors: &[attachment_ref], // Color attachments (we defined one above).
            inputs: &[],               // Attachments to read from shader.
            resolves: &[],             // Attachments for multi-sampling.
            depth_stencil: None,       // Attachments for depth and stencil data.
            preserves: &[],            // Attachment which are not used, but should be preserved.
        };
        let stage_begin = pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT;
        let stage_end = pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT;
        let rw_flags = image::Access::COLOR_ATTACHMENT_READ;
        let dependency = pass::SubpassDependency {
            passes: pass::SubpassRef::External..pass::SubpassRef::Pass(0),
            stages: stage_begin..stage_end,
            accesses: image::Access::empty()..rw_flags,
            flags: memory::Dependencies::empty(),
        };

        // 2. Create a render pass which will specify how many color buffers and depth buffers
        //    will be in the frame buffer. Also how many samples to use for each of them and how
        //    their contents should be handled throughout the rendering operations.
        let render_pass = unsafe {
            gpu.device
                .create_render_pass(&[attachment], &[subpass], &[dependency])
                .unwrap()
        };

        // 3. Create a frame buffers.
        let frame_buffer_extent = image::Extent {
            width: size.width,
            height: size.height,
            depth: 1,
        };

        // Create 2d color image views for every image in the swapchain.
        let frame_buffer_image_format = pixel_format;
        let frame_buffer_image_kind = image::ViewKind::D2;
        let frame_buffer_image_swizzle = format::Swizzle::NO;
        let frame_buffer_image_aspects = format::Aspects::COLOR;
        let frame_buffer_image_resource_range = image::SubresourceRange {
            aspects: frame_buffer_image_aspects,
            levels: 0..1, // Mip-mapping levels with progressively lower resolution.
            layers: 0..1, // Single layer image (multiple layers could be used in VR).
        };
        let frame_buffer_images = backbuffer
            .into_iter()
            .map(|buffer_image| {
                let buffer_image_view = unsafe {
                    gpu.device
                        .create_image_view(
                            &buffer_image,
                            frame_buffer_image_kind,
                            frame_buffer_image_format,
                            frame_buffer_image_swizzle,
                            frame_buffer_image_resource_range.clone(),
                        )
                        .unwrap()
                };
                (buffer_image, buffer_image_view)
            })
            .collect::<Vec<(B::Image, B::ImageView)>>();

        // For each image in the swap chain, create a corresponding frame buffer with the size
        // equal to the window size.
        let frame_buffer_list = frame_buffer_images
            .iter()
            .map(|&(_, ref buffer_image_view)| {
                let frame_buffer_render_pass = &render_pass;
                let frame_buffer_image_view = Some(buffer_image_view);
                let frame_buffer = unsafe {
                    gpu.device
                        .create_framebuffer(
                            frame_buffer_render_pass,
                            frame_buffer_image_view,
                            frame_buffer_extent,
                        )
                        .unwrap()
                };
                frame_buffer
            })
            .collect();

        let frames_count = frame_buffer_images.len();

        // Allocate per frame buffer stuff:
        // Fences are mainly used to synchronize CPU work with rendering operation on GPU, whereas
        // semaphores are used to synchronize operations within or across command queues on GPU.
        let mut frame_buffer_fences = Vec::with_capacity(frames_count);
        let mut command_buffer_pools = Vec::with_capacity(frames_count);
        let mut command_buffer_pools_lists = Vec::with_capacity(frames_count);

        // Semaphores to signal that the image has been acquired and is ready for rendering.
        let mut acquire_semaphores = Vec::with_capacity(frames_count);

        // Semaphores to signal that rendering has finished and presentation to swapchain
        // can happen.
        let mut present_semaphores = Vec::with_capacity(frames_count);

        // Command pool flags.
        let command_pool_flags = pool::CommandPoolCreateFlags::RESET_INDIVIDUAL;

        // Populate frame buffer stuff.
        unsafe {
            for _ in 0..frames_count {
                // Create frame fence.
                let signaled = true;
                let fence = gpu.device.create_fence(signaled);
                frame_buffer_fences.push(fence.unwrap());

                // Create frame command pool for GPU queue.
                let command_pool = gpu
                    .device
                    .create_command_pool(queues.family, command_pool_flags);
                command_buffer_pools.push(command_pool.unwrap());

                // Create frame command buffer.
                let command_buffer = Vec::new();
                command_buffer_pools_lists.push(command_buffer);

                // Create semaphores.
                let acquire_semaphore = gpu.device.create_semaphore();
                let present_semaphore = gpu.device.create_semaphore();
                acquire_semaphores.push(acquire_semaphore.unwrap());
                present_semaphores.push(present_semaphore.unwrap());
            }
        }

        // Initial semaphore index in the swap-chain.
        let sem_index = 0;

        // JIT GLSL -> SPIRV shader compiler.
        // See more here: https://github.com/google/shaderc-rs
        let mut compiler = shaderc::Compiler::new().unwrap();

        // Compile vertex and fragment shaders into SPIR-V byte code.
        // Here we simply inline shader code using macro functions.
        // Both shaders will use their main functions as an entry point.
        let shader_entry = "main";
        let shader_vert_file = "vertex.vert";
        let shader_frag_file = "vertex.frag";
        let shader_vert_artifact = compiler
            .compile_into_spirv(
                include_str!("vertex.vert"),
                shaderc::ShaderKind::Vertex,
                shader_vert_file,
                shader_entry,
                None,
            )
            .unwrap();
        let shader_frag_artifact = compiler
            .compile_into_spirv(
                include_str!("vertex.frag"),
                shaderc::ShaderKind::Fragment,
                shader_frag_file,
                shader_entry,
                None,
            )
            .unwrap();

        // Load shaders to GPU and create shader set to build a graphics pipeline.
        let shader_vert_module = unsafe {
            gpu.device
                .create_shader_module(shader_vert_artifact.as_binary())
                .unwrap()
        };
        let shader_frag_module = unsafe {
            gpu.device
                .create_shader_module(shader_frag_artifact.as_binary())
                .unwrap()
        };

        // Here we just describe a collection of shaders and their entry points
        // for each programmable part of the graphics pipeline.
        let shaders_set = pso::GraphicsShaderSet {
            vertex: pso::EntryPoint {
                entry: shader_entry,
                module: &shader_vert_module,
                specialization: pso::Specialization::default(),
            },
            hull: None,
            domain: None,
            geometry: None,
            fragment: Some(pso::EntryPoint {
                entry: shader_entry,
                module: &shader_frag_module,
                specialization: pso::Specialization::default(),
            }),
        };

        // Describe vertex buffer format. We have only single description
        // for 2D position data.
        let vertex_buffers = vec![pso::VertexBufferDesc {
            binding: 0,
            stride: Triangle::vertex_stride(),
            rate: pso::VertexInputRate::Vertex,
        }];
        let vertex_attributes = Triangle::vertex_attributes();
        // Describe the rasterizer.
        let rasterizer = pso::Rasterizer {
            polygon_mode: pso::PolygonMode::Fill,
            cull_face: pso::Face::NONE,
            front_face: pso::FrontFace::Clockwise,
            depth_clamping: false,
            depth_bias: None,
            conservative: false,
        };

        // Describe depth testing step. This step is before the color blending
        // and after the fragment shader: in addition to having colors, an image
        // also has depth values for each pixel. After a fragment shader runs
        // there's a depth test. We are not using depth testing right now.
        let depth_stencil = pso::DepthStencilDesc {
            depth: None,
            depth_bounds: false,
            stencil: None,
        };

        // Describe color blending – the final step in the pipeline. After a
        // fragment shader has returned a color, it needs to be combined
        // (blended) with the color that is already in the frame-buffer.
        let blender = pso::BlendDesc {
            logic_op: Some(pso::LogicOp::Copy),
            targets: vec![pso::ColorBlendDesc {
                // Apply to all color components.
                mask: pso::ColorMask::ALL,
                // This is essentially describes the following blending
                // operation:
                //  final.rgb = new.rgb*new.alpha + old.rgb * (1-new.alpha)
                //  final.a = 1.0 * new.a + 0.0 * old.a
                // As you see in the code below, we setting old factor to 0
                // to completely ignore the old rgb and old alpha value
                // saved in the frame buffer.
                blend: Some(pso::BlendState {
                    color: pso::BlendOp::Add {
                        src: pso::Factor::SrcAlpha,
                        dst: pso::Factor::OneMinusSrcAlpha,
                    },
                    alpha: pso::BlendOp::Add {
                        src: pso::Factor::One,
                        dst: pso::Factor::Zero,
                    },
                }),
            }],
        };

        // This binds the resource information to the pipeline.
        let bindings: Vec<pso::DescriptorSetLayoutBinding> = vec![];
        let samplers: Vec<B::Sampler> = vec![];
        let layout = unsafe {
            gpu.device
                .create_descriptor_set_layout(bindings, samplers)
                .unwrap()
        };
        let layouts = vec![layout];

        let constants: Vec<(pso::ShaderStageFlags, Range<u32>)> =
            vec![(pso::ShaderStageFlags::FRAGMENT, 0..4)];

        let pipeline_layout = unsafe {
            gpu.device
                .create_pipeline_layout(&layouts, constants)
                .unwrap()
        };

        // Create graphics pipeline.
        let pipeline_primitive = pso::Primitive::TriangleList;
        let pipeline_desc = pso::GraphicsPipelineDesc {
            shaders: shaders_set,
            rasterizer,
            vertex_buffers: vertex_buffers,
            attributes: vertex_attributes,
            input_assembler: pso::InputAssemblerDesc::new(pipeline_primitive),
            blender: blender,
            depth_stencil: depth_stencil,
            multisampling: None,
            baked_states: pso::BakedStates::default(),
            layout: &pipeline_layout,
            subpass: pass::Subpass {
                index: 0,
                main_pass: &render_pass,
            },
            flags: pso::PipelineCreationFlags::empty(),
            parent: pso::BasePipeline::None,
        };

        let pipeline = unsafe {
            gpu.device
                .create_graphics_pipeline(&pipeline_desc, None)
                .unwrap()
        };

        // Create a vertex buffer. Here we are going to allocate buffer for just a
        // single triangle
        let vertex_buffer_num_triangles = 1;
        let vertex_buffer_stride_size = Triangle::vertex_stride() * 3;
        let vertex_buffer_total_size =
            (vertex_buffer_num_triangles * vertex_buffer_stride_size) as u64;
        let vertex_buffer_usage = buffer::Usage::VERTEX;
        let mut vertex_buffer = unsafe {
            gpu.device
                .create_buffer(vertex_buffer_total_size, vertex_buffer_usage)
                .unwrap()
        };
        let vertex_buffer_requirements =
            unsafe { gpu.device.get_buffer_requirements(&vertex_buffer) };
        let vertex_memory_size = vertex_buffer_requirements.size;
        let gpu_memory_types = adapter.physical_device.memory_properties().memory_types;
        let cpu_visible = memory::Properties::CPU_VISIBLE;
        let vertex_buffer_memory_type_id = gpu_memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                let id_u64 = id as u64;
                let supports_mask = vertex_buffer_requirements.type_mask & (id_u64 << 1) != 0;
                let cpu_visible = mem_type.properties.contains(cpu_visible);
                supports_mask && cpu_visible
            })
            .unwrap()
            .into();
        let vertex_memory = unsafe {
            gpu.device
                .allocate_memory(vertex_buffer_memory_type_id, vertex_memory_size)
                .unwrap()
        };

        unsafe {
            gpu.device
                .bind_buffer_memory(&vertex_memory, 0, &mut vertex_buffer)
                .unwrap();
        };

        let compute = ComputeExample::new(&gpu, &adapter, &queues);

        EngineState {
            size,
            swapchain,
            render_pass,
            frame_buffer_fences,
            frame_buffer_images,
            frame_buffer_list,
            command_buffer_pools,
            command_buffer_pools_lists,
            acquire_semaphores,
            present_semaphores,
            pipeline,
            pipeline_layout,
            descriptor_set_layouts: layouts,
            sem_index,
            vertex_buffer,
            vertex_memory,
            vertex_memory_size,
            frame_counter: 0,
            compute,
        }
    }

    /// This function should be called on every frame to get the index of the current
    /// frame in the swapchain and also to advance the index to the next frame.
    fn advance_semaphore_index(&mut self) -> usize {
        if self.sem_index >= self.acquire_semaphores.len() {
            self.sem_index = 0;
        }
        let ret = self.sem_index;
        self.sem_index += 1;
        ret
    }

    /// Draw will draw engine state.
    pub fn draw(
        &mut self,
        gpu: &mut Gpu<B>,
        queue: &mut B::CommandQueue,
        color: [f32; 4],
        triangle: &Triangle,
    ) -> Result<(), String> {
        self.frame_counter += 1;

        // Write triangle data to vertex buffer.
        unsafe {
            let mapping = gpu
                .device
                .map_memory(&self.vertex_memory, 0..self.vertex_memory_size)
                .unwrap();

            let mut points = triangle.points.clone();


            let computed = self.compute.compute(gpu, [points[0], points[1]]);
            points[2] = computed[0];
            points[3] = computed[1];
            points[4] = computed[2];
            points[5] = computed[3];

            let triangle_flat = points.as_ptr() as *const u8;
            let triangle_size = mem::size_of::<f32>() * 6;
            ptr::copy_nonoverlapping(triangle_flat, mapping, triangle_size);
            let memory_range = iter::once((&self.vertex_memory, 0..self.vertex_memory_size));
            gpu.device.flush_mapped_memory_ranges(memory_range).unwrap();
        }

        // Get current semaphore index and corresponding semaphores.
        let sem_index = self.advance_semaphore_index();
        let sem_acquire = &self.acquire_semaphores[sem_index];
        let sem_present = &self.present_semaphores[sem_index];

        // Max time to wait for synchronization.
        let max_wait = core::u64::MAX;

        // Get current frame index.
        let (frame_id, _) = unsafe {
            self.swapchain
                .acquire_image(max_wait, Some(sem_acquire), None)
                .unwrap()
        };
        let frame_id_usize = frame_id as usize;

        // Get current frame buffer stuff.
        let fence = &mut self.frame_buffer_fences[frame_id_usize];
        let frame_buffer = &mut self.frame_buffer_list[frame_id_usize];
        let command_pool = &mut self.command_buffer_pools[frame_id_usize];
        let command_buffers = &mut self.command_buffer_pools_lists[frame_id_usize];

        // Before writing to the command buffer, wait for its fence to be signaled.
        unsafe {
            if !gpu.device.wait_for_fence(fence, max_wait).unwrap() {
                panic!("fence wait timeout");
            }
            gpu.device.reset_fence(fence).unwrap();
        }

        // Create view port.
        let frame_viewport = pso::Viewport {
            rect: pso::Rect {
                x: 0,
                y: 0,
                w: self.size.width as i16,
                h: self.size.height as i16,
            },
            depth: 0.0..1.0,
        };

        // Get command buffer from the current frame pool.
        let mut command_buffer = unsafe {
            match command_buffers.pop() {
                Some(buffer) => buffer,
                None => command_pool.allocate_one(command::Level::Primary),
            }
        };

        let vertex_buffer_iter = iter::once((&self.vertex_buffer, 0));
        unsafe {
            // Start buffer.
            let flags = command::CommandBufferFlags::ONE_TIME_SUBMIT;
            command_buffer.begin_primary(flags);
            command_buffer.set_viewports(0, &[frame_viewport.clone()]);
            command_buffer.set_scissors(0, &[frame_viewport.rect.clone()]);
            command_buffer.bind_graphics_pipeline(&self.pipeline);
            command_buffer.bind_vertex_buffers(0, vertex_buffer_iter);
            command_buffer.begin_render_pass(
                &self.render_pass,
                frame_buffer,
                frame_viewport.rect,
                &[command::ClearValue {
                    color: command::ClearColor { float32: color },
                }],
                command::SubpassContents::Inline,
            );
            command_buffer.push_graphics_constants(
                &self.pipeline_layout,
                pso::ShaderStageFlags::FRAGMENT,
                0,
                &[self.frame_counter],
            );
            command_buffer.draw(0..3, 0..1);
            command_buffer.end_render_pass();
            command_buffer.finish();
        }

        // Create queue submission for the swapchain.
        let submission = queue::Submission {
            command_buffers: iter::once(&command_buffer),
            wait_semaphores: iter::once((&*sem_acquire, pso::PipelineStage::BOTTOM_OF_PIPE)),
            signal_semaphores: iter::once(&*sem_present),
        };

        unsafe {
            // Put the submission to the command queue.
            queue.submit(submission, Some(fence));
        }

        // Return command buffer back.
        command_buffers.push(command_buffer);

        // Present queue to swapchain.
        let result = unsafe { self.swapchain.present(queue, frame_id, Some(&*sem_present)) };

        // Panic if failed to present.
        match result {
            Ok(_) => Ok(()),
            Err(err) => Err(format!("{:?}", err)),
        }
    }
}

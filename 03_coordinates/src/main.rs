extern crate gfx_backend_metal as back;
extern crate gfx_hal as hal;
extern crate nalgebra_glm as glm;

use core::ops::Range;
use std::borrow::Borrow;
use std::io;
use std::iter;
use std::mem;
use std::time;

use gfx::memory::cast_slice;
use hal::{Backend, Instance};
use hal::adapter::{Adapter, Gpu, PhysicalDevice};
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
use hal::pso::DescriptorPool;
use hal::queue;
use hal::queue::CommandQueue;
use hal::queue::family::{QueueFamily, QueueGroup};
use hal::window::{Extent2D, Surface, Swapchain, SwapchainConfig};
use log;
use shaderc;
use simple_logger;
use winit::{Event, EventsLoop, Window, WindowBuilder, WindowEvent};
use winit::dpi::{LogicalPosition, LogicalSize};
use zstd;

use class3::{
    Buffer,
    DepthImage,
    FrameImage,
    LightSource,
    LightInfo,
    Mesh,
    Triangle,
    Vertex,
};

fn main() {

    // Setup a logger.
    simple_logger::init_with_level(log::Level::Info).unwrap();

    let title = String::from("GPU Learning Class: 03 Coordinates");
    let mut events_loop = EventsLoop::new();
    let mut size = LogicalSize {
        width: 960.0,
        height: 540.0,
    };
    let window = WindowBuilder::new()
        .with_dimensions(size)
        .with_resizable(false)
        .build(&events_loop)
        .unwrap();

    // Create GPU API instance and a surface where we will draw pixels.
    let instance = back::Instance::create(title.as_str(), 1).unwrap();
    let surface = unsafe { instance.create_surface(&window).unwrap() };

    // Select adapter (a logical GPU device).
    // Here we just use the first available adapter from the list.
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
    let gpu_queue_family = adapter
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
            .open(&[(gpu_queue_family, &[1.0])], hal::Features::empty())
            .unwrap()
    };

    // Load meshes and construct.

    let mesh_bunny = {
        let zstd_bytes = include_bytes!("../assets/bunny.obj.zst");
        let zstd_reader = zstd::stream::Decoder::new(&zstd_bytes[..]).unwrap();
        let reader = io::BufReader::new(zstd_reader);
        Mesh::read_from_obj(reader).unwrap()
    };

    let mesh_sphere = {
        let zstd_bytes = include_bytes!("../assets/sphere.obj.zst");
        let zstd_reader = zstd::stream::Decoder::new(&zstd_bytes[..]).unwrap();
        let reader = io::BufReader::new(zstd_reader);
        Mesh::read_from_obj(reader).unwrap()
    };

    let mesh_plane = Mesh{
        triangles: vec![
            Triangle{vertices: [
                Vertex{xyz: [ 1.0, 0.0, -1.0], vn: [0.0, 1.0, 0.0]},
                Vertex{xyz: [ 1.0, 0.0,  1.0], vn: [0.0, 1.0, 0.0]},
                Vertex{xyz: [-1.0, 0.0,  1.0], vn: [0.0, 1.0, 0.0]},
            ]},
            Triangle{vertices: [
                Vertex{xyz: [-1.0, 0.0, -1.0], vn: [0.0, 1.0, 0.0]},
                Vertex{xyz: [ 1.0, 0.0, -1.0], vn: [0.0, 1.0, 0.0]},
                Vertex{xyz: [-1.0, 0.0,  1.0], vn: [0.0, 1.0, 0.0]},
            ]},
        ],
        transform: glm::Mat4::identity(),
    };

    let bunny_1 = {
        let s = 3.2;
        let mut m = mesh_bunny.clone();

        let scale = glm::make_vec3(&[s, s, s]);
        let translation = glm::make_vec3(&[0.0, 0.0, 0.45]);
        m.transform = glm::scale(&m.transform, &scale);
        m.transform = glm::translate(&m.transform, &translation);
        m
    };

    let bunny_2 = {
        let s = 1.0;
        let mut m = mesh_bunny.clone();
        let scale = glm::make_vec3(&[s, s, s]);
        let translation = glm::make_vec3(&[-0.3, -0.15, 1.2]);
        m.transform = glm::scale(&m.transform, &scale);
        m.transform = glm::translate(&m.transform, &translation);
        m
    };

    let bunny_3 = {
        let s = 1.0;
        let mut m = mesh_bunny.clone();
        let scale = glm::make_vec3(&[s, s, s]);
        let translation = glm::make_vec3(&[0.3, -0.15, 1.2]);
        m.transform = glm::scale(&m.transform, &scale);
        m.transform = glm::translate(&m.transform, &translation);
        m
    };

    let terrain = {
        let s = 5.0;
        let scale = glm::make_vec3(&[s, s, s]);
        let translation = glm::make_vec3(&[0.0, -0.3, 1.0]);
        let mut m = mesh_plane.clone();
        m.transform = glm::translate(&m.transform, &translation);
        m.transform = glm::scale(&m.transform, &scale);
        m
    };

    let light_source_1 = {
        let s = 0.03;
        let scale = glm::make_vec3(&[s, s, s]);
        let translation = glm::make_vec3(&[0.0, 0.0, 1.0]);
        let mut m = mesh_sphere.clone();
        m.transform = glm::translate(&m.transform, &translation);
        m.transform = glm::scale(&m.transform, &scale);
        m
    };

    let light_source_2 = {
        let mut m = light_source_1.clone();
        *m.transform.index_mut((0, 3)) = 0.32;
        *m.transform.index_mut((1, 3)) = -0.28;
        *m.transform.index_mut((2, 3)) = 1.0;
        m
    };

    let mut meshes = vec![
        terrain,
        light_source_1,
        light_source_2,
        bunny_1,
        bunny_2,
        bunny_3,
    ];

    let mut lights = vec![
        LightSource{
            xyz: [0.0, 0.5, -0.6, 0.0],
            color: [1.0, 0.95, 0.95, 1.0],
        },
        LightSource{
            xyz: [0.32, -0.28, 1.0, 0.0],
            color: [0.55, 0.55, 1.0, 1.0],
        },
    ];

    // Rotation axis for meshes.
    let rot = &glm::make_vec3(&[0.0, 1.0, 0.0]);

    // Camera view and projection matrices.
    let vec_eye = glm::make_vec3(&[0.0, 0.0, 0.0]);
    let vec_target = glm::make_vec3(&[0.0, 0.0, 10.0]);
    let vec_up = glm::make_vec3(&[0.0, 1.0, 0.0]);
    let v = glm::look_at_lh(&vec_eye, &vec_target, &vec_up);

    let p_near = 0.1;
    let p_far  = 1000.0;
    let p = {
        let mut projection = glm::perspective_lh_zo(
            size.width as f32 / size.height as f32,
            f32::to_radians(50.0),
            p_near,
            p_far,
        );
        projection[(1, 1)] *= -1.0;
        projection
    };

    let mat_pv = p.clone() * v.clone();
    let mat_pv_inv = glm::inverse(&mat_pv);

    // Create our engine for drawing state.
    let mut engine = GraphicsEngine::init(
        gpu,
        surface,
        adapter,
        size,
        pixel_format,
        window,
        title);

    // Main loop:
    //
    //  1. Get interaction event for window.
    //  2. Update state.
    //  3. Draw state.
    //
    let mut cursor_pose = LogicalPosition::new(0.0, 0.0);
    let mut is_end_requested = false;
    let mut is_resize_requested = false;

    loop {

        // If end requested, exit the loop.
        if is_end_requested {
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
            } => is_end_requested = true,

            // Window resize.
            Event::WindowEvent {
                event: WindowEvent::Resized(new_size),
                ..
            } => {
                is_resize_requested = true;
                size = new_size;
            }

            // Ignore the rest.
            _ => (),
        });

        if is_resize_requested {
            engine.update_window_size(size);
            is_resize_requested = false;
        }

        for (i, m) in &mut meshes.iter_mut().enumerate() {
            // Skip light and terrain meshes.
            if i == 0 || i == 1 || i == 2 {
                continue;
            }
            let speed =  (1.3 * i as f32 - 1.7) / 40.0;
            m.transform = glm::rotate(&m.transform, speed, &rot);
        }

        // Project cursor coordinates from screen space to world space.

        let screen_x = (cursor_pose.x / size.width)  as f32 * 2.0 - 1.0;
        let screen_y = (cursor_pose.y / size.height) as f32 * 2.0 - 1.0;
        let screen_z = 3.0;

        let vec_xyzw = glm::make_vec4(&[screen_x, screen_y, screen_z, 0.0]);
        let vec_mouse_world_xyzw = mat_pv_inv.clone() * vec_xyzw;
        let vec_mouse_world_xyz = glm::make_vec3(&[
             vec_mouse_world_xyzw[0],
             vec_mouse_world_xyzw[1],
             vec_mouse_world_xyzw[2] + 1.0,
        ]);

        *meshes[1].transform.index_mut((0, 3)) = vec_mouse_world_xyz[0];
        *meshes[1].transform.index_mut((1, 3)) = vec_mouse_world_xyz[1];
        *meshes[1].transform.index_mut((2, 3)) = vec_mouse_world_xyz[2];

        lights[0].xyz = [
            vec_mouse_world_xyz[0],
            vec_mouse_world_xyz[1],
            vec_mouse_world_xyz[2],
            0.0,
        ];

        // Draw meshes.
        engine.draw(&v, &p, &meshes, &lights);
    }
}

/// GraphicsEngine will encapsulate the logic of drawing to surface with one gpu
/// and re-creating the swap-chain when it gets outdated (for example, when the
/// window size gets changed).
///
pub struct GraphicsEngine<B: Backend> {
    surface: B::Surface,
    adapter: Adapter<B>,
    gpu: Gpu<B>,
    queue_group: QueueGroup<B>,
    pixel_format: format::Format,
    clear_color: [f32; 4],
    window_size: LogicalSize,

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
///
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
        let clear_color = [0.05, 0.05, 0.05, 1.0];
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
    pub fn draw(
        &mut self,
        v: &glm::Mat4,
        p: &glm::Mat4,
        models: &Vec<Mesh>,
        lights: &Vec<LightSource>) {

        // Update frame count.
        self.frame_i += 1;
        let begin_time = time::SystemTime::now();

        // Render state.
        let result = self.state.as_mut().unwrap().draw(
            &mut self.gpu,
            &mut self.queue_group.queues[0],
            self.clear_color,
            v,
            p,
            models,
            lights,
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
                self.gpu.device.destroy_render_pass(state.render_pass);
                self.gpu.device.destroy_swapchain(state.swapchain);
                for fence in state.fences.drain(..) {
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
                for frame_buffer in state.frame_buffers.drain(..) {
                    self.gpu.device.destroy_framebuffer(frame_buffer);
                }
                for frame_image in state.frame_images.drain(..) {
                    self.gpu.device.destroy_image_view(frame_image.image_view);
                    self.gpu.device.destroy_image(frame_image.image);
                }
                for depth_image in state.depth_images.drain(..) {
                    self.gpu.device.destroy_image_view(depth_image.image_view);
                    self.gpu.device.destroy_image(depth_image.image);
                    self.gpu.device.free_memory(depth_image.memory);
                }
            }
        }
    }
}

/// Manually destroy all resources allocated on GPU when graphics engine is dropped.
///
impl<B: Backend> Drop for GraphicsEngine<B> {
    fn drop(&mut self) {
        self.destroy_state();
    }
}

/// Engine will be encapsulating the state management and the drawing logic.
///
struct EngineState<B: Backend> {
    // Swap-chain and a back-buffer. The back-buffer is the secondary buffer which is not currently
    // being displayed. The back buffer then becomes the front buffer (and vice versa).
    size: Extent2D,
    swapchain: B::Swapchain,

    // Render pass describing output of graphics pipeline, for example, color attachments and their
    // pixel format.
    render_pass: B::RenderPass,

    // Frame buffers and corresponding stuff for each image in the swap-chain.
    // Each frame in the swap-chain will have a corresponding frame-buffer, a fence for
    // synchronizing CPU and GPU, the actual image and it's view, semaphores, and command pools
    // with currently allocated command buffers.
    frame_buffers: Vec<B::Framebuffer>,
    frame_images: Vec<FrameImage<B>>,
    depth_images: Vec<DepthImage<B>>,

    fences: Vec<B::Fence>,
    acquire_semaphores: Vec<B::Semaphore>,
    present_semaphores: Vec<B::Semaphore>,

    command_buffer_pools: Vec<B::CommandPool>,
    command_buffer_pools_lists: Vec<Vec<B::CommandBuffer>>,

    // Graphic pipeline state objects and vertex buffer.
    pipeline: B::GraphicsPipeline,
    pipeline_layout: B::PipelineLayout,
    pipeline_descriptor_pool: B::DescriptorPool,
    pipeline_descriptor_set: B::DescriptorSet,

    // Pipeline resources.
    vertex_buffer: Buffer<B, Vertex>,
    transform_buffer: Buffer<B, glm::Mat4>,
    lights_buffer: Buffer<B, LightSource>,

    // Index of the currently displayed image in the swap-chain.
    // Needed to acquire buffers, semaphores and fences corresponding to the current image
    // in the back buffer of the swap-chain.
    sem_index: usize,
    frame_counter: u32,

    loaded: bool,
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
        let config = SwapchainConfig::from_caps(&caps, pixel_format, size);

        // Pretty much all the methods on GPU require unsafe scope.
        // You have to manually destroy all resources allocated on GPU.
        let (swapchain, backbuffer) =
            unsafe { gpu.device.create_swapchain(surface, config, None).unwrap() };

        // Specify the layout the image will have before the render pass begins. Here we don't care
        // about the previous image since we're going to clear it anyway.
        let initial_layout = image::Layout::Undefined;
        // Specify the layout to transition when the render pass finishes. The image after this
        // pass will be presented to the swap-chain.
        let final_color_layout = image::Layout::Present;
        let final_depth_layout = image::Layout::DepthStencilAttachmentOptimal;

        // Create an attachments describing the output of the render pass.
        // Here we will use and output the same pixel format as in the swap-chain.
        // The number of samples per pixel is 1 since we are not doing any multi-sampling yet. We
        // are going to just clear the values in the attachment on every pass and store a new
        // value.
        let color_attachment = pass::Attachment {
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
            layouts: initial_layout..final_color_layout,
        };

        // This is similar to color attachment, except that the format isn't going to be surface
        // pixel format.
        let depth_attachment = pass::Attachment {
            format: Some(format::Format::D32Sfloat),
            samples: 1,
            ops: pass::AttachmentOps::new(
                pass::AttachmentLoadOp::Clear,
                pass::AttachmentStoreOp::DontCare,
            ),
            // We are not using a stencil buffer, so we aren't adding any operations for it.
            stencil_ops: pass::AttachmentOps::new(
                pass::AttachmentLoadOp::DontCare,
                pass::AttachmentStoreOp::DontCare,
            ),
            layouts: initial_layout..final_depth_layout,
        };

        // Here we reference our color attachment with index 0 defined above.
        let color_ref: pass::AttachmentRef = (0, image::Layout::ColorAttachmentOptimal);
        let depth_ref: pass::AttachmentRef = (1, image::Layout::DepthStencilAttachmentOptimal);

        // Sub-passes are the subsequent rendering operations. For example, post-processing effects.
        let subpass = pass::SubpassDesc {
            colors: &[color_ref],            // Color attachments (we defined one above).
            inputs: &[],                     // Attachments to read from shader.
            resolves: &[],                   // Attachments for multi-sampling.
            depth_stencil: Some(&depth_ref), // Attachments for depth data.
            preserves: &[], // Attachment which are not used, but should be preserved.
        };

        let access_flags = image::Access::COLOR_ATTACHMENT_READ
            | image::Access::COLOR_ATTACHMENT_WRITE
            | image::Access::DEPTH_STENCIL_ATTACHMENT_READ
            | image::Access::DEPTH_STENCIL_ATTACHMENT_WRITE;

        let stage_color_out = pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT;
        let stage_fragment_tests = pso::PipelineStage::EARLY_FRAGMENT_TESTS;

        let in_begin = stage_color_out;
        let in_end = stage_color_out | stage_fragment_tests;
        let in_dependency = pass::SubpassDependency {
            passes: pass::SubpassRef::External..pass::SubpassRef::Pass(0),
            stages: in_begin..in_end,
            accesses: image::Access::empty()..access_flags,
            flags: memory::Dependencies::empty(),
        };

        let out_begin = stage_color_out | stage_fragment_tests;
        let out_end = stage_color_out;
        let out_dependency = pass::SubpassDependency {
            passes: pass::SubpassRef::Pass(0)..pass::SubpassRef::External,
            stages: out_begin..out_end,
            accesses: access_flags..image::Access::empty(),
            flags: memory::Dependencies::empty(),
        };

        // 2. Create a render pass which will specify how many color buffers and depth buffers
        //    will be in the frame buffer. Also how many samples to use for each of them and how
        //    their contents should be handled throughout the rendering operations.
        let render_pass = unsafe {
            gpu.device
                .create_render_pass(
                    &[color_attachment, depth_attachment],
                    &[subpass],
                    &[in_dependency, out_dependency],
                )
                .unwrap()
        };

        // Create frame buffer images.
        let frame_images: Vec<FrameImage<B>> = backbuffer
            .into_iter()
            .map(|backbuffer_image| FrameImage::new(&gpu, backbuffer_image, pixel_format))
            .collect();

        // Create depth buffer images.
        let depth_images: Vec<DepthImage<B>> = frame_images
            .iter()
            .map(|_| DepthImage::new(&adapter, &gpu, size))
            .collect();

        // For each image in the swap chain, create a corresponding frame buffer
        // with the size equal to the window size.
        let frame_buffer_extent = image::Extent {
            width: size.width,
            height: size.height,
            depth: 1,
        };
        let frame_buffers = frame_images
            .iter()
            .zip(depth_images.iter())
            .map(|(frame_image, depth_image)| {
                let frame_buffer_render_pass = &render_pass;
                let frame_buffer_attachments: [_; 2] = [
                    frame_image.image_view.borrow(),
                    depth_image.image_view.borrow(),
                ];
                let frame_buffer = unsafe {
                    gpu.device
                        .create_framebuffer(
                            frame_buffer_render_pass,
                            frame_buffer_attachments.iter().cloned(),
                            frame_buffer_extent,
                        )
                        .unwrap()
                };
                frame_buffer
            })
            .collect();

        let frames_count = frame_images.len();

        // Allocate per frame buffer stuff:
        // Fences are mainly used to synchronize CPU work with rendering operation on GPU, whereas
        // semaphores are used to synchronize operations within or across command queues on GPU.
        let mut fences = Vec::with_capacity(frames_count);
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
                fences.push(fence.unwrap());

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

        // GLSL -> SPIRV shader compiler.
        // See more here: https://github.com/google/shaderc-rs
        let mut compiler = shaderc::Compiler::new().unwrap();

        // Compile vertex and fragment shaders into SPIR-V byte code.
        // Here we simply inline shader code using macro functions.
        // Both shaders will use their main functions as an entry point.
        let shader_entry = "main";
        let shader_vert_file = "frag_shader.glsl";
        let shader_frag_file = "vert_shader.glsl";
        let shader_vert_artifact = compiler
            .compile_into_spirv(
                include_str!("shaders/frag_shader.glsl"),
                shaderc::ShaderKind::Vertex,
                shader_vert_file,
                shader_entry,
                None,
            )
            .unwrap();
        let shader_frag_artifact = compiler
            .compile_into_spirv(
                include_str!("shaders/vert_shader.glsl"),
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
        let buffer_attributes = Vertex::buffer_attributes();
        let vertex_attributes = Vertex::vertex_attributes();

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
            depth: Some(pso::DepthTest {
                fun: pso::Comparison::LessEqual,
                write: true,
            }),
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
                // This essentially describes the following blending
                // operation:
                //  final.rgb = new.rgb * new.alpha + old.rgb * (1.0 - new.alpha)
                //  final.alpha = 1.0 * new.alpha + 0.0 * old.alpha
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

        // Create a vertex buffer. Here we are going to allocate buffer for just a
        // single triangle.
        let mut vertex_buffer: Buffer<B, Vertex> = Buffer::new_vertex_buffer(
            &adapter, &gpu, 1_000_000);
        let mut transform_buffer: Buffer<B, glm::Mat4> = Buffer::new_uniform_buffer(
            &adapter, &gpu, 10);
        let mut lights_buffer: Buffer<B, LightSource> = Buffer::new_uniform_buffer(
            &adapter, &gpu, 10);

        // Set buffer names.
        vertex_buffer.set_name(&gpu, "vertex_buffer");
        transform_buffer.set_name(&gpu, "transforms_buffer");
        lights_buffer.set_name(&gpu, "lights_buffer");

        let constants_range = 2 * mem::size_of::<glm::Mat4>() as u32;
        let constants: Vec<(pso::ShaderStageFlags, Range<u32>)> = vec![
            (pso::ShaderStageFlags::GRAPHICS, 0..constants_range),
        ];

        let descriptor_bindings = vec![pso::DescriptorSetLayoutBinding {
            binding: 0,
            ty: pso::DescriptorType::UniformBuffer,
            count: 1,
            stage_flags: pso::ShaderStageFlags::GRAPHICS,
            immutable_samplers: false,
        }, pso::DescriptorSetLayoutBinding {
            binding: 1,
            ty: pso::DescriptorType::UniformBufferDynamic,
            count: 1,
            stage_flags: pso::ShaderStageFlags::GRAPHICS,
            immutable_samplers: false,
        }];

        let descriptor_set_layout = unsafe {
            gpu.device.create_descriptor_set_layout(&descriptor_bindings, &[]).unwrap()
        };
        let descriptor_set_layouts = vec![&descriptor_set_layout];
        let pipeline_layout = unsafe {
            gpu.device
                .create_pipeline_layout(descriptor_set_layouts, constants)
                .unwrap()
        };

        let pipeline_descriptor_range = pso::DescriptorRangeDesc {
            ty: pso::DescriptorType::UniformBufferDynamic,
            count: 2,
        };
        let pipeline_descriptor_pool_max_sets = 1;
        let pipeline_descriptor_pool_flsgs = pso::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET;
        let mut pipeline_descriptor_pool = unsafe {
            gpu.device.create_descriptor_pool(
                pipeline_descriptor_pool_max_sets,
                &[pipeline_descriptor_range],
                pipeline_descriptor_pool_flsgs).unwrap()
        };
        let pipeline_descriptor_set = unsafe {
            pipeline_descriptor_pool.allocate_set(&descriptor_set_layout).unwrap()
        };

        unsafe {
            let desc1 = pso::Descriptor::Buffer(&lights_buffer.buffer, None..None);
            let desc2 = pso::Descriptor::Buffer(&transform_buffer.buffer, None..None);
            let write1 = vec![pso::DescriptorSetWrite {
                set: &pipeline_descriptor_set,
                binding: 0,
                array_offset: 0,
                descriptors: Some(desc1),
            }];
            let write2 = vec![pso::DescriptorSetWrite {
                set: &pipeline_descriptor_set,
                binding: 1,
                array_offset: 0,
                descriptors: Some(desc2),
            }];
            gpu.device.write_descriptor_sets(write1);
            gpu.device.write_descriptor_sets(write2);
        }

        // Create graphics pipeline.
        let pipeline_primitive = pso::Primitive::TriangleList;
        let pipeline_desc = pso::GraphicsPipelineDesc {
            shaders: shaders_set,
            rasterizer,
            vertex_buffers: buffer_attributes,
            attributes: vertex_attributes,
            input_assembler: pso::InputAssemblerDesc::new(pipeline_primitive),
            blender,
            depth_stencil,
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

        EngineState {
            size,
            swapchain,
            render_pass,
            fences,
            frame_images,
            frame_buffers,
            depth_images,
            command_buffer_pools,
            command_buffer_pools_lists,
            acquire_semaphores,
            present_semaphores,
            pipeline,
            pipeline_layout,
            pipeline_descriptor_pool,
            pipeline_descriptor_set,
            sem_index,
            vertex_buffer,
            transform_buffer,
            lights_buffer,
            frame_counter: 0,
            loaded: false,
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
        clear_color: [f32; 4],
        v: &glm::Mat4,
        p: &glm::Mat4,
        meshes: &Vec<Mesh>,
        lights: &Vec<LightSource>,
    ) -> Result<(), String> {
        self.frame_counter += 1;

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
        let fence = &mut self.fences[frame_id_usize];
        let frame_buffer = &mut self.frame_buffers[frame_id_usize];
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
        unsafe { command_pool.reset(false) };

        // Write vertices (once);
        if !self.loaded {
            self.vertex_buffer.reset();
            for mesh in meshes {
                let vertex_num = mesh.triangles.len() * 3;
                let vertex_prt = mesh.triangles.as_ptr() as *const u8;
                self.vertex_buffer.copy(gpu, vertex_prt, vertex_num);
            }
            self.loaded = true;
        }

        // Write transforms.
        self.transform_buffer.reset();
        for mesh in meshes {
            let transform_num = 1;
            let transform_ptr = mesh.transform.as_ptr() as *const u8;
            self.transform_buffer.copy(gpu, transform_ptr, transform_num);
        }

        // Write light info.
        self.lights_buffer.reset();
        let num = lights.len() as u32;
        let num_info = LightInfo{
            sources_num: [num, 0, 0, 0],
            junk: [0, 0, 0, 0],
        };
        let num_info_ptr = &num_info as *const _ as *const u8;
        let lights_ptr = lights.as_ptr() as *const u8;
        self.lights_buffer.copy(gpu, num_info_ptr, 1);
        self.lights_buffer.copy(gpu, lights_ptr, lights.len());

        // Allocate command buffer.
        let mut command_buffer = match command_buffers.pop() {
            Some(b) => b,
            None => unsafe {
                command_pool.allocate_one(command::Level::Primary)
            },
        };

        unsafe {

            let vertex_buffer_iter = iter::once((&self.vertex_buffer.buffer, 0));

            command_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);
            command_buffer.set_viewports(0, &[frame_viewport.clone()]);
            command_buffer.set_scissors(0, &[frame_viewport.rect.clone()]);
            command_buffer.bind_graphics_pipeline(&self.pipeline);
            command_buffer.bind_vertex_buffers(0, vertex_buffer_iter);

            command_buffer.begin_render_pass(
                &self.render_pass,
                frame_buffer,
                frame_viewport.rect,
                &[
                    command::ClearValue {
                        color: command::ClearColor { float32: clear_color },
                    },
                    command::ClearValue {
                        depth_stencil: command::ClearDepthStencil {
                            depth: 1.0,
                            stencil: 0,
                        },
                    },
                ],
                command::SubpassContents::Inline,
            );

            command_buffer.push_graphics_constants(
                &self.pipeline_layout,
                pso::ShaderStageFlags::GRAPHICS,
                0,
                cast_slice::<f32, u32>(p.as_slice()),
            );

            command_buffer.push_graphics_constants(
                &self.pipeline_layout,
                pso::ShaderStageFlags::GRAPHICS,
                mem::size_of::<glm::Mat4>() as u32,
                cast_slice::<f32, u32>(v.as_slice()),
            );

            command_buffer.bind_graphics_descriptor_sets(&self.pipeline_layout,
                0,
                vec![&self.pipeline_descriptor_set],
                &[],
            );

            let mut offset: u32 = 0;
            for (i, mesh) in meshes.iter().enumerate() {
                let mesh_vertices_num = mesh.triangles.len() as u32 * 3;
                let mesh_vertices_range = offset..offset+mesh_vertices_num;
                let mesh_instance_range = i as u32 .. (i + 1) as u32;

                command_buffer.draw(
                    mesh_vertices_range,
                    mesh_instance_range,
                );

                offset += mesh_vertices_num;
            }

            command_buffer.end_render_pass();
            command_buffer.finish();

        }

        let mut recorded_buffers = vec![command_buffer];

        // Create queue submission for the swapchain.
        let submission = queue::Submission {
            command_buffers: recorded_buffers.iter(),
            wait_semaphores: iter::once((&*sem_acquire, pso::PipelineStage::BOTTOM_OF_PIPE)),
            signal_semaphores: iter::once(&*sem_present),
        };

        unsafe {
            // Put the submission to the command queue.
            queue.submit(submission, Some(fence));
        }

        // Return command buffer back.
        for buffer in recorded_buffers.drain(..) {
            command_buffers.push(buffer);
        }

        // Present queue to swapchain.
        let result = unsafe { self.swapchain.present(queue, frame_id, Some(&*sem_present)) };

        // Panic if failed to present.
        match result {
            Ok(_) => Ok(()),
            Err(err) => Err(format!("{:?}", err)),
        }
    }
}

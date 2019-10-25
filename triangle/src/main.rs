extern crate gfx_backend_metal as back;
extern crate gfx_hal as hal;

use hal::command;
use hal::format;
use hal::image;
use hal::pass;
use hal::pool;
use hal::pso;
use hal::queue;
use hal::window::Extent2D;
use hal::{
    Backend, CompositeAlpha, Device, Instance, PresentMode, QueueGroup, Surface, Swapchain,
    SwapchainConfig,
};

use winit::dpi::{LogicalPosition, LogicalSize};
use winit::{Event, EventsLoop, WindowBuilder, WindowEvent};

use std::iter;

fn main() {
    // Window size.
    let size = LogicalSize {
        width: 512.0,
        height: 512.0,
    };

    let title = "Triangle Example";
    let mut events_loop = EventsLoop::new();

    let window = WindowBuilder::new()
        .with_title(title)
        .with_dimensions(size)
        .with_resizable(false)
        .build(&events_loop)
        .unwrap();

    // Create API instance and surface where to draw.
    let instance = back::Instance::create(title, 1);
    let mut surface = instance.create_surface(&window);

    // Select adapter (logical GPU).
    let adapter = instance.enumerate_adapters().remove(0);

    // Define default pixel format: standart RGB + Alpha channel, 8 bits each.
    let default_pixel_format = Some(format::Format::Rgba8Srgb);
    let default_channel_format = format::ChannelType::Srgb;

    // Check that device supports surface.
    let (capabilities, formats, modes) = surface.compatibility(&adapter.physical_device);
    println!("capabilities {:?}", capabilities);
    println!("modes: {:?}", modes);

    // Check that surface formats contain SRGB or default to SRGBA.
    let pixel_format_option = formats.map_or(default_pixel_format, |formats| {
        formats
            .iter()
            .find(|surface_pixel_format| {
                let channel_type = surface_pixel_format.base_format().1;
                channel_type == default_channel_format
            })
            .map(|format| *format)
    });
    let pixel_format = pixel_format_option.expect("supported pixel format not found");

    // Open a physical device and queues (just 1) for Graphics commands.
    // Selector function will check if the queues can support output to our surface.
    let num_queues = 1;
    let (gpu, queues) = adapter
        .open_with::<_, hal::Graphics>(num_queues, |family| -> bool {
            surface.supports_queue_family(family)
        })
        .unwrap();

    // Create our engine for drawing state.
    let mut engine = GraphicsEngine::new(gpu, queues, &size, pixel_format, &mut surface);

    // Window event flags.
    let mut end_requested = false;
    let mut cursor_pose: LogicalPosition = LogicalPosition::new(0.0, 0.0);

    // Main loop:
    //  1. get interaction event for window.
    //  2. update stat.
    //  3. draw state.
    loop {
        // If end requested, exit the loop.
        if end_requested {
            break;
        }

        // Get event.
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
            // Default.
            _ => (),
        });

        // Update state.
        let r = cursor_pose.x / size.width;
        let g = cursor_pose.y / size.height;
        let b = 1.0;
        let a = 1.0;
        engine.update([r as f32, g as f32, b, a]);

        // Draw state.
        match engine.draw() {
            Err(msg) => {
                panic!("failed to draw: {}", msg);
            }
            Ok(()) => (),
        }
    }
}

/// Engine will incapsulate state managenemt and drawing logic.
/// Fields that has to be destroyed, should be in some container, e.g. Option or Vec.
struct GraphicsEngine<B: Backend> {
    // Physical GPU and its graphics queues.
    pub gpu: B::Device,
    pub queues: QueueGroup<B, hal::Graphics>,

    // Frame size and its image pixel format.
    pub size: Extent2D,
    pub pixel_format: format::Format,

    // Swapchain with backbuffer. Backbuffer is the secondary buffer which is not currently
    // being displayed. The back buffer then becomes the front buffer (and vice versa).
    pub swapchain: Option<B::Swapchain>,

    // Render pass describing output of graphics pipeline, e.g. attachments and their format.
    pub render_pass: Option<B::RenderPass>,
    // Framebuffers and corresponding stuff for each image in the swapchain.
    pub frame_buffer_list: Vec<B::Framebuffer>,
    pub frame_buffer_fences: Vec<B::Fence>,
    pub frame_buffer_images: Vec<(B::Image, B::ImageView)>,
    pub acquire_semaphores: Vec<B::Semaphore>,
    pub present_semaphores: Vec<B::Semaphore>,
    pub command_buffer_pools: Vec<hal::CommandPool<B, hal::Graphics>>,
    pub command_buffer_pools_lists: Vec<Vec<hal::command::CommandBuffer<B, hal::Graphics>>>,
    // Index of the currently displayed image in the swapchain.
    // Needed to aqquire buffers, semaphores and fences corresponding to the current image
    // in the back buffer of the swapchain.
    pub sem_index: usize,
    pub clear_color: [f32; 4],
}

/// Engine implementation.
impl<B: Backend> GraphicsEngine<B> {
    /// Creates a new graphics engine.
    pub fn new(
        gpu: B::Device,
        queues: QueueGroup<B, hal::Graphics>,
        size: &LogicalSize,
        pixel_format: format::Format,
        surface: &mut B::Surface,
    ) -> Self {
        // 1. Create device swapchain config with window image size, two images (double buffering)
        //    and where the value will represent the color on screen (and not depth, fo example).
        let size = Extent2D {
            width: size.width as u32,
            height: size.height as u32,
        };

        let config = SwapchainConfig {
            present_mode: PresentMode::Fifo,
            composite_alpha: CompositeAlpha::INHERIT,
            format: pixel_format,
            extent: size,
            image_count: 2,
            image_layers: 1,
            image_usage: image::Usage::COLOR_ATTACHMENT,
        };

        // 2. Create a swapchain with backbuffer for surface and gpu. Backbuffer will contain the
        //    images which are currently not in frontbuffer e.g. the image presented to the
        //    screen). Pretty much all the methods on gpu require unsafe scope.
        //    You have to destroy resources allocated on gpu by yourself.
        let (swapchain, backbuffer) =
            unsafe { gpu.create_swapchain(surface, config, None).unwrap() };

        // Specify the layout the image will have before the render pass begins. Here we don't care
        // about the previous image since we're going to clear it anyway.
        let initial_layout = image::Layout::Undefined;
        // Speficy the layout to transition when the render pass finishes. The image after this
        // pass will be presented to the swapchain.
        let final_layout = image::Layout::Present;
        // Create an attachment describing the output of our render pass.
        // Here we will use the same pixel format as in the swapchain.
        // The number of samples per pixel is 1 since we are not doing any multisampling yet.
        // We going to clear values in the attachment on every pass and store a new value.
        // We are not using a stencil buffer, so we aren't adding any operations for it.
        let attachment = pass::Attachment {
            format: Some(pixel_format),
            samples: 1,
            ops: pass::AttachmentOps::new(
                pass::AttachmentLoadOp::Clear,
                pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: pass::AttachmentOps::new(
                pass::AttachmentLoadOp::DontCare,
                pass::AttachmentStoreOp::DontCare,
            ),
            layouts: initial_layout..final_layout,
        };
        // Subpasses are ubsequent rendering operations. For example, post-processing effects.
        // Here we reference color attachment 0.
        let attachment_ref: pass::AttachmentRef = (0, image::Layout::ColorAttachmentOptimal);
        let subpass = pass::SubpassDesc {
            colors: &[attachment_ref],
            inputs: &[],         // Attachments to read from shader.
            resolves: &[],       // Attachments for multisampling.
            depth_stencil: None, // Attachments for depth and stencil data.
            preserves: &[],      // Attachment which are not used, but should be preserved.
        };
        let stage_begin = pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT;
        let stage_end = pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT;
        let rw_flags = image::Access::COLOR_ATTACHMENT_READ;
        let dependency = pass::SubpassDependency {
            passes: pass::SubpassRef::External..pass::SubpassRef::Pass(0),
            stages: stage_begin..stage_end,
            accesses: image::Access::empty()..rw_flags,
        };

        // 3. Create a render pass which will specify how many color and depth buffers will be
        // in the framebuffer be, also how many samples to use for each of them and how their
        // contents should be handled throughout the rendering operations.
        let render_pass = unsafe {
            gpu.create_render_pass(&[attachment], &[subpass], &[dependency])
                .unwrap()
        };

        // 4. Create frame buffer.
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
            levels: 0..1,
            layers: 0..1,
        };
        let frame_buffer_images = backbuffer
            .into_iter()
            .map(|buffer_image| {
                let buffer_image_view = unsafe {
                    gpu.create_image_view(
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

        // For each frame buffer image, create a corresponding frame buffer with size
        // equal to the window surface size.
        let frame_buffer_list = frame_buffer_images
            .iter()
            .map(|&(_, ref buffer_image_view)| {
                let frame_buffer_render_pass = &render_pass;
                let frame_buffer_image_view = Some(buffer_image_view);
                let frame_buffer = unsafe {
                    gpu.create_framebuffer(
                        frame_buffer_render_pass,
                        frame_buffer_image_view,
                        frame_buffer_extent,
                    )
                    .unwrap()
                };
                frame_buffer
            })
            .collect();

        // Allocate per frame buffer stuff.
        let frames_count = frame_buffer_images.len();
        let mut frame_buffer_fences = Vec::with_capacity(frames_count);
        let mut command_buffer_pools = Vec::with_capacity(frames_count);
        let mut command_buffer_pools_lists = Vec::with_capacity(frames_count);
        let mut acquire_semaphores = Vec::with_capacity(frames_count);
        let mut present_semaphores = Vec::with_capacity(frames_count);

        // Command pool flags.
        let command_pool_flags = pool::CommandPoolCreateFlags::RESET_INDIVIDUAL;

        // Populate frame buffer stuff.
        unsafe {
            for _ in 0..frames_count {
                // Create frame fence.
                let signaled = true;
                let fence = gpu.create_fence(signaled);
                frame_buffer_fences.push(fence.unwrap());

                // Create frame command pool for GPU queue.
                let command_pool = gpu.create_command_pool_typed(&queues, command_pool_flags);
                command_buffer_pools.push(command_pool.unwrap());

                // Create frame command buffer.
                let command_buffer = Vec::new();
                command_buffer_pools_lists.push(command_buffer);

                // Create semaphores.
                let acquire_semaphor = gpu.create_semaphore();
                let present_semaphor = gpu.create_semaphore();
                acquire_semaphores.push(acquire_semaphor.unwrap());
                present_semaphores.push(present_semaphor.unwrap());
            }
        }

        let sem_index = 0;
        let clear_color = [0.0, 0.0, 0.0, 1.0];

        // Omg that's all!
        Self {
            gpu,
            queues,
            size,
            pixel_format,
            // All these to be destroyed manually in drop method:
            swapchain: Some(swapchain),
            render_pass: Some(render_pass),
            frame_buffer_fences: frame_buffer_fences,
            frame_buffer_images: frame_buffer_images,
            frame_buffer_list: frame_buffer_list,
            command_buffer_pools: command_buffer_pools,
            command_buffer_pools_lists: command_buffer_pools_lists,
            acquire_semaphores: acquire_semaphores,
            present_semaphores: present_semaphores,
            sem_index: sem_index,
            clear_color: clear_color,
        }
    }

    /// Update will updatet state of graphics engine without drawing anything.
    /// For example, you can update the size of the swap chain images, etc.
    pub fn update(&mut self, color: [f32; 4]) {
        self.clear_color = color;
    }

    /// This function should be called on every frame to get the index of the current
    /// frame in the swapchain and also to advance it to the next frame.
    fn advance_semaphore_index(&mut self) -> usize {
        if self.sem_index >= self.acquire_semaphores.len() {
            self.sem_index = 0;
        }
        let ret = self.sem_index;
        self.sem_index += 1;
        ret
    }

    /// Draw will draw provided color.
    pub fn draw(&mut self) -> Result<(), String> {
        // Prep.
        let sem_index = self.advance_semaphore_index();
        let sem_acquire = &self.acquire_semaphores[sem_index];
        let sem_present = &self.present_semaphores[sem_index];

        // Max time to wait for synchronization.
        let max_wait = core::u64::MAX;

        // Get current frame index.
        let (frame_id, _) = unsafe {
            self.swapchain
                .as_mut()
                .unwrap()
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
            if !self.gpu.wait_for_fence(fence, max_wait).unwrap() {
                panic!("fence wait timeout");
            }
            self.gpu.reset_fence(fence).unwrap();
        }

        // Create view port.
        let frame_viewport = pso::Rect {
            x: 0,
            y: 0,
            w: self.size.width as i16,
            h: self.size.height as i16,
        };

        // Create clear color comand using float RGB format.
        let clear_color_cmd = command::ClearColor::Sfloat(self.clear_color);

        // Get command buffer.
        let mut command_buffer = match command_buffers.pop() {
            Some(buffer) => buffer,
            None => command_pool.acquire_command_buffer(),
        };

        // Write commands.
        unsafe {
            // Start buffer.
            command_buffer.begin();
            command_buffer.begin_render_pass_inline(
                self.render_pass.as_mut().unwrap(),
                frame_buffer,
                frame_viewport,
                &[command::ClearValue::Color(clear_color_cmd)],
            );
            command_buffer.finish();
        }

        // Create queue submussion for the swapchain.
        let submission = queue::Submission {
            command_buffers: iter::once(&command_buffer),
            wait_semaphores: iter::once((
                &*sem_acquire, //
                pso::PipelineStage::BOTTOM_OF_PIPE,
            )),
            signal_semaphores: iter::once(&*sem_present),
        };

        // Get gpu queue for graphics commands (we have at least one queue);
        let command_queue = &mut self.queues.queues[0];
        unsafe {
            // Put submission to the command queue.
            command_queue.submit(submission, Some(fence));
        }

        // Return command buffer back.
        command_buffers.push(command_buffer);

        // Present queue to swapchain.
        let result = unsafe {
            self.swapchain
                .as_mut()
                .unwrap()
                .present(command_queue, frame_id, Some(&*sem_present))
        };

        // Panic if failed to present.
        match result {
            Ok(_) => Ok(()),
            Err(err) => {
                panic!("failed to present queue: {:?}", err);
            }
        }
    }
}

// Destroys resources allocated on gpu.
impl<B: Backend> Drop for GraphicsEngine<B> {
    fn drop(&mut self) {
        unsafe {
            if let Some(render_pass) = self.render_pass.take() {
                self.gpu.destroy_render_pass(render_pass);
            }
            if let Some(swapchain) = self.swapchain.take() {
                self.gpu.destroy_swapchain(swapchain);
            }
            for fence in self.frame_buffer_fences.drain(..) {
                self.gpu.wait_for_fence(&fence, !0).unwrap();
                self.gpu.destroy_fence(fence);
            }
            let pools = self.command_buffer_pools.drain(..);
            let pools_list = self.command_buffer_pools_lists.drain(..);
            for (mut pool, pool_list) in pools.zip(pools_list) {
                pool.free(pool_list);
                self.gpu.destroy_command_pool(pool.into_raw());
            }
            for semaphore in self.acquire_semaphores.drain(..) {
                self.gpu.destroy_semaphore(semaphore);
            }
            for semaphore in self.present_semaphores.drain(..) {
                self.gpu.destroy_semaphore(semaphore);
            }
            for frame_buffer in self.frame_buffer_list.drain(..) {
                self.gpu.destroy_framebuffer(frame_buffer);
            }
            for (_, image_view) in self.frame_buffer_images.drain(..) {
                self.gpu.destroy_image_view(image_view);
            }
        }
    }
}

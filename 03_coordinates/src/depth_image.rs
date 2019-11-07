extern crate gfx_backend_metal as back;
extern crate gfx_hal as hal;

use hal::adapter::{Adapter, Gpu, PhysicalDevice};
use hal::device::Device;
use hal::format;
use hal::image;
use hal::memory::Properties;
use hal::window::Extent2D;
use hal::{Backend, MemoryTypeId};

///
/// TODO
///
pub struct DepthImage<B: Backend> {
    pub image: B::Image,
    pub image_view: B::ImageView,
    pub memory: B::Memory,
}

impl<B: Backend> DepthImage<B> {
    ///
    /// TODO
    ///
    pub fn new(adapter: &Adapter<B>, gpu: &Gpu<B>, extent: Extent2D) -> Self {
        // Image properties.

        let image_kind = image::Kind::D2(extent.width, extent.height, 1, 1);
        let image_mip_levels = 1;
        let image_format = format::Format::D32Sfloat;
        let image_tiling = image::Tiling::Optimal;
        let image_usage = image::Usage::DEPTH_STENCIL_ATTACHMENT;
        let image_caps = image::ViewCapabilities::empty();

        let (image, image_view, memory) = unsafe {
            // Create image.
            let mut image = gpu
                .device
                .create_image(
                    image_kind,
                    image_mip_levels,
                    image_format,
                    image_tiling,
                    image_usage,
                    image_caps,
                )
                .unwrap();

            // Allocate memory for image.
            let requirements = gpu.device.get_image_requirements(&image);
            let memory_type_id = adapter
                .physical_device
                .memory_properties()
                .memory_types
                .iter()
                .enumerate()
                .find(|&(memory_type_id, memory_type)| {
                    let supports_mask = requirements.type_mask & (1 << memory_type_id) != 0;
                    let device_local = memory_type.properties.contains(Properties::DEVICE_LOCAL);
                    supports_mask && device_local
                })
                .map(|(memory_id, _)| MemoryTypeId(memory_id))
                .unwrap();

            let memory = gpu
                .device
                .allocate_memory(memory_type_id, requirements.size)
                .unwrap();

            // Bind memory to image.
            gpu.device
                .bind_image_memory(&memory, 0, &mut image)
                .unwrap();

            // Create image view.
            let image_view = gpu
                .device
                .create_image_view(
                    &image,
                    image::ViewKind::D2,
                    format::Format::D32Sfloat,
                    format::Swizzle::NO,
                    image::SubresourceRange {
                        aspects: format::Aspects::DEPTH,
                        levels: 0..1,
                        layers: 0..1,
                    },
                )
                .unwrap();

            (image, image_view, memory)
        };
        Self {
            image,
            image_view,
            memory,
        }
    }
}

extern crate gfx_backend_metal as back;
extern crate gfx_hal as hal;

use hal::adapter::Gpu;
use hal::Backend;
use hal::device::Device;
use hal::format;
use hal::image;

/// FrameImage bundles color image with its view.
///
pub struct FrameImage<B: Backend> {
    pub image: B::Image,
    pub image_view: B::ImageView,
}

/// Implementation.
///
impl<B: Backend> FrameImage<B> {

    /// Create frame buffer image (must be taken from back-buffer) with its view.
    ///
    pub fn new(gpu: &Gpu<B>, image: B::Image, pixel_format: format::Format) -> Self {
        let image_kind = image::ViewKind::D2;
        let image_format = pixel_format;
        let image_swizzle = format::Swizzle::NO;
        let image_resource_ranges = image::SubresourceRange {
            aspects: format::Aspects::COLOR,
            levels: 0..1, // Mip-mapping levels with progressively lower resolution.
            layers: 0..1, // Single layer image (multiple layers could be used in VR).
        };
        let image_view = unsafe {
            gpu.device
                .create_image_view(
                    &image,
                    image_kind,
                    image_format,
                    image_swizzle,
                    image_resource_ranges.clone(),
                )
                .unwrap()
        };
        Self { image, image_view }
    }
}

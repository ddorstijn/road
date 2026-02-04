use vulkanalia::vk;
use vulkanalia_vma::{self as vma};

#[derive(Debug)]
pub struct AllocatedImage {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub allocation: vma::Allocation,
    pub image_extent: vk::Extent3D,
    pub image_format: vk::Format,
}

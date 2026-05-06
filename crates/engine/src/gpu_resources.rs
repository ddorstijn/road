use vulkanalia::prelude::v1_4::*;
use vulkanalia_bootstrap::Device;
use vulkanalia_vma::{self as vma, Alloc, AllocationCreateFlags};

// ---------------------------------------------------------------------------
// GpuImage — a VMA-allocated image with its view
// ---------------------------------------------------------------------------

pub struct GpuImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub allocation: vma::Allocation,
    pub extent: vk::Extent3D,
    pub format: vk::Format,
}

impl GpuImage {
    /// Create a 2D storage image suitable for compute writes + transfer src (blit to swapchain).
    pub fn new_storage_2d(
        device: &Device,
        allocator: &vma::Allocator,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> anyhow::Result<Self> {
        let extent = vk::Extent3D::builder()
            .width(width)
            .height(height)
            .depth(1)
            .build();

        let usage = vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::COLOR_ATTACHMENT;

        let create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::_2D)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage);

        let alloc_opts = vma::AllocationOptions::default();
        let (image, allocation) = unsafe { allocator.create_image(create_info, &alloc_opts) }?;

        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::builder()
                    .view_type(vk::ImageViewType::_2D)
                    .image(image)
                    .format(format)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    ),
                None,
            )?
        };

        Ok(Self {
            image,
            view,
            allocation,
            extent,
            format,
        })
    }

    pub fn extent_2d(&self) -> vk::Extent2D {
        vk::Extent2D::builder()
            .width(self.extent.width)
            .height(self.extent.height)
            .build()
    }

    /// Destroy the image and free its memory. Must be called before the allocator is dropped.
    pub fn destroy(&self, device: &Device, allocator: &vma::Allocator) {
        unsafe {
            device.destroy_image_view(self.view, None);
            allocator.destroy_image(self.image, self.allocation);
        }
    }
}

// ---------------------------------------------------------------------------
// GpuBuffer — a VMA-allocated buffer
// ---------------------------------------------------------------------------

pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub allocation: vma::Allocation,
    pub size: vk::DeviceSize,
}

impl GpuBuffer {
    /// Create a GPU-only buffer with the given usage flags.
    pub fn new(
        allocator: &vma::Allocator,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> anyhow::Result<Self> {
        let create_info = vk::BufferCreateInfo::builder().size(size).usage(usage);

        let alloc_opts = vma::AllocationOptions::default();
        let (buffer, allocation) = unsafe { allocator.create_buffer(create_info, &alloc_opts) }?;

        Ok(Self {
            buffer,
            allocation,
            size,
        })
    }

    /// Create a host-visible, persistently mapped buffer.
    /// Returns the buffer and a raw pointer to mapped memory.
    pub fn new_mapped(
        allocator: &vma::Allocator,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> anyhow::Result<(Self, *mut u8)> {
        let create_info = vk::BufferCreateInfo::builder().size(size).usage(usage);

        let alloc_opts = vma::AllocationOptions {
            flags: AllocationCreateFlags::MAPPED
                | AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            usage: vma::MemoryUsage::AutoPreferHost,
            ..Default::default()
        };

        let (buffer, allocation) = unsafe { allocator.create_buffer(create_info, &alloc_opts) }?;

        let info = allocator.get_allocation_info(allocation);
        let mapped_ptr = info.pMappedData as *mut u8;
        assert!(!mapped_ptr.is_null(), "VMA mapped allocation returned null");

        Ok((
            Self {
                buffer,
                allocation,
                size,
            },
            mapped_ptr,
        ))
    }

    pub fn destroy(&self, allocator: &vma::Allocator) {
        unsafe {
            allocator.destroy_buffer(self.buffer, self.allocation);
        }
    }
}

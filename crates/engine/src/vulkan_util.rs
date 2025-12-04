use std::sync::Arc;
use vulkano::{
    Version, VulkanLibrary,
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags, physical::PhysicalDeviceType,
    },
    image::{ImageUsage, view::ImageView},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
};
use winit::{dpi::PhysicalSize, event_loop::ActiveEventLoop};

pub fn init_vulkan(event_loop: &ActiveEventLoop) -> (Arc<Instance>, Arc<Device>, Arc<Queue>) {
    let library = VulkanLibrary::new().unwrap();
    let required_extensions = Surface::required_extensions(event_loop).unwrap();

    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .unwrap();

    let mut device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| {
            p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering
        })
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && p.presentation_support(i as u32, event_loop).unwrap()
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            _ => 2,
        })
        .expect("no suitable physical device found");

    if physical_device.api_version() < Version::V1_3 {
        device_extensions.khr_dynamic_rendering = true;
    }

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: device_extensions,
            enabled_features: DeviceFeatures {
                dynamic_rendering: true,
                ..DeviceFeatures::empty()
            },
            ..Default::default()
        },
    )
    .unwrap();

    (instance, device, queues.next().unwrap())
}

pub fn create_swapchain(
    window_size: PhysicalSize<u32>,
    surface: Arc<Surface>,
    device: Arc<Device>,
    old_swapchain: Option<Arc<Swapchain>>,
) -> (Arc<Swapchain>, Vec<Arc<ImageView>>) {
    let (swapchain, images) = match old_swapchain {
        Some(swapchain) => swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: window_size.into(),
                ..swapchain.create_info()
            })
            .expect("failed to recreate swapchain"),
        None => {
            let surface_caps = device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();
            let (image_format, _) = device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0];

            Swapchain::new(
                device.clone(),
                surface,
                SwapchainCreateInfo {
                    min_image_count: surface_caps.min_image_count.max(2),
                    image_format,
                    image_extent: window_size.into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
                    composite_alpha: surface_caps
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    ..Default::default()
                },
            )
            .unwrap()
        }
    };

    let image_views = images
        .iter()
        .map(|image| ImageView::new_default(image.clone()).unwrap())
        .collect();

    (swapchain, image_views)
}

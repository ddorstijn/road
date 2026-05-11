use vulkanalia::prelude::v1_4::*;
use vulkanalia_bootstrap::Device;

/// Pipeline statistics counters returned per query.
///
/// Each field maps to one bit in `QueryPipelineStatisticFlags`.
/// The order matches the Vulkan spec's bit order — the driver returns
/// results for enabled bits in ascending-bit order.
#[derive(Debug, Default, Clone, Copy)]
pub struct PipelineStats {
    pub compute_shader_invocations: u64,
}

/// GPU pipeline statistics profiler.
///
/// Wraps a `VK_QUERY_TYPE_PIPELINE_STATISTICS` query pool that measures
/// per-frame counters (currently compute shader invocations).
/// Results are smoothed with an exponential moving average to match
/// `GpuTimestamps`.
pub struct GpuPipelineStats {
    query_pool: vk::QueryPool,
    /// Smoothed stats.
    pub stats: PipelineStats,
    active: bool,
    prev_active: bool,
}

impl GpuPipelineStats {
    pub fn new(device: &Device) -> anyhow::Result<Self> {
        let pool_info = vk::QueryPoolCreateInfo::builder()
            .query_type(vk::QueryType::PIPELINE_STATISTICS)
            .query_count(1)
            .pipeline_statistics(vk::QueryPipelineStatisticFlags::COMPUTE_SHADER_INVOCATIONS);
        let query_pool = unsafe { device.create_query_pool(&pool_info, None)? };
        Ok(Self {
            query_pool,
            stats: PipelineStats::default(),
            active: false,
            prev_active: false,
        })
    }

    /// Read results from the previous frame's query.
    pub fn read_results(&mut self, device: &Device) {
        if !self.prev_active {
            return;
        }
        // We enabled 1 statistic flag → 1 u64 result.
        let mut result_data = [0u64; 1];
        let res = unsafe {
            let data = std::slice::from_raw_parts_mut(
                result_data.as_mut_ptr() as *mut u8,
                std::mem::size_of_val(&result_data),
            );
            device.get_query_pool_results(
                self.query_pool,
                0,
                1,
                data,
                std::mem::size_of::<u64>() as u64,
                vk::QueryResultFlags::_64 | vk::QueryResultFlags::WAIT,
            )
        };
        if res.is_ok() {
            let alpha = 0.1;
            let prev = self.stats.compute_shader_invocations as f64;
            let raw = result_data[0] as f64;
            self.stats.compute_shader_invocations = (prev * (1.0 - alpha) + raw * alpha) as u64;
        }
    }

    /// Reset the query and begin recording pipeline statistics.
    /// Call this at the start of the command buffer, before any dispatches.
    pub fn begin(&mut self, device: &Device, cmd: vk::CommandBuffer) {
        self.prev_active = self.active;
        unsafe {
            device.cmd_reset_query_pool(cmd, self.query_pool, 0, 1);
            device.cmd_begin_query(cmd, self.query_pool, 0, vk::QueryControlFlags::empty());
        }
        self.active = true;
    }

    /// End recording pipeline statistics.
    /// Call this at the end of the command buffer, after all dispatches.
    pub fn end(&self, device: &Device, cmd: vk::CommandBuffer) {
        unsafe {
            device.cmd_end_query(cmd, self.query_pool, 0);
        }
    }

    /// Format stats for display.
    pub fn format(&self) -> String {
        let invocations = self.stats.compute_shader_invocations;
        if invocations >= 1_000_000 {
            format!("{}M invocations", invocations / 1_000_000)
        } else if invocations >= 1_000 {
            format!("{}K invocations", invocations / 1_000)
        } else {
            format!("{} invocations", invocations)
        }
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_query_pool(self.query_pool, None);
        }
    }
}

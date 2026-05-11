use vulkanalia::prelude::v1_4::*;
use vulkanalia_bootstrap::Device;

/// GPU timestamp profiler using Vulkan timestamp queries.
///
/// Tracks N phase markers per frame (begin + N phase-end timestamps) and
/// computes smoothed durations via exponential moving average.
pub struct GpuTimestamps {
    pub query_pool: vk::QueryPool,
    timestamp_period_ns: f32,
    num_phases: u32,
    /// Phase names for display.
    pub phase_names: Vec<&'static str>,
    /// Smoothed phase durations (ms), one per phase.
    pub phase_ms: Vec<f32>,
    pub total_ms: f32,
    /// Next query slot to write (1-based; slot 0 is frame-begin).
    next_query: u32,
    /// How many queries were written in the previous frame (for read_results).
    prev_query_count: u32,
}

impl GpuTimestamps {
    /// Create a new GPU timestamp profiler.
    ///
    /// `timestamp_period` is the device's timestamp period in nanoseconds per tick
    /// (from `VkPhysicalDeviceLimits::timestampPeriod`).
    /// `phase_names` defines the phases to track; the query pool will have
    /// `phase_names.len() + 1` slots (one for frame-begin plus one per phase).
    pub fn new(
        device: &Device,
        timestamp_period: f32,
        phase_names: &[&'static str],
    ) -> anyhow::Result<Self> {
        let num_phases = phase_names.len() as u32;
        let query_count = num_phases + 1; // slot 0 = frame begin
        let pool_info = vk::QueryPoolCreateInfo::builder()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(query_count);
        let query_pool = unsafe { device.create_query_pool(&pool_info, None)? };
        Ok(Self {
            query_pool,
            timestamp_period_ns: timestamp_period,
            num_phases,
            phase_names: phase_names.to_vec(),
            phase_ms: vec![0.0; num_phases as usize],
            total_ms: 0.0,
            next_query: 1,
            prev_query_count: 0,
        })
    }

    /// Read results from the previous frame's timestamp queries.
    pub fn read_results(&mut self, device: &Device) {
        let count = self.prev_query_count;
        if count < 2 {
            // Need at least frame-begin + one phase to compute anything
            return;
        }
        let mut timestamps = vec![0u64; count as usize];
        let result = unsafe {
            let data = std::slice::from_raw_parts_mut(
                timestamps.as_mut_ptr() as *mut u8,
                std::mem::size_of::<u64>() * count as usize,
            );
            device.get_query_pool_results(
                self.query_pool,
                0,
                count,
                data,
                std::mem::size_of::<u64>() as u64,
                vk::QueryResultFlags::_64 | vk::QueryResultFlags::WAIT,
            )
        };
        if result.is_ok() {
            let period = self.timestamp_period_ns as f64;
            let to_ms = |a: u64, b: u64| -> f32 {
                if b >= a {
                    ((b - a) as f64 * period / 1_000_000.0) as f32
                } else {
                    0.0
                }
            };

            let phases_written = count - 1; // subtract frame-begin slot
            let alpha = 0.1f32;
            for i in 0..phases_written as usize {
                let raw = to_ms(timestamps[i], timestamps[i + 1]);
                self.phase_ms[i] = self.phase_ms[i] * (1.0 - alpha) + raw * alpha;
            }
            let last = (count - 1) as usize;
            let total = to_ms(timestamps[0], timestamps[last]);
            self.total_ms = self.total_ms * (1.0 - alpha) + total * alpha;
        }
    }

    /// Reset queries and write the frame-begin timestamp.
    pub fn reset_and_begin(&mut self, device: &Device, cmd: vk::CommandBuffer) {
        // Save how many queries were written this frame for next frame's read
        self.prev_query_count = self.next_query;
        let query_count = self.num_phases + 1;
        unsafe {
            device.cmd_reset_query_pool(cmd, self.query_pool, 0, query_count);
            device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                self.query_pool,
                0,
            );
        }
        self.next_query = 1;
    }

    /// Write the next phase-end timestamp. Phases must be written in order.
    pub fn write_phase(&mut self, device: &Device, cmd: vk::CommandBuffer) {
        let idx = self.next_query;
        assert!(idx <= self.num_phases, "too many write_phase calls");
        unsafe {
            device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                self.query_pool,
                idx,
            );
        }
        self.next_query += 1;
    }

    /// Format phase timings for display (e.g. in a window title).
    pub fn format_phases(&self) -> String {
        let parts: Vec<String> = self
            .phase_names
            .iter()
            .zip(self.phase_ms.iter())
            .map(|(name, ms)| format!("{name}:{ms:.2}"))
            .collect();
        format!("{:.2}ms ({})", self.total_ms, parts.join(" "))
    }

    /// Destroy the query pool.
    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_query_pool(self.query_pool, None);
        }
    }
}

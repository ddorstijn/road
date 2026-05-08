use vulkanalia::prelude::v1_4::*;
use vulkanalia_bootstrap::Device;

/// Timestamp query indices
const TS_FRAME_BEGIN: u32 = 0;
const TS_COUNT: u32 = 5;

/// GPU timestamp profiler using Vulkan timestamp queries.
///
/// Tracks up to 5 timestamps per frame (begin + 4 phase markers) and
/// computes smoothed durations via exponential moving average.
pub struct GpuTimestamps {
    pub query_pool: vk::QueryPool,
    timestamp_period_ns: f32,
    /// Smoothed phase durations (ms)
    pub phase_ms: [f32; 4],
    pub total_ms: f32,
    has_results: bool,
}

impl GpuTimestamps {
    /// Create a new GPU timestamp profiler.
    ///
    /// `timestamp_period` is the device's timestamp period in nanoseconds per tick
    /// (from `VkPhysicalDeviceLimits::timestampPeriod`).
    pub fn new(device: &Device, timestamp_period: f32) -> anyhow::Result<Self> {
        let pool_info = vk::QueryPoolCreateInfo::builder()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(TS_COUNT);
        let query_pool = unsafe { device.create_query_pool(&pool_info, None)? };
        Ok(Self {
            query_pool,
            timestamp_period_ns: timestamp_period,
            phase_ms: [0.0; 4],
            total_ms: 0.0,
            has_results: false,
        })
    }

    /// Read results from the previous frame's timestamp queries.
    pub fn read_results(&mut self, device: &Device) {
        if !self.has_results {
            return;
        }
        let mut timestamps = [0u64; TS_COUNT as usize];
        let result = unsafe {
            let data = std::slice::from_raw_parts_mut(
                timestamps.as_mut_ptr() as *mut u8,
                std::mem::size_of_val(&timestamps),
            );
            device.get_query_pool_results(
                self.query_pool,
                0,
                TS_COUNT,
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

            let raw = [
                to_ms(timestamps[0], timestamps[1]),
                to_ms(timestamps[1], timestamps[2]),
                to_ms(timestamps[2], timestamps[3]),
                to_ms(timestamps[3], timestamps[4]),
            ];
            let total = to_ms(timestamps[0], timestamps[4]);

            // Exponential moving average (α = 0.1)
            let a = 0.1f32;
            for (i, &val) in raw.iter().enumerate() {
                self.phase_ms[i] = self.phase_ms[i] * (1.0 - a) + val * a;
            }
            self.total_ms = self.total_ms * (1.0 - a) + total * a;
        }
    }

    /// Reset queries and write the frame-begin timestamp.
    pub fn reset_and_begin(&mut self, device: &Device, cmd: vk::CommandBuffer) {
        unsafe {
            device.cmd_reset_query_pool(cmd, self.query_pool, 0, TS_COUNT);
            device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                self.query_pool,
                TS_FRAME_BEGIN,
            );
        }
        self.has_results = true;
    }

    /// Write a phase-end timestamp (index 1..4).
    pub fn write_phase(&self, device: &Device, cmd: vk::CommandBuffer, phase_index: u32) {
        assert!((1..TS_COUNT).contains(&phase_index));
        unsafe {
            device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                self.query_pool,
                phase_index,
            );
        }
    }

    /// Destroy the query pool.
    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_query_pool(self.query_pool, None);
        }
    }
}

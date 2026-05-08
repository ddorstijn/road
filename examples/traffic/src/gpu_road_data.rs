use engine::gpu_resources::GpuBuffer;
use engine::vk;
use road::network::RoadNetwork;

// ---------------------------------------------------------------------------
// GPU data structs (match shader layout)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSegment {
    pub segment_type: u32,
    pub s_start: f32,
    pub origin: [f32; 2],
    pub heading: f32,
    pub length: f32,
    pub k_start: f32,
    pub k_end: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRoad {
    pub segment_offset: u32,
    pub segment_count: u32,
    pub lane_section_offset: u32,
    pub lane_section_count: u32,
    pub total_length: f32,
    pub _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLaneSection {
    pub s_start: f32,
    pub s_end: f32,
    pub lane_offset: u32,
    pub lane_count: u32,
    pub left_lane_count: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLane {
    pub width: f32,
    pub lane_type: u32,
    pub marking_type: u32,
    pub _pad: u32,
}

// ---------------------------------------------------------------------------
// Segment type conversion helper
// ---------------------------------------------------------------------------

/// Extract (segment_type, k_start, k_end) from a road segment.
pub fn segment_type_info(seg: &road::primitives::Segment) -> (u32, f32, f32) {
    match seg {
        road::primitives::Segment::Line { .. } => (0, 0.0, 0.0),
        road::primitives::Segment::Arc { curvature, .. } => (1, *curvature, *curvature),
        road::primitives::Segment::Spiral { k_start, k_end, .. } => (2, *k_start, *k_end),
    }
}

// ---------------------------------------------------------------------------
// GPU Road Data manager
// ---------------------------------------------------------------------------

pub struct GpuRoadData {
    pub segment_buffer: Option<GpuBuffer>,
    pub road_buffer: Option<GpuBuffer>,
    pub lane_section_buffer: Option<GpuBuffer>,
    pub lane_buffer: Option<GpuBuffer>,
}

impl GpuRoadData {
    pub fn new() -> Self {
        Self {
            segment_buffer: None,
            road_buffer: None,
            lane_section_buffer: None,
            lane_buffer: None,
        }
    }

    pub fn destroy(&mut self, allocator: &engine::vma::Allocator) {
        if let Some(b) = self.segment_buffer.take() {
            b.destroy(allocator);
        }
        if let Some(b) = self.road_buffer.take() {
            b.destroy(allocator);
        }
        if let Some(b) = self.lane_section_buffer.take() {
            b.destroy(allocator);
        }
        if let Some(b) = self.lane_buffer.take() {
            b.destroy(allocator);
        }
    }

    pub fn upload(
        &mut self,
        allocator: &engine::vma::Allocator,
        network: &RoadNetwork,
    ) -> anyhow::Result<()> {
        self.destroy(allocator);

        if network.roads.is_empty() {
            return Ok(());
        }

        let mut segments = Vec::new();
        let mut roads = Vec::new();
        let mut lane_sections = Vec::new();
        let mut lanes = Vec::new();

        for road_data in &network.roads {
            let seg_offset = segments.len() as u32;
            let rl = &road_data.reference_line;

            for (i, seg) in rl.segments.iter().enumerate() {
                let (seg_type, k_start, k_end) = segment_type_info(seg);

                segments.push(GpuSegment {
                    segment_type: seg_type,
                    s_start: rl.s_offsets[i],
                    origin: [rl.origins[i].x, rl.origins[i].y],
                    heading: rl.headings[i],
                    length: seg.length(),
                    k_start,
                    k_end,
                });
            }

            let ls_offset = lane_sections.len() as u32;
            for section in &road_data.lane_sections {
                let lane_offset = lanes.len() as u32;
                let left_count = section.left_lanes.len() as u32;
                let all_lanes: Vec<_> = section
                    .left_lanes
                    .iter()
                    .chain(section.right_lanes.iter())
                    .collect();

                for lane in &all_lanes {
                    lanes.push(GpuLane {
                        width: lane.width,
                        lane_type: lane.lane_type as u32,
                        marking_type: lane.marking as u32,
                        _pad: 0,
                    });
                }

                lane_sections.push(GpuLaneSection {
                    s_start: section.s_start,
                    s_end: section.s_end,
                    lane_offset,
                    lane_count: all_lanes.len() as u32,
                    left_lane_count: left_count,
                    _pad: 0,
                });
            }

            roads.push(GpuRoad {
                segment_offset: seg_offset,
                segment_count: rl.segments.len() as u32,
                lane_section_offset: ls_offset,
                lane_section_count: road_data.lane_sections.len() as u32,
                total_length: rl.total_length,
                _pad: [0; 3],
            });
        }

        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;

        if !segments.is_empty() {
            let data = bytemuck::cast_slice::<_, u8>(&segments);
            let (buf, ptr) = GpuBuffer::new_mapped(allocator, data.len() as u64, usage)?;
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()) };
            self.segment_buffer = Some(buf);
        }

        if !roads.is_empty() {
            let data = bytemuck::cast_slice::<_, u8>(&roads);
            let (buf, ptr) = GpuBuffer::new_mapped(allocator, data.len() as u64, usage)?;
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()) };
            self.road_buffer = Some(buf);
        }

        if !lane_sections.is_empty() {
            let data = bytemuck::cast_slice::<_, u8>(&lane_sections);
            let (buf, ptr) = GpuBuffer::new_mapped(allocator, data.len() as u64, usage)?;
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()) };
            self.lane_section_buffer = Some(buf);
        }

        if !lanes.is_empty() {
            let data = bytemuck::cast_slice::<_, u8>(&lanes);
            let (buf, ptr) = GpuBuffer::new_mapped(allocator, data.len() as u64, usage)?;
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()) };
            self.lane_buffer = Some(buf);
        }

        Ok(())
    }
}

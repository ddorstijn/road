use crate::fitting::{ControlPoint, ReferenceLine};

/// Type of a lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum LaneType {
    Driving = 0,
    Shoulder = 1,
    Median = 2,
}

/// Marking type for the boundary between lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MarkingType {
    None = 0,
    SolidWhite = 1,
    DashedWhite = 2,
    SolidYellow = 3,
    DashedYellow = 4,
    DoubleSolidYellow = 5,
}

/// A single lane within a lane section.
#[derive(Debug, Clone, Copy)]
pub struct Lane {
    pub width: f32,
    pub lane_type: LaneType,
    /// Marking on the outer (away from center) boundary of this lane.
    pub marking: MarkingType,
}

/// A lane section spans a range of s along the reference line.
/// Within this range, the lane configuration is constant.
#[derive(Debug, Clone)]
pub struct LaneSection {
    pub s_start: f32,
    pub s_end: f32,
    /// Lanes to the left of center (index 0 = closest to center).
    pub left_lanes: Vec<Lane>,
    /// Lanes to the right of center (index 0 = closest to center).
    pub right_lanes: Vec<Lane>,
}

/// A road: control points, fitted reference line, and lane configuration.
#[derive(Debug, Clone)]
pub struct Road {
    pub control_points: Vec<ControlPoint>,
    pub reference_line: ReferenceLine,
    pub lane_sections: Vec<LaneSection>,
}

/// The full road network.
#[derive(Debug, Clone, Default)]
pub struct RoadNetwork {
    pub roads: Vec<Road>,
}

impl Road {
    /// Create a road from control points with a default lane setup:
    /// 2 lanes each direction, 3.5 m width, dashed center marking, solid edge.
    pub fn new_with_defaults(control_points: Vec<ControlPoint>) -> Option<Self> {
        let reference_line = ReferenceLine::fit(&control_points)?;
        let total_length = reference_line.total_length;

        let lane_sections = vec![LaneSection {
            s_start: 0.0,
            s_end: total_length,
            left_lanes: vec![
                Lane {
                    width: 3.5,
                    lane_type: LaneType::Driving,
                    marking: MarkingType::DashedWhite,
                },
                Lane {
                    width: 3.5,
                    lane_type: LaneType::Driving,
                    marking: MarkingType::SolidWhite,
                },
            ],
            right_lanes: vec![
                Lane {
                    width: 3.5,
                    lane_type: LaneType::Driving,
                    marking: MarkingType::DashedWhite,
                },
                Lane {
                    width: 3.5,
                    lane_type: LaneType::Driving,
                    marking: MarkingType::SolidWhite,
                },
            ],
        }];

        Some(Road {
            control_points,
            reference_line,
            lane_sections,
        })
    }

    /// Get the lane section at a given s-coordinate.
    pub fn lane_section_at(&self, s: f32) -> Option<&LaneSection> {
        self.lane_sections
            .iter()
            .find(|ls| s >= ls.s_start && s < ls.s_end)
    }

    /// Total width of the road at a given s-coordinate (both sides combined).
    pub fn total_width_at(&self, s: f32) -> f32 {
        match self.lane_section_at(s) {
            Some(ls) => {
                let left: f32 = ls.left_lanes.iter().map(|l| l.width).sum();
                let right: f32 = ls.right_lanes.iter().map(|l| l.width).sum();
                left + right
            }
            None => 0.0,
        }
    }

    /// Get the lane center offset from the reference line for a given lane index.
    /// Positive lane indices are left, negative are right.
    /// Lane 1 = first left lane, lane -1 = first right lane, etc.
    pub fn lane_center_offset(&self, s: f32, lane_index: i32) -> Option<f32> {
        let section = self.lane_section_at(s)?;

        if lane_index == 0 {
            return Some(0.0); // on the reference line
        }

        let (lanes, sign) = if lane_index > 0 {
            (&section.left_lanes, 1.0f32)
        } else {
            (&section.right_lanes, -1.0f32)
        };

        let idx = (lane_index.unsigned_abs() as usize).checked_sub(1)?;
        if idx >= lanes.len() {
            return None;
        }

        // Accumulate widths from center to the middle of the target lane
        let mut offset = 0.0f32;
        for i in 0..idx {
            offset += lanes[i].width;
        }
        offset += lanes[idx].width / 2.0;

        Some(sign * offset)
    }
}

impl RoadNetwork {
    pub fn new() -> Self {
        Self { roads: Vec::new() }
    }

    pub fn add_road(&mut self, road: Road) -> usize {
        let id = self.roads.len();
        self.roads.push(road);
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    fn sample_road() -> Road {
        let points = vec![
            ControlPoint { position: Vec2::new(0.0, 0.0), turn_radius: 0.0, spiral_length: 0.0 },
            ControlPoint { position: Vec2::new(200.0, 0.0), turn_radius: 0.0, spiral_length: 0.0 },
        ];
        Road::new_with_defaults(points).unwrap()
    }

    #[test]
    fn default_road_has_four_lanes() {
        let road = sample_road();
        let section = &road.lane_sections[0];
        assert_eq!(section.left_lanes.len(), 2);
        assert_eq!(section.right_lanes.len(), 2);
    }

    #[test]
    fn total_width() {
        let road = sample_road();
        let w = road.total_width_at(50.0);
        assert!((w - 14.0).abs() < 0.01); // 4 lanes × 3.5m = 14m
    }

    #[test]
    fn lane_center_offset_first_left() {
        let road = sample_road();
        let offset = road.lane_center_offset(50.0, 1).unwrap();
        assert!((offset - 1.75).abs() < 0.01); // half of 3.5
    }

    #[test]
    fn lane_center_offset_second_left() {
        let road = sample_road();
        let offset = road.lane_center_offset(50.0, 2).unwrap();
        assert!((offset - 5.25).abs() < 0.01); // 3.5 + 1.75
    }

    #[test]
    fn lane_center_offset_first_right() {
        let road = sample_road();
        let offset = road.lane_center_offset(50.0, -1).unwrap();
        assert!((offset + 1.75).abs() < 0.01); // -1.75
    }

    #[test]
    fn lane_center_offset_invalid() {
        let road = sample_road();
        assert!(road.lane_center_offset(50.0, 5).is_none());
    }

    #[test]
    fn road_network_add() {
        let mut net = RoadNetwork::new();
        let road = sample_road();
        let id = net.add_road(road);
        assert_eq!(id, 0);
        assert_eq!(net.roads.len(), 1);
    }
}

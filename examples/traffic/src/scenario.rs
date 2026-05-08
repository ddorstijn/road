use std::path::Path;

use anyhow::{bail, Context};
use road::fitting::ControlPoint;
use road::network::{Road, RoadNetwork};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct ScenarioCar {
    pub road_id: usize,
    pub lane: i32,
    pub s: f32,
    pub speed: f32,
    pub desired_speed: f32,
}

#[derive(Debug, Deserialize)]
pub struct ScenarioRoad {
    pub control_points: Vec<ControlPoint>,
}

#[derive(Debug, Deserialize)]
pub struct Scenario {
    pub roads: Vec<ScenarioRoad>,
    pub cars: Vec<ScenarioCar>,
}

impl Scenario {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read scenario file: {}", path.display()))?;
        let scenario: Scenario =
            serde_json::from_str(&data).context("failed to parse scenario JSON")?;
        Ok(scenario)
    }

    pub fn build_network(&self) -> anyhow::Result<RoadNetwork> {
        let mut network = RoadNetwork::new();
        for (i, sr) in self.roads.iter().enumerate() {
            let road = Road::new_with_defaults(sr.control_points.clone())
                .with_context(|| format!("failed to build road {i} from control points"))?;
            network.add_road(road);
        }
        Ok(network)
    }

    pub fn validate(&self, network: &RoadNetwork) -> anyhow::Result<()> {
        for (i, car) in self.cars.iter().enumerate() {
            if car.road_id >= network.roads.len() {
                bail!(
                    "car {i}: road_id {} out of range (network has {} roads)",
                    car.road_id,
                    network.roads.len()
                );
            }
            let road = &network.roads[car.road_id];
            if car.s < 0.0 || car.s > road.reference_line.total_length {
                bail!(
                    "car {i}: s={} out of range [0, {}] for road {}",
                    car.s,
                    road.reference_line.total_length,
                    car.road_id
                );
            }
            if !road.lane_sections.is_empty() {
                let sec = &road.lane_sections[0];
                let valid = if car.lane >= 0 {
                    (car.lane as usize) < sec.right_lanes.len()
                } else {
                    ((-car.lane) as usize) <= sec.left_lanes.len()
                };
                if !valid {
                    bail!(
                        "car {i}: lane {} not valid for road {} (right={}, left={})",
                        car.lane,
                        car.road_id,
                        sec.right_lanes.len(),
                        sec.left_lanes.len()
                    );
                }
            }
        }
        Ok(())
    }
}

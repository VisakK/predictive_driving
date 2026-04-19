from __future__ import annotations

import numpy as np
import gymnasium as gym

from highway_env import utils
from highway_env.envs.merge_env import MergeEnv
from highway_env.envs.roundabout_env import RoundaboutEnv
from highway_env.vehicle.controller import ControlledVehicle


class MergeMoreAgentsEnv(MergeEnv):
    """Merge env with doubled traffic (8 others instead of 4) and optional raw rewards."""

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({"normalize_reward": True, "duration": 0})
        return cfg

    def _is_truncated(self) -> bool:
        return bool(self.config.get("duration", 0)) and self.time >= self.config["duration"]

    def _reward(self, action: int) -> float:
        reward = sum(
            self.config.get(name, 0) * reward
            for name, reward in self._rewards(action).items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"] + self.config["merging_speed_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        return reward

    def _make_vehicles(self) -> None:
        super()._make_vehicles()
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        for position, speed in [(120.0, 28.0), (50.0, 30.0), (30.0, 32.0)]:
            lane = road.network.get_lane(("a", "b", self.np_random.integers(2)))
            position = lane.position(position + self.np_random.uniform(-5.0, 5.0), 0.0)
            speed += self.np_random.uniform(-1.0, 1.0)
            road.vehicles.append(other_vehicles_type(road, position, speed=speed))

        merging_v = other_vehicles_type(
            road, road.network.get_lane(("j", "k", 0)).position(80.0, 0.0), speed=22.0
        )
        merging_v.target_speed = 30.0
        road.vehicles.append(merging_v)


class RoundaboutMoreAgentsEnv(RoundaboutEnv):
    """Roundabout env with doubled traffic (8 others instead of 4)."""

    def _make_vehicles(self) -> None:
        super()._make_vehicles()
        position_deviation = 2.0
        speed_deviation = 2.0
        destinations = ["exr", "sxr", "nxr"]
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        vehicle = other_vehicles_type.make_on_lane(
            self.road,
            ("ner", "nes", 0),
            longitudinal=50.0 + self.np_random.normal() * position_deviation,
            speed=16.0 + self.np_random.normal() * speed_deviation,
        )
        vehicle.plan_route_to(self.np_random.choice(destinations))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        vehicle = other_vehicles_type.make_on_lane(
            self.road,
            ("wer", "wes", 0),
            longitudinal=50.0 + self.np_random.normal() * position_deviation,
            speed=16.0 + self.np_random.normal() * speed_deviation,
        )
        vehicle.plan_route_to(self.np_random.choice(destinations))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        for i in list(range(1, 2)) + list(range(-1, 0)):
            vehicle = other_vehicles_type.make_on_lane(
                self.road,
                ("ee", "nx", 0),
                longitudinal=20.0 * float(i)
                + self.np_random.normal() * position_deviation,
                speed=16.0 + self.np_random.normal() * speed_deviation,
            )
            vehicle.plan_route_to(self.np_random.choice(destinations))
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)


gym.register(id="merge-moreagents-v0", entry_point="driving.envs:MergeMoreAgentsEnv")
gym.register(id="roundabout-moreagents-v0", entry_point="driving.envs:RoundaboutMoreAgentsEnv")

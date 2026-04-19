from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from highway_env import utils
from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

from driving.envs import RoundaboutMoreAgentsEnv


class AdversarialIDMVehicle(IDMVehicle):
    """IDM vehicle with aggressive driving parameters (~15% of traffic)."""

    TIME_WANTED = 0.5
    DISTANCE_WANTED = 2.0 + ControlledVehicle.LENGTH
    COMFORT_ACC_MAX = 5.0
    COMFORT_ACC_MIN = -8.0
    LANE_CHANGE_MIN_ACC_GAIN = 0.05
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 4.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_adversarial = True


class AdversarialHighwayEnv(HighwayEnv):
    """Highway env injecting ~15% adversarial agents into traffic."""

    ADVERSARIAL_RATIO = 0.15

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "observation": {
                "type": "OccupancyGrid",
                "features": ["presence", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {"vx": [-20, 20], "vy": [-20, 20]},
                "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
                "grid_step": [5, 5],
                "absolute": False,
            },
            "vehicles_count": 100,
            "duration": 80,
            "simulation_frequency": 5,
            "policy_frequency": 1,
            "normalize_reward": False,
        })
        return cfg

    def _create_vehicles(self) -> None:
        other_vehicles_type = utils.class_from_path(
            self.config["other_vehicles_type"]
        )
        other_per_controlled = utils.near_split(
            self.config["vehicles_count"],
            num_bins=self.config["controlled_vehicles"],
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25.0,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                if self.np_random.random() < self.ADVERSARIAL_RATIO:
                    v = AdversarialIDMVehicle.create_random(
                        self.road,
                        spacing=1 / self.config["vehicles_density"],
                    )
                else:
                    v = other_vehicles_type.create_random(
                        self.road,
                        spacing=1 / self.config["vehicles_density"],
                    )
                v.randomize_behavior()
                self.road.vehicles.append(v)


class AdversarialRoundaboutEnv(RoundaboutMoreAgentsEnv):
    """Roundabout env injecting ~15% adversarial agents into traffic."""

    ADVERSARIAL_RATIO = 0.15

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "observation": {
                "type": "OccupancyGrid",
                "features": ["presence", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {"vx": [-20, 20], "vy": [-20, 20]},
                "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
                "grid_step": [5, 5],
                "absolute": False,
            },
            "simulation_frequency": 5,
            "policy_frequency": 1,
            "normalize_reward": False,
        })
        return cfg

    def _make_vehicles(self) -> None:
        super()._make_vehicles()
        ego = self.vehicle
        for i, v in enumerate(self.road.vehicles):
            if v is ego:
                continue
            if self.np_random.random() < self.ADVERSARIAL_RATIO:
                adv = AdversarialIDMVehicle(
                    self.road,
                    v.position,
                    heading=v.heading,
                    speed=v.speed,
                    target_lane_index=v.target_lane_index,
                    target_speed=v.target_speed,
                    route=v.route,
                )
                adv.randomize_behavior()
                self.road.vehicles[i] = adv


class DictObsWrapper(gym.ObservationWrapper):
    """Wraps env to provide Dict observation: occupancy_grid + agent_kinematics."""

    N_AGENTS = 15
    AGENT_FEATURES = 7  # presence, rel_x, rel_y, rel_vx, rel_vy, cos_h, sin_h

    def __init__(self, env: gym.Env, n_agents: int = 15):
        super().__init__(env)
        self.N_AGENTS = n_agents
        grid_space = env.observation_space
        kin_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.N_AGENTS, self.AGENT_FEATURES),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict({
            "occupancy_grid": spaces.Box(
                low=grid_space.low.astype(np.float32),
                high=grid_space.high.astype(np.float32),
                shape=grid_space.shape,
                dtype=np.float32,
            ),
            "agent_kinematics": kin_space,
        })

    def observation(self, obs):
        return {
            "occupancy_grid": obs.astype(np.float32),
            "agent_kinematics": self._get_agent_kinematics(),
        }

    def _get_agent_kinematics(self) -> np.ndarray:
        base_env = self.unwrapped
        vehicles = base_env.road.vehicles
        ego = base_env.vehicle
        kin = np.zeros((self.N_AGENTS, self.AGENT_FEATURES), dtype=np.float32)

        others = [v for v in vehicles if v is not ego]
        others.sort(key=lambda v: np.sum((v.position - ego.position) ** 2))

        ego_cos = np.cos(ego.heading)
        ego_sin = np.sin(ego.heading)

        for i, v in enumerate(others[: self.N_AGENTS]):
            dx = v.position[0] - ego.position[0]
            dy = v.position[1] - ego.position[1]
            dvx = v.speed * np.cos(v.heading) - ego.speed * ego_cos
            dvy = v.speed * np.sin(v.heading) - ego.speed * ego_sin
            kin[i] = [
                1.0,
                np.clip(dx / 30.0, -1, 1),
                np.clip(dy / 30.0, -1, 1),
                np.clip(dvx / 20.0, -1, 1),
                np.clip(dvy / 20.0, -1, 1),
                np.cos(v.heading - ego.heading),
                np.sin(v.heading - ego.heading),
            ]
        return kin

    def _get_adversarial_mask(self) -> np.ndarray:
        """Return boolean mask: True for adversarial agents (sorted by distance)."""
        base_env = self.unwrapped
        vehicles = base_env.road.vehicles
        ego = base_env.vehicle
        others = [v for v in vehicles if v is not ego]
        others.sort(key=lambda v: np.sum((v.position - ego.position) ** 2))
        mask = np.zeros(self.N_AGENTS, dtype=bool)
        for i, v in enumerate(others[: self.N_AGENTS]):
            mask[i] = getattr(v, "is_adversarial", False)
        return mask

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["adversarial_mask"] = self._get_adversarial_mask()
        info["crashed_with_adversarial"] = self._check_adversarial_crash()
        return self.observation(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["adversarial_mask"] = self._get_adversarial_mask()
        return self.observation(obs), info

    def _check_adversarial_crash(self) -> bool:
        base_env = self.unwrapped
        if not base_env.vehicle.crashed:
            return False
        for v in base_env.road.vehicles:
            if v is base_env.vehicle:
                continue
            if v.crashed and getattr(v, "is_adversarial", False):
                return True
        return False


def make_adversarial_highway(**kwargs):
    env = AdversarialHighwayEnv(**kwargs)
    return DictObsWrapper(env)


def make_adversarial_roundabout(**kwargs):
    env = AdversarialRoundaboutEnv(**kwargs)
    return DictObsWrapper(env)


gym.register(
    id="adversarial-highway-v0",
    entry_point="driving.adversarial:make_adversarial_highway",
)
gym.register(
    id="adversarial-roundabout-v0",
    entry_point="driving.adversarial:make_adversarial_roundabout",
)

from __future__ import annotations

from collections import deque

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


class CutInIDMVehicle(IDMVehicle):
    """Adversarial IDM that deliberately cuts in front of the ego.

    Triggers a forced ``target_lane_index = ego.lane_index`` (bypassing MOBIL
    safety) when:
      - ego is in a side lane reachable from this vehicle's current lane,
      - ego is *behind* this vehicle within ``EGO_DETECTION_RANGE_LON``
        (in self frame; we cut in front of someone we're already ahead of),
      - ego is laterally within ``EGO_DETECTION_RANGE_LAT``,
      - the ego is not pulling away (component of ego velocity along self
        heading minus self speed >= ``EGO_CLOSING_THRESHOLD``),
      - ``_cooldown_steps`` has elapsed since the last cut-in.

    Once triggered, the target lane is held for ``COMMIT_STEPS`` so the lane
    change can complete even if conditions briefly stop matching, then a
    cooldown begins. The longitudinal IDM controller stays nominal so the
    vehicle does not crash into its own leader.

    Time bookkeeping uses sim-step counts (one decrement per
    ``change_lane_policy`` call); at the standard simulation_frequency=5,
    one step ≈ 0.2s of real time.
    """

    ARCHETYPE = "cut_in"

    EGO_DETECTION_RANGE_LON = 25.0  # [m]
    EGO_DETECTION_RANGE_LAT = 6.0   # [m]
    EGO_CLOSING_THRESHOLD = -1.0    # [m/s]: ego_v_along - self_v >= this
    COOLDOWN_STEPS = 25             # ~5s at sim_freq=5Hz
    COMMIT_STEPS = 10               # ~2s at sim_freq=5Hz

    POLITENESS = 0.0
    LANE_CHANGE_MIN_ACC_GAIN = 0.0
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 4.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_adversarial = True
        self.archetype = self.ARCHETYPE
        self._cooldown_steps = 0
        self._commit_steps = 0
        self._committed_target_lane = None

    def _randomize_archetype(self, rng) -> None:
        # Stagger initial cooldowns so cut-ins from different vehicles don't
        # synchronously fire at the first opportunity.
        self._cooldown_steps = int(rng.integers(0, self.COOLDOWN_STEPS + 1))

    def _ego(self):
        for v in self.road.vehicles:
            if isinstance(v, ControlledVehicle) and not getattr(v, "is_adversarial", False):
                return v
        return None

    def _ego_target_lane(self):
        """Return the ego's lane_index if cut-in conditions are met, else None."""
        ego = self._ego()
        if ego is None:
            return None
        if ego.lane_index == self.lane_index:
            return None
        try:
            side_lanes = list(self.road.network.side_lanes(self.lane_index))
        except Exception:
            return None
        if ego.lane_index not in side_lanes:
            return None

        dx_world = float(ego.position[0] - self.position[0])
        dy_world = float(ego.position[1] - self.position[1])
        cos_h = float(np.cos(self.heading))
        sin_h = float(np.sin(self.heading))
        dx = cos_h * dx_world + sin_h * dy_world  # +x ahead of self
        dy = -sin_h * dx_world + cos_h * dy_world

        # Ego must be behind us (we cut in front of someone catching up).
        if dx > 0:
            return None
        if dx < -self.EGO_DETECTION_RANGE_LON:
            return None
        if abs(dy) > self.EGO_DETECTION_RANGE_LAT:
            return None

        v_self = float(self.speed)
        v_ego_along = float(ego.speed * np.cos(ego.heading - self.heading))
        if (v_ego_along - v_self) < self.EGO_CLOSING_THRESHOLD:
            return None

        return ego.lane_index

    def change_lane_policy(self) -> None:
        if self._cooldown_steps > 0:
            self._cooldown_steps -= 1
        if self._commit_steps > 0:
            self._commit_steps -= 1

        # While committed to a cut-in, hold the target so the lane change
        # completes even if the ego briefly drops out of the trigger zone.
        if self._commit_steps > 0 and self._committed_target_lane is not None:
            self.target_lane_index = self._committed_target_lane
            return

        if self._cooldown_steps <= 0:
            target = self._ego_target_lane()
            if target is not None:
                self.target_lane_index = target
                self._committed_target_lane = target
                self._commit_steps = self.COMMIT_STEPS
                self._cooldown_steps = self.COOLDOWN_STEPS + self.COMMIT_STEPS
                return

        super().change_lane_policy()


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


class OccGridFrameStack(gym.Wrapper):
    """Stacks the last n_frames occupancy grids along the channel dimension."""

    def __init__(self, env: gym.Env, n_frames: int = 3):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames: deque = deque(maxlen=n_frames)

        old_space = env.observation_space
        if isinstance(old_space, spaces.Dict):
            grid_space = old_space["occupancy_grid"]
            new_low = np.repeat(grid_space.low, n_frames, axis=0)
            new_high = np.repeat(grid_space.high, n_frames, axis=0)
            self.observation_space = spaces.Dict({
                "occupancy_grid": spaces.Box(low=new_low, high=new_high, dtype=np.float32),
                "agent_kinematics": old_space["agent_kinematics"],
            })
            self._dict_obs = True
        else:
            new_low = np.repeat(old_space.low, n_frames, axis=0)
            new_high = np.repeat(old_space.high, n_frames, axis=0)
            self.observation_space = spaces.Box(low=new_low, high=new_high, dtype=old_space.dtype)
            self._dict_obs = False

    def _get_grid(self, obs):
        return obs["occupancy_grid"] if self._dict_obs else obs

    def _set_grid(self, obs, stacked):
        if self._dict_obs:
            return {**obs, "occupancy_grid": stacked}
        return stacked

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        grid = self._get_grid(obs)
        for _ in range(self.n_frames):
            self.frames.append(grid.copy())
        stacked = np.concatenate(list(self.frames), axis=0)
        return self._set_grid(obs, stacked), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        grid = self._get_grid(obs)
        self.frames.append(grid.copy())
        stacked = np.concatenate(list(self.frames), axis=0)
        return self._set_grid(obs, stacked), reward, terminated, truncated, info


def _convert_to_adversarial(vehicle) -> None:
    """In-place class swap of an IDMVehicle to AdversarialIDMVehicle."""
    vehicle.__class__ = AdversarialIDMVehicle
    vehicle.is_adversarial = True


def _proximity_pick(
    distances: np.ndarray,
    n_pick: int,
    n_forced_near: int,
    near_pool_size: int,
    rng,
) -> np.ndarray:
    """Pick n_pick indices: force n_forced_near of them into the nearest near_pool_size,
    sample the remainder uniformly from the rest of the population.

    This guarantees adversarial agents interact with the ego without turning the
    ego's vicinity into a minefield (which pure exponential weighting does).
    """
    n = len(distances)
    n_pick = min(n_pick, n)
    sorted_idx = np.argsort(distances)
    near_pool = sorted_idx[: min(near_pool_size, n)]
    far_pool = sorted_idx[min(near_pool_size, n):]

    n_forced_near = min(n_forced_near, n_pick, len(near_pool))
    near_picks = rng.choice(near_pool, size=n_forced_near, replace=False)

    rest = n_pick - n_forced_near
    if rest > 0 and len(far_pool) > 0:
        # Allow sampling from both remaining near and far pool uniformly
        remaining_pool = np.setdiff1d(sorted_idx, near_picks)
        far_picks = rng.choice(
            remaining_pool, size=min(rest, len(remaining_pool)), replace=False
        )
    else:
        far_picks = np.array([], dtype=int)

    return np.concatenate([near_picks, far_picks]).astype(int)


class AdversarialHighwayV2Env(AdversarialHighwayEnv):
    """Highway env with proximity-guaranteed adversarial placement (v2).

    Creates all traffic as nominal IDM vehicles, then promotes ~15% of them
    to AdversarialIDMVehicle with a structured pattern: at least
    N_FORCED_NEAR adversaries are placed among the NEAR_POOL_SIZE nearest to
    ego, and the remainder are sampled uniformly from the rest. This
    guarantees adversarial interaction without making the vicinity a minefield.
    """

    N_FORCED_NEAR = 3
    NEAR_POOL_SIZE = 8

    def _create_vehicles(self) -> None:
        other_vehicles_type = utils.class_from_path(
            self.config["other_vehicles_type"]
        )
        other_per_controlled = utils.near_split(
            self.config["vehicles_count"],
            num_bins=self.config["controlled_vehicles"],
        )

        self.controlled_vehicles = []
        nominal_others: list = []
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
                v = other_vehicles_type.create_random(
                    self.road,
                    spacing=1 / self.config["vehicles_density"],
                )
                v.randomize_behavior()
                self.road.vehicles.append(v)
                nominal_others.append(v)

        ego = self.controlled_vehicles[0]
        if not nominal_others:
            return
        distances = np.array([
            np.linalg.norm(np.asarray(v.position) - np.asarray(ego.position))
            for v in nominal_others
        ])
        n_adv = int(round(len(nominal_others) * self.ADVERSARIAL_RATIO))
        if n_adv <= 0:
            return
        picked = _proximity_pick(
            distances,
            n_adv,
            self.N_FORCED_NEAR,
            self.NEAR_POOL_SIZE,
            self.np_random,
        )
        for k in picked:
            _convert_to_adversarial(nominal_others[k])


class AdversarialRoundaboutV2Env(AdversarialRoundaboutEnv):
    """Roundabout env with proximity-guaranteed adversarial placement (v2).

    The roundabout has far fewer vehicles (~9), so we force 2 adversaries
    into the nearest 4 and fill the rest from the remaining pool.
    """

    N_FORCED_NEAR = 2
    NEAR_POOL_SIZE = 4

    def _make_vehicles(self) -> None:
        RoundaboutMoreAgentsEnv._make_vehicles(self)
        ego = self.vehicle
        others = [v for v in self.road.vehicles if v is not ego]
        if not others:
            return
        distances = np.array([
            np.linalg.norm(np.asarray(v.position) - np.asarray(ego.position))
            for v in others
        ])
        # Roundabout has ~9 vehicles; force at least 2 near, 15% of population
        # gives <=2 so this wrapper will typically place 2 near (and possibly 0 far).
        n_adv = max(
            int(round(len(others) * self.ADVERSARIAL_RATIO)), self.N_FORCED_NEAR
        )
        if n_adv <= 0:
            return
        picked = _proximity_pick(
            distances,
            n_adv,
            self.N_FORCED_NEAR,
            self.NEAR_POOL_SIZE,
            self.np_random,
        )
        for k in picked:
            _convert_to_adversarial(others[k])


class KinHistoryDictObsWrapper(gym.ObservationWrapper):
    """Dict observation with occupancy_grid, agent_kinematics (current), and
    agent_kin_history (N-frame per-agent trajectory, identity-tracked)."""

    N_AGENTS = 15
    AGENT_FEATURES = 7
    N_FRAMES = 10

    def __init__(self, env: gym.Env, n_agents: int = 15, n_frames: int = 10):
        super().__init__(env)
        self.N_AGENTS = n_agents
        self.N_FRAMES = n_frames
        grid_space = env.observation_space
        self.observation_space = spaces.Dict({
            "occupancy_grid": spaces.Box(
                low=grid_space.low.astype(np.float32),
                high=grid_space.high.astype(np.float32),
                shape=grid_space.shape,
                dtype=np.float32,
            ),
            "agent_kinematics": spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.N_AGENTS, self.AGENT_FEATURES),
                dtype=np.float32,
            ),
            "agent_kin_history": spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.N_AGENTS, self.N_FRAMES, self.AGENT_FEATURES),
                dtype=np.float32,
            ),
        })
        self._history: dict[int, deque] = {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._history = {}
        self._update_histories()
        info["adversarial_mask"] = self._get_adversarial_mask()
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_histories()
        info["adversarial_mask"] = self._get_adversarial_mask()
        info["crashed_with_adversarial"] = self._check_adversarial_crash()
        return self.observation(obs), reward, terminated, truncated, info

    def observation(self, obs):
        return {
            "occupancy_grid": obs.astype(np.float32),
            "agent_kinematics": self._get_current_kinematics(),
            "agent_kin_history": self._get_kinematics_history(),
        }

    def _vehicle_features(self, v, ego) -> np.ndarray:
        ego_cos = np.cos(ego.heading)
        ego_sin = np.sin(ego.heading)
        dx = v.position[0] - ego.position[0]
        dy = v.position[1] - ego.position[1]
        dvx = v.speed * np.cos(v.heading) - ego.speed * ego_cos
        dvy = v.speed * np.sin(v.heading) - ego.speed * ego_sin
        return np.array([
            1.0,
            np.clip(dx / 30.0, -1, 1),
            np.clip(dy / 30.0, -1, 1),
            np.clip(dvx / 20.0, -1, 1),
            np.clip(dvy / 20.0, -1, 1),
            np.cos(v.heading - ego.heading),
            np.sin(v.heading - ego.heading),
        ], dtype=np.float32)

    def _update_histories(self) -> None:
        base = self.unwrapped
        ego = base.vehicle
        seen: set[int] = set()
        for v in base.road.vehicles:
            if v is ego:
                continue
            vid = id(v)
            seen.add(vid)
            feats = self._vehicle_features(v, ego)
            if vid not in self._history:
                self._history[vid] = deque(maxlen=self.N_FRAMES)
            self._history[vid].append(feats)
        # Purge vehicles no longer in the scene
        for vid in list(self._history.keys()):
            if vid not in seen:
                del self._history[vid]

    def _sorted_nearest(self):
        base = self.unwrapped
        ego = base.vehicle
        others = [v for v in base.road.vehicles if v is not ego]
        others.sort(key=lambda v: np.sum((v.position - ego.position) ** 2))
        return others[: self.N_AGENTS]

    def _get_current_kinematics(self) -> np.ndarray:
        out = np.zeros((self.N_AGENTS, self.AGENT_FEATURES), dtype=np.float32)
        for i, v in enumerate(self._sorted_nearest()):
            hist = self._history.get(id(v))
            if hist:
                out[i] = hist[-1]
        return out

    def _get_kinematics_history(self) -> np.ndarray:
        out = np.zeros(
            (self.N_AGENTS, self.N_FRAMES, self.AGENT_FEATURES), dtype=np.float32
        )
        for i, v in enumerate(self._sorted_nearest()):
            hist = self._history.get(id(v))
            if not hist:
                continue
            hist_arr = np.stack(list(hist), axis=0)  # (len, 7), len <= N_FRAMES
            pad_n = self.N_FRAMES - hist_arr.shape[0]
            if pad_n > 0:
                out[i, pad_n:] = hist_arr
            else:
                out[i] = hist_arr
        return out

    def _get_adversarial_mask(self) -> np.ndarray:
        mask = np.zeros(self.N_AGENTS, dtype=bool)
        for i, v in enumerate(self._sorted_nearest()):
            mask[i] = getattr(v, "is_adversarial", False)
        return mask

    def _check_adversarial_crash(self) -> bool:
        base = self.unwrapped
        if not base.vehicle.crashed:
            return False
        for v in base.road.vehicles:
            if v is base.vehicle:
                continue
            if v.crashed and getattr(v, "is_adversarial", False):
                return True
        return False


class ExpectedObservedDictObsWrapper(KinHistoryDictObsWrapper):
    """Adds causal expected-vs-observed anomaly features.

    The anomaly at time t compares the current observed kinematics with a
    constant-velocity prediction made from the agent's previous two tracked
    states. This is causal: inference never uses future ground truth, only the
    next observation after a prior prediction.
    """

    ANOMALY_FEATURES = 4  # presence, anomaly, risk, prediction_error

    def __init__(self, env: gym.Env, n_agents: int = 15, n_frames: int = 10):
        super().__init__(env, n_agents=n_agents, n_frames=n_frames)
        self._last_anomaly: dict[int, np.ndarray] = {}
        self.observation_space = spaces.Dict({
            **self.observation_space.spaces,
            "agent_anomaly": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.N_AGENTS, self.ANOMALY_FEATURES),
                dtype=np.float32,
            ),
        })

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._history = {}
        self._last_anomaly = {}
        self._update_histories_and_anomalies()
        info["adversarial_mask"] = self._get_adversarial_mask()
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_histories_and_anomalies()
        info["adversarial_mask"] = self._get_adversarial_mask()
        info["crashed_with_adversarial"] = self._check_adversarial_crash()
        info["expected_observed_anomaly"] = float(
            self._get_anomaly_features()[:, 2].max(initial=0.0)
        )
        return self.observation(obs), reward, terminated, truncated, info

    def observation(self, obs):
        out = super().observation(obs)
        out["agent_anomaly"] = self._get_anomaly_features()
        return out

    def _update_histories(self) -> None:
        self._update_histories_and_anomalies()

    def _update_histories_and_anomalies(self) -> None:
        base = self.unwrapped
        ego = base.vehicle
        seen: set[int] = set()
        next_anomaly: dict[int, np.ndarray] = {}

        for v in base.road.vehicles:
            if v is ego:
                continue
            vid = id(v)
            seen.add(vid)
            feats = self._vehicle_features(v, ego)
            hist = self._history.get(vid)

            if hist is not None and len(hist) >= 2:
                prev = hist[-2]
                last = hist[-1]
                pred = last + (last - prev)
                err_vec = feats - pred
                pos_err = float(np.linalg.norm(err_vec[1:3]))
                vel_err = float(np.linalg.norm(err_vec[3:5]))
                heading_err = float(np.linalg.norm(err_vec[5:7]))
                raw_error = pos_err + 0.5 * vel_err + 0.25 * heading_err
                anomaly = float(np.clip(raw_error / 0.35, 0.0, 1.0))
            else:
                raw_error = 0.0
                anomaly = 0.0

            distance = float(np.linalg.norm(feats[1:3]))
            proximity = float(np.clip(1.0 - distance, 0.0, 1.0))
            closing_speed = float(np.clip(-feats[3], 0.0, 1.0))
            risk = float(np.clip(anomaly * (0.5 + 0.5 * proximity) * (0.5 + 0.5 * closing_speed), 0.0, 1.0))
            next_anomaly[vid] = np.array(
                [1.0, anomaly, risk, np.clip(raw_error, 0.0, 1.0)],
                dtype=np.float32,
            )

            if hist is None:
                hist = deque(maxlen=self.N_FRAMES)
                self._history[vid] = hist
            hist.append(feats)

        for vid in list(self._history.keys()):
            if vid not in seen:
                del self._history[vid]
        self._last_anomaly = next_anomaly

    def _get_anomaly_features(self) -> np.ndarray:
        out = np.zeros(
            (self.N_AGENTS, self.ANOMALY_FEATURES),
            dtype=np.float32,
        )
        for i, v in enumerate(self._sorted_nearest()):
            feats = self._last_anomaly.get(id(v))
            if feats is not None:
                out[i] = feats
        return out


class HorizonExpectedObservedDictObsWrapper(KinHistoryDictObsWrapper):
    """Causal H-step expected-vs-observed anomaly features.

    At each step, the wrapper predicts the next ``horizon`` kinematic states for
    every tracked agent using constant-velocity extrapolation. Predictions are
    stored in a pending queue. As future observations arrive, each observation is
    scored against all predictions that mature at that time. The policy only
    sees anomaly from predictions made in the past, so the signal is valid at
    inference time.
    """

    ANOMALY_FEATURES = 4  # presence, anomaly, risk, prediction_error

    def __init__(
        self,
        env: gym.Env,
        n_agents: int = 15,
        n_frames: int = 10,
        horizon: int = 10,
    ):
        super().__init__(env, n_agents=n_agents, n_frames=n_frames)
        self.horizon = horizon
        self._pending_predictions: dict[int, list[tuple[int, np.ndarray]]] = {}
        self._last_anomaly: dict[int, np.ndarray] = {}
        self.observation_space = spaces.Dict({
            **self.observation_space.spaces,
            "agent_anomaly": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.N_AGENTS, self.ANOMALY_FEATURES),
                dtype=np.float32,
            ),
        })

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._history = {}
        self._pending_predictions = {}
        self._last_anomaly = {}
        self._update_histories_and_anomalies()
        info["adversarial_mask"] = self._get_adversarial_mask()
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_histories_and_anomalies()
        info["adversarial_mask"] = self._get_adversarial_mask()
        info["crashed_with_adversarial"] = self._check_adversarial_crash()
        info["expected_observed_anomaly"] = float(
            self._get_anomaly_features()[:, 2].max(initial=0.0)
        )
        return self.observation(obs), reward, terminated, truncated, info

    def observation(self, obs):
        out = super().observation(obs)
        out["agent_anomaly"] = self._get_anomaly_features()
        return out

    def _update_histories(self) -> None:
        self._update_histories_and_anomalies()

    @staticmethod
    def _prediction_error(feats: np.ndarray, pred: np.ndarray) -> float:
        err_vec = feats - pred
        pos_err = float(np.linalg.norm(err_vec[1:3]))
        vel_err = float(np.linalg.norm(err_vec[3:5]))
        heading_err = float(np.linalg.norm(err_vec[5:7]))
        return pos_err + 0.5 * vel_err + 0.25 * heading_err

    def _future_predictions(
        self,
        current: np.ndarray,
        previous: np.ndarray,
    ) -> np.ndarray:
        delta = current - previous
        preds = []
        for h in range(1, self.horizon + 1):
            pred = current + h * delta
            pred = np.clip(pred, -1.0, 1.0)
            pred[0] = 1.0
            preds.append(pred.astype(np.float32))
        return np.stack(preds, axis=0)

    def _update_histories_and_anomalies(self) -> None:
        base = self.unwrapped
        ego = base.vehicle
        seen: set[int] = set()
        next_anomaly: dict[int, np.ndarray] = {}

        for v in base.road.vehicles:
            if v is ego:
                continue
            vid = id(v)
            seen.add(vid)
            feats = self._vehicle_features(v, ego)
            hist = self._history.get(vid)

            pending = self._pending_predictions.get(vid, [])
            kept_pending: list[tuple[int, np.ndarray]] = []
            matured_errors: list[float] = []
            for age, preds in pending:
                matured_errors.append(self._prediction_error(feats, preds[age]))
                if age + 1 < self.horizon:
                    kept_pending.append((age + 1, preds))

            raw_error = float(np.mean(matured_errors)) if matured_errors else 0.0
            anomaly = float(np.clip(raw_error / 0.35, 0.0, 1.0))

            distance = float(np.linalg.norm(feats[1:3]))
            proximity = float(np.clip(1.0 - distance, 0.0, 1.0))
            closing_speed = float(np.clip(-feats[3], 0.0, 1.0))
            risk = float(
                np.clip(
                    anomaly
                    * (0.5 + 0.5 * proximity)
                    * (0.5 + 0.5 * closing_speed),
                    0.0,
                    1.0,
                )
            )
            next_anomaly[vid] = np.array(
                [1.0, anomaly, risk, np.clip(raw_error, 0.0, 1.0)],
                dtype=np.float32,
            )

            if hist is None:
                hist = deque(maxlen=self.N_FRAMES)
                self._history[vid] = hist
            elif len(hist) >= 1:
                kept_pending.append((0, self._future_predictions(feats, hist[-1])))
            self._pending_predictions[vid] = kept_pending
            hist.append(feats)

        for vid in list(self._history.keys()):
            if vid not in seen:
                del self._history[vid]
        for vid in list(self._pending_predictions.keys()):
            if vid not in seen:
                del self._pending_predictions[vid]
        self._last_anomaly = next_anomaly

    def _get_anomaly_features(self) -> np.ndarray:
        out = np.zeros(
            (self.N_AGENTS, self.ANOMALY_FEATURES),
            dtype=np.float32,
        )
        for i, v in enumerate(self._sorted_nearest()):
            feats = self._last_anomaly.get(id(v))
            if feats is not None:
                out[i] = feats
        return out


def make_adversarial_highway(**kwargs):
    env = AdversarialHighwayEnv(**kwargs)
    return DictObsWrapper(env)


def make_adversarial_roundabout(**kwargs):
    env = AdversarialRoundaboutEnv(**kwargs)
    return DictObsWrapper(env)


def make_adversarial_highway_framestack(**kwargs):
    env = AdversarialHighwayEnv(**kwargs)
    env = DictObsWrapper(env)
    return OccGridFrameStack(env, n_frames=3)


def make_adversarial_roundabout_framestack(**kwargs):
    env = AdversarialRoundaboutEnv(**kwargs)
    env = DictObsWrapper(env)
    return OccGridFrameStack(env, n_frames=3)


def make_adversarial_highway_v2(**kwargs):
    env = AdversarialHighwayV2Env(**kwargs)
    return KinHistoryDictObsWrapper(env)


def make_adversarial_roundabout_v2(**kwargs):
    env = AdversarialRoundaboutV2Env(**kwargs)
    return KinHistoryDictObsWrapper(env)


def make_adversarial_highway_expected(**kwargs):
    env = AdversarialHighwayV2Env(**kwargs)
    return ExpectedObservedDictObsWrapper(env)


def make_adversarial_roundabout_expected(**kwargs):
    env = AdversarialRoundaboutV2Env(**kwargs)
    return ExpectedObservedDictObsWrapper(env)


def make_adversarial_highway_expected_h10(**kwargs):
    env = AdversarialHighwayV2Env(**kwargs)
    return HorizonExpectedObservedDictObsWrapper(env, horizon=10)


def make_adversarial_roundabout_expected_h10(**kwargs):
    env = AdversarialRoundaboutV2Env(**kwargs)
    return HorizonExpectedObservedDictObsWrapper(env, horizon=10)


gym.register(
    id="adversarial-highway-v0",
    entry_point="driving.adversarial:make_adversarial_highway",
)
gym.register(
    id="adversarial-roundabout-v0",
    entry_point="driving.adversarial:make_adversarial_roundabout",
)
gym.register(
    id="adversarial-highway-framestack-v0",
    entry_point="driving.adversarial:make_adversarial_highway_framestack",
)
gym.register(
    id="adversarial-roundabout-framestack-v0",
    entry_point="driving.adversarial:make_adversarial_roundabout_framestack",
)
gym.register(
    id="adversarial-highway-v2",
    entry_point="driving.adversarial:make_adversarial_highway_v2",
)
gym.register(
    id="adversarial-roundabout-v2",
    entry_point="driving.adversarial:make_adversarial_roundabout_v2",
)
gym.register(
    id="adversarial-highway-expected-v0",
    entry_point="driving.adversarial:make_adversarial_highway_expected",
)
gym.register(
    id="adversarial-roundabout-expected-v0",
    entry_point="driving.adversarial:make_adversarial_roundabout_expected",
)
gym.register(
    id="adversarial-highway-expected-h10-v0",
    entry_point="driving.adversarial:make_adversarial_highway_expected_h10",
)
gym.register(
    id="adversarial-roundabout-expected-h10-v0",
    entry_point="driving.adversarial:make_adversarial_roundabout_expected_h10",
)

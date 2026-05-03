"""Adversarial v3 — behavioral archetype mixture.

Defines four qualitatively distinct adversarial archetypes that subclass IDMVehicle
and share an `AdversarialBehaviorMixin`. The v3 highway env reuses v2's
proximity-guaranteed placement but samples each adversary's archetype from a
configurable categorical distribution.

This module imports from `driving.adversarial` and does NOT modify it.
See experiments/050_adversarial_v3/design.md for rationale.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym

from highway_env import utils
from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

from driving.adversarial import (
    AdversarialHighwayV2Env,
    DictObsWrapper,
    HorizonExpectedObservedDictObsWrapper,
    KinHistoryDictObsWrapper,
    _proximity_pick,
)


# ---------------------------------------------------------------------------
# Adversarial marking helper
# ---------------------------------------------------------------------------
#
# Note: archetype classes inherit IDMVehicle *directly* (no extra mixin base).
# Python's __class__ assignment requires identical C-level object layout, so
# multi-base mixins in front of IDMVehicle are rejected. Behavior is provided
# via per-class method overrides; shared bookkeeping lives in mark_adversarial.


def mark_adversarial(vehicle, archetype: str) -> None:
    vehicle.is_adversarial = True
    vehicle.archetype = archetype


def randomize_archetype(vehicle, rng: np.random.Generator) -> None:
    """Dispatch to per-archetype seeding (no-op for archetypes without state)."""
    fn = getattr(vehicle, "_randomize_archetype", None)
    if fn is not None:
        fn(rng)


# ---------------------------------------------------------------------------
# Archetype 1 — Tailgater (≈ existing AdversarialIDMVehicle)
# ---------------------------------------------------------------------------


class TailgaterVehicle(IDMVehicle):
    ARCHETYPE = "tailgater"

    TIME_WANTED = 0.5
    DISTANCE_WANTED = 2.0 + ControlledVehicle.LENGTH
    COMFORT_ACC_MAX = 5.0
    COMFORT_ACC_MIN = -8.0
    POLITENESS = 0.0
    LANE_CHANGE_MIN_ACC_GAIN = 0.05
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 4.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mark_adversarial(self, self.ARCHETYPE)


# ---------------------------------------------------------------------------
# Archetype 2 — SuddenBraker
# ---------------------------------------------------------------------------


class SuddenBrakerVehicle(IDMVehicle):
    """Triggers hard-brake bursts when the ego is close in front.

    Trigger conditions (all must hold):
      - euclidean distance to ego <= PROXIMITY_THRESHOLD
      - ego is in front (dx > 0 in self frame) and laterally close (|dy| < LATERAL_BAND)
      - closing speed (v_self_along - v_ego_along) >= 0 (not pulling away)
      - cooldown elapsed since last brake event

    Once triggered, the vehicle outputs `BRAKE_ACCEL` for a sampled duration
    drawn from Uniform(BRAKE_DURATION_MIN, BRAKE_DURATION_MAX).
    """

    ARCHETYPE = "sudden_braker"

    PROXIMITY_THRESHOLD = 25.0  # [m]
    LATERAL_BAND = 8.0  # [m]
    BRAKE_ACCEL = -8.0  # [m/s^2]
    BRAKE_DURATION_MIN = 1.0  # [s]
    BRAKE_DURATION_MAX = 2.0  # [s]
    COOLDOWN = 5.0  # [s]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mark_adversarial(self, self.ARCHETYPE)
        self._brake_remaining = 0.0
        self._cooldown_remaining = 0.0
        self._brake_duration_default = 1.5

    def _randomize_archetype(self, rng: np.random.Generator) -> None:
        # Stagger initial cooldowns so not all brakers fire at the same step
        self._cooldown_remaining = float(rng.uniform(0.0, self.COOLDOWN))

    def _ego(self):
        for v in self.road.vehicles:
            if isinstance(v, ControlledVehicle) and getattr(v, "is_ego", False):
                return v
        # Fallback: assume the first ControlledVehicle is the learner ego
        for v in self.road.vehicles:
            if isinstance(v, ControlledVehicle):
                return v
        return None

    def _ego_in_threat_zone(self) -> bool:
        ego = self._ego()
        if ego is None:
            return False
        dx_world = ego.position[0] - self.position[0]
        dy_world = ego.position[1] - self.position[1]
        cos_h, sin_h = np.cos(self.heading), np.sin(self.heading)
        # Ego position in self frame: +x = ahead of self
        dx = cos_h * dx_world + sin_h * dy_world
        dy = -sin_h * dx_world + cos_h * dy_world
        distance = float(np.hypot(dx, dy))
        if distance > self.PROXIMITY_THRESHOLD:
            return False
        if dx <= 0:  # ego is behind us — braking would be useless
            return False
        if abs(dy) > self.LATERAL_BAND:
            return False
        # Closing along self heading: positive if self is faster than ego in self direction
        v_self = self.speed
        v_ego_along = ego.speed * np.cos(ego.heading - self.heading)
        return (v_self - v_ego_along) >= 0.0

    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None) -> float:
        # dt is policy-frequency-dependent; fall back to 1/policy_frequency if available
        dt = 1.0 / getattr(self.road, "_policy_frequency", 1.0) if self.road else 1.0
        if self._brake_remaining > 0.0:
            self._brake_remaining = max(0.0, self._brake_remaining - dt)
            if self._brake_remaining == 0.0:
                self._cooldown_remaining = self.COOLDOWN
            return self.BRAKE_ACCEL
        if self._cooldown_remaining > 0.0:
            self._cooldown_remaining = max(0.0, self._cooldown_remaining - dt)
            return super().acceleration(ego_vehicle, front_vehicle, rear_vehicle)
        if self._ego_in_threat_zone():
            self._brake_remaining = float(self._brake_duration_default)
            return self.BRAKE_ACCEL
        return super().acceleration(ego_vehicle, front_vehicle, rear_vehicle)


# ---------------------------------------------------------------------------
# Archetype 3 — LaneDrifter
# ---------------------------------------------------------------------------


class LaneDrifterVehicle(IDMVehicle):
    """Frequent lane changes with relaxed MOBIL safety.

    Behaviour delta vs nominal IDM:
      - LANE_CHANGE_DELAY shortened (more frequent attempts).
      - mobil() bypasses the 'unsafe braking on new follower' check, so the
        drifter cuts into small gaps. The route-direction check is preserved.
      - LANE_CHANGE_MIN_ACC_GAIN lowered to 0.0.
    """

    ARCHETYPE = "lane_drifter"

    LANE_CHANGE_DELAY = 0.4  # [s], default IDM is ~1.0
    LANE_CHANGE_MIN_ACC_GAIN = 0.0
    POLITENESS = 0.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mark_adversarial(self, self.ARCHETYPE)

    def mobil(self, lane_index) -> bool:
        # Reproduce IDM.mobil() but skip the new-follower-safety floor.
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)
        old_preceding, old_following = self.road.neighbour_vehicles(self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)

        if self.route and self.route[0][2] is not None:
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(
                self.route[0][2] - self.target_lane_index[2]
            ):
                return False
            # Keep the *self* unsafe-braking check so we don't suicide
            if self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False
        else:
            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            jerk = self_pred_a - self_a
            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False
        return True


# ---------------------------------------------------------------------------
# Archetype 4 — ErraticSpeed
# ---------------------------------------------------------------------------


class ErraticSpeedVehicle(IDMVehicle):
    """Target speed toggles between low and high modes on a slow timer.

    Inherits IDM following so rear-end collisions among ado vehicles are
    avoided. The lane speed limit is consulted to keep the low mode within
    reasonable bounds.
    """

    ARCHETYPE = "erratic_speed"

    SWITCH_INTERVAL_MIN = 2.0  # [s]
    SWITCH_INTERVAL_MAX = 5.0  # [s]
    LOW_OFFSET_MIN = -17.0  # [m/s] relative to lane speed limit
    LOW_OFFSET_MAX = -11.0
    HIGH_BAND_MIN = 30.0
    HIGH_BAND_MAX = 36.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mark_adversarial(self, self.ARCHETYPE)
        self._next_switch_time = 0.0
        self._sim_time = 0.0

    def _randomize_archetype(self, rng: np.random.Generator) -> None:
        self._next_switch_time = float(rng.uniform(0.0, self.SWITCH_INTERVAL_MAX))
        self.target_speed = self._sample_target(rng)

    def _sample_target(self, rng: np.random.Generator) -> float:
        speed_limit = getattr(getattr(self, "lane", None), "speed_limit", 25.0) or 25.0
        if rng.random() < 0.5:
            return float(np.clip(
                speed_limit + rng.uniform(self.LOW_OFFSET_MIN, self.LOW_OFFSET_MAX),
                3.0,
                speed_limit,
            ))
        return float(rng.uniform(self.HIGH_BAND_MIN, self.HIGH_BAND_MAX))

    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None) -> float:
        dt = 1.0 / getattr(self.road, "_policy_frequency", 1.0) if self.road else 1.0
        self._sim_time += dt
        if self._sim_time >= self._next_switch_time:
            rng = self.road.np_random if self.road is not None else np.random.default_rng()
            self.target_speed = self._sample_target(rng)
            self._next_switch_time = self._sim_time + float(
                rng.uniform(self.SWITCH_INTERVAL_MIN, self.SWITCH_INTERVAL_MAX)
            )
        return super().acceleration(ego_vehicle, front_vehicle, rear_vehicle)


# ---------------------------------------------------------------------------
# Archetype dispatch
# ---------------------------------------------------------------------------


ARCHETYPE_REGISTRY = {
    "tailgater": TailgaterVehicle,
    "sudden_braker": SuddenBrakerVehicle,
    "lane_drifter": LaneDrifterVehicle,
    "erratic_speed": ErraticSpeedVehicle,
}

DEFAULT_ARCHETYPE_WEIGHTS = {
    "tailgater": 0.40,
    "sudden_braker": 0.20,
    "lane_drifter": 0.20,
    "erratic_speed": 0.20,
}


def _convert_to_archetype(
    vehicle: IDMVehicle,
    archetype: str,
    rng: np.random.Generator,
) -> None:
    """In-place class swap of an IDMVehicle to the requested archetype."""
    cls = ARCHETYPE_REGISTRY[archetype]
    vehicle.__class__ = cls
    mark_adversarial(vehicle, archetype)
    # Initialise per-archetype mutable state that __init__ would have set.
    if archetype == "sudden_braker":
        vehicle._brake_remaining = 0.0
        vehicle._cooldown_remaining = 0.0
        vehicle._brake_duration_default = 1.5
    elif archetype == "erratic_speed":
        vehicle._next_switch_time = 0.0
        vehicle._sim_time = 0.0
    randomize_archetype(vehicle, rng)


def _normalize_weights(weights: dict) -> tuple[list[str], np.ndarray]:
    keys = list(weights.keys())
    arr = np.array([float(weights[k]) for k in keys], dtype=np.float64)
    arr = np.clip(arr, 0.0, None)
    if arr.sum() <= 0:
        raise ValueError(f"archetype_weights must sum to >0, got {weights}")
    return keys, arr / arr.sum()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class AdversarialHighwayV3Env(AdversarialHighwayV2Env):
    """v2 placement, v3 archetype mixture.

    Reads `archetype_weights` from `self.config` (a dict mapping archetype name
    -> non-negative weight). Defaults to DEFAULT_ARCHETYPE_WEIGHTS.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg["archetype_weights"] = dict(DEFAULT_ARCHETYPE_WEIGHTS)
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

        if not nominal_others:
            return

        ego = self.controlled_vehicles[0]
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
        keys, probs = _normalize_weights(self.config["archetype_weights"])
        for k in picked:
            archetype = str(self.np_random.choice(keys, p=probs))
            _convert_to_archetype(nominal_others[k], archetype, self.np_random)


# ---------------------------------------------------------------------------
# Wrapper that exposes archetype info
# ---------------------------------------------------------------------------


class _ArchetypeInfoMixin:
    """Adds `adversarial_archetypes` and `crashed_with_archetype` to info."""

    def _archetype_labels(self):
        base = self.unwrapped
        ego = base.vehicle
        others = [v for v in base.road.vehicles if v is not ego]
        others.sort(key=lambda v: np.sum((v.position - ego.position) ** 2))
        labels: list[Optional[str]] = []
        for v in others[: self.N_AGENTS]:
            labels.append(getattr(v, "archetype", None) if getattr(v, "is_adversarial", False) else None)
        while len(labels) < self.N_AGENTS:
            labels.append(None)
        return labels

    def _crashed_with_archetype(self) -> Optional[str]:
        base = self.unwrapped
        if not base.vehicle.crashed:
            return None
        for v in base.road.vehicles:
            if v is base.vehicle:
                continue
            if v.crashed and getattr(v, "is_adversarial", False):
                return getattr(v, "archetype", "generic")
        return None


class HorizonExpectedObservedDictObsWrapperV3(
    _ArchetypeInfoMixin, HorizonExpectedObservedDictObsWrapper
):
    """H10 anomaly wrapper with v3 archetype info exposure."""

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["adversarial_archetypes"] = self._archetype_labels()
        info["crashed_with_archetype"] = self._crashed_with_archetype()
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        info["adversarial_archetypes"] = self._archetype_labels()
        return obs, info


class KinHistoryDictObsWrapperV3(_ArchetypeInfoMixin, KinHistoryDictObsWrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["adversarial_archetypes"] = self._archetype_labels()
        info["crashed_with_archetype"] = self._crashed_with_archetype()
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        info["adversarial_archetypes"] = self._archetype_labels()
        return obs, info


class DictObsWrapperV3(_ArchetypeInfoMixin, DictObsWrapper):
    """DictObsWrapper with v3 archetype info exposure (used by ViT-only on v3)."""

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["adversarial_archetypes"] = self._archetype_labels()
        info["crashed_with_archetype"] = self._crashed_with_archetype()
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        info["adversarial_archetypes"] = self._archetype_labels()
        return obs, info


# ---------------------------------------------------------------------------
# Factories + registrations
# ---------------------------------------------------------------------------


def _split_env_kwargs(kwargs: dict) -> tuple[dict, Optional[dict]]:
    """Pull archetype_weights out of env kwargs and put it into env config."""
    weights = kwargs.pop("archetype_weights", None)
    return kwargs, weights


def make_adversarial_highway_v3_h10(**kwargs):
    kwargs, weights = _split_env_kwargs(kwargs)
    env = AdversarialHighwayV3Env(**kwargs)
    if weights is not None:
        env.config["archetype_weights"] = dict(weights)
    return HorizonExpectedObservedDictObsWrapperV3(env, horizon=10)


def make_adversarial_highway_v3_kin(**kwargs):
    kwargs, weights = _split_env_kwargs(kwargs)
    env = AdversarialHighwayV3Env(**kwargs)
    if weights is not None:
        env.config["archetype_weights"] = dict(weights)
    return KinHistoryDictObsWrapperV3(env)


def make_adversarial_highway_v3_dict(**kwargs):
    kwargs, weights = _split_env_kwargs(kwargs)
    env = AdversarialHighwayV3Env(**kwargs)
    if weights is not None:
        env.config["archetype_weights"] = dict(weights)
    return DictObsWrapperV3(env)


gym.register(
    id="adversarial-highway-v3-h10",
    entry_point="driving.adversarial_v3:make_adversarial_highway_v3_h10",
)
gym.register(
    id="adversarial-highway-v3-kin",
    entry_point="driving.adversarial_v3:make_adversarial_highway_v3_kin",
)
gym.register(
    id="adversarial-highway-v3-dict",
    entry_point="driving.adversarial_v3:make_adversarial_highway_v3_dict",
)

"""V3 *interaction* env (suffix ``v3i``) — pack-around-ego spawn with
archetype-specific adversary placement.

Implements the plan in ``HighwayEnvironmentDesign.md`` (and the modified
version negotiated in chat). Key differences from ``TargetSpeedV3HighwayEnv``:

  1. **Pack spawn.** Traffic is placed in a ``pack_window`` longitudinal
     window centred on the ego. Ego is spawned in the middle of that
     window so vehicles exist both ahead of and behind it from step 0.
     In-lane rejection sampling keeps a ``pack_min_gap`` minimum spacing.

  2. **Archetype-specific adversary placement.** With probability
     ``adversary_probability`` (default 0.83 ≈ 5/6, giving ≈17 % nominal
     episodes), exactly one adversary is spawned. Lane and longitudinal
     offset relative to ego are dictated by the archetype:

       - sudden_braker → same lane, 25–35 m ahead
       - lane_drifter  → adjacent lane, 5–15 m ahead
       - tailgater     → same lane, 8–15 m behind
       - rear_ender    → same lane, 30–40 m behind, target_speed = v_ego + Δ
       - erratic_speed → same or adjacent lane, 20–30 m ahead

     The five archetypes are sampled uniformly by default. Archetypes are
     instantiated directly from ``ARCHETYPE_REGISTRY`` (no in-place class
     swap of a nominal — the placement is the whole point).

  3. **Pack-tracking respawn.** Each step, any nominal vehicle that has
     drifted more than ``pack_window/2 + respawn_buffer`` from the ego
     longitudinally is removed and re-spawned at the opposite edge in a
     random lane. Adversaries are *never* re-spawned — the scripted
     scenario is the scenario. Buffer is large enough that respawn
     happens well outside the 27.5 m OccupancyGrid crop, so the H10
     per-slot GRU does not see identity flips of in-frame vehicles.

  4. **Ego spawn-lane bias.** Lanes 0/1/2/3 are weighted [0.1, 0.4, 0.4,
     0.1]; the ego starts in a middle lane 80 % of the time. Doesn't
     prevent mid-episode migration but discourages edge-camping as a
     trivial opening move.

  5. **5-level target_speeds** ``[18, 22, 25, 28, 32]`` (vs default
     ``[20, 25, 30]``). The discrete action space cardinality is
     unchanged (still 5 actions: LANE_LEFT/IDLE/LANE_RIGHT/FASTER/SLOWER),
     so models trained on the 3-level env still load on this env;
     FASTER/SLOWER just step through finer speed bins.

Inherits target-speed Gaussian reward + ``right_lane_reward=0`` from
``TargetSpeedV3HighwayEnv`` unchanged.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

from highway_env import utils

from driving.adversarial_v3 import (
    ARCHETYPE_REGISTRY,
    DictObsWrapperV3,
    HorizonExpectedObservedDictObsWrapperV3,
    KinHistoryDictObsWrapperV3,
    _normalize_weights,
    _split_env_kwargs,
    randomize_archetype,
)
from driving.adversarial_v3_ts import TargetSpeedV3HighwayEnv


# (lane_offset_options, (dx_min, dx_max)) — dx is along-road, +ahead of ego.
# 2026-05-11: lane_drifter and tailgater nudged +5 m further from ego to
# probe whether the high ego_initiated crash share at pack=5 came from
# impossible-to-react initial spacing rather than policy choice.
ARCHETYPE_PLACEMENT: dict[str, tuple[list[int], tuple[float, float]]] = {
    "sudden_braker": ([0], (25.0, 35.0)),
    "lane_drifter":  ([-1, +1], (10.0, 20.0)),
    "tailgater":     ([0], (-20.0, -13.0)),
    "rear_ender":    ([0], (-40.0, -30.0)),
    "erratic_speed": ([0, -1, +1], (20.0, 30.0)),
}

# Adversary cruise speed at spawn (m/s). For rear_ender, target_speed is
# overwritten after construction with ego_speed + Δ; the spawn speed below
# is just the initial velocity.
ARCHETYPE_SPEEDS: dict[str, tuple[float, float]] = {
    "sudden_braker": (22.0, 27.0),
    "lane_drifter":  (22.0, 27.0),
    "tailgater":     (27.0, 32.0),
    "rear_ender":    (32.0, 35.0),
    "erratic_speed": (20.0, 28.0),
}

DEFAULT_INTERACTION_ARCHETYPE_WEIGHTS = {
    "tailgater":     0.20,
    "sudden_braker": 0.20,
    "lane_drifter":  0.20,
    "erratic_speed": 0.20,
    "rear_ender":    0.20,
}

INTERACTION_DEFAULTS = {
    "pack_window": 150.0,
    "pack_n_vehicles": 10,
    "pack_min_gap": 8.0,
    "respawn_buffer": 20.0,
    "adversary_probability": 0.83,
    "ego_spawn_lane_bias": [0.1, 0.4, 0.4, 0.1],
}


class InteractionV3HighwayEnv(TargetSpeedV3HighwayEnv):
    """Pack-around-ego spawn + archetype-specific adversary placement +
    pack-tracking respawn."""

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(INTERACTION_DEFAULTS)
        cfg["vehicles_count"] = INTERACTION_DEFAULTS["pack_n_vehicles"]
        cfg["archetype_weights"] = dict(DEFAULT_INTERACTION_ARCHETYPE_WEIGHTS)
        cfg["action"] = {
            "type": "DiscreteMetaAction",
            "target_speeds": [18.0, 22.0, 25.0, 28.0, 32.0],
        }
        return cfg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _lane_index_for(self, lane_id: int):
        return ("0", "1", int(lane_id))

    def _world_pos(self, lane_id: int, s: float) -> np.ndarray:
        lane = self.road.network.get_lane(self._lane_index_for(lane_id))
        return lane.position(s, 0)

    def _world_heading(self, lane_id: int, s: float) -> float:
        lane = self.road.network.get_lane(self._lane_index_for(lane_id))
        return lane.heading_at(s)

    def _sample_ego_lane(self) -> int:
        n_lanes = int(self.config["lanes_count"])
        bias = self.config.get("ego_spawn_lane_bias")
        if not bias or len(bias) != n_lanes:
            return int(self.np_random.integers(0, n_lanes))
        probs = np.array(bias, dtype=float)
        probs = probs / probs.sum()
        return int(self.np_random.choice(n_lanes, p=probs))

    def _gap_ok(
        self,
        lane_id: int,
        s: float,
        occupied: list[tuple[int, float]],
        min_gap: float,
    ) -> bool:
        for (lid, s_other) in occupied:
            if lid == lane_id and abs(s - s_other) < min_gap:
                return False
        return True

    # ------------------------------------------------------------------
    # Spawn
    # ------------------------------------------------------------------

    def _create_vehicles(self) -> None:
        n_lanes = int(self.config["lanes_count"])
        window = float(self.config["pack_window"])
        n_total = int(self.config["pack_n_vehicles"])
        min_gap = float(self.config["pack_min_gap"])
        other_vehicles_type = utils.class_from_path(
            self.config["other_vehicles_type"]
        )
        ego_speed = 25.0

        # Ego at window centre, biased middle-lane.
        ego_lane = self._sample_ego_lane()
        ego_s = window / 2.0
        ego = self.action_type.vehicle_class(
            self.road,
            self._world_pos(ego_lane, ego_s),
            self._world_heading(ego_lane, ego_s),
            ego_speed,
        )
        self.controlled_vehicles = [ego]
        self.road.vehicles.append(ego)

        occupied: list[tuple[int, float]] = [(ego_lane, ego_s)]

        # Optional adversary.
        p_adv = float(self.config["adversary_probability"])
        if self.np_random.random() < p_adv:
            keys, probs = _normalize_weights(self.config["archetype_weights"])
            name = str(self.np_random.choice(keys, p=probs))
            placed = self._spawn_archetype(name, ego_lane, ego_s, n_lanes,
                                           ego_speed, occupied, min_gap)
            if placed is not None:
                occupied.append(placed)

        # Fill remaining slots with nominals.
        n_nominal = n_total - (len(occupied) - 1)  # subtract ego
        attempts_per_slot = 50
        placed_count = 0
        for _ in range(n_nominal):
            chosen = None
            for _attempt in range(attempts_per_slot):
                lane_id = int(self.np_random.integers(0, n_lanes))
                s = float(self.np_random.uniform(0.0, window))
                if self._gap_ok(lane_id, s, occupied, min_gap):
                    chosen = (lane_id, s)
                    break
            if chosen is None:
                break
            lane_id, s = chosen
            speed = float(self.np_random.uniform(21.0, 27.0))
            v = other_vehicles_type(
                self.road,
                self._world_pos(lane_id, s),
                self._world_heading(lane_id, s),
                speed,
            )
            v.randomize_behavior()
            self.road.vehicles.append(v)
            occupied.append((lane_id, s))
            placed_count += 1

    def _spawn_archetype(
        self,
        name: str,
        ego_lane: int,
        ego_s: float,
        n_lanes: int,
        ego_speed: float,
        occupied: list[tuple[int, float]],
        min_gap: float,
    ) -> tuple[int, float] | None:
        if name not in ARCHETYPE_PLACEMENT:
            return None
        lane_offsets, (xmin, xmax) = ARCHETYPE_PLACEMENT[name]
        valid_offsets = [d for d in lane_offsets if 0 <= ego_lane + d < n_lanes]
        if not valid_offsets:
            valid_offsets = [0]
        d_lane = int(self.np_random.choice(valid_offsets))
        adv_lane = ego_lane + d_lane
        # A few attempts to find a non-overlapping spot (almost always works
        # on the first try since adversary is the second vehicle placed).
        for _ in range(10):
            dx = float(self.np_random.uniform(xmin, xmax))
            adv_s = max(1.0, ego_s + dx)
            if self._gap_ok(adv_lane, adv_s, occupied, min_gap):
                break
        else:
            return None

        cls = ARCHETYPE_REGISTRY[name]
        speed_min, speed_max = ARCHETYPE_SPEEDS[name]
        spawn_speed = float(self.np_random.uniform(speed_min, speed_max))
        v = cls(
            self.road,
            self._world_pos(adv_lane, adv_s),
            self._world_heading(adv_lane, adv_s),
            spawn_speed,
        )
        randomize_archetype(v, self.np_random)
        if name == "rear_ender":
            v.target_speed = ego_speed + float(self.np_random.uniform(6.0, 10.0))
        self.road.vehicles.append(v)
        return (adv_lane, adv_s)

    # ------------------------------------------------------------------
    # Pack-tracking respawn
    # ------------------------------------------------------------------

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if not (terminated or truncated):
            self._respawn_drifted_nominals()
        return obs, reward, terminated, truncated, info

    def _respawn_drifted_nominals(self) -> None:
        ego = self.vehicle
        if ego is None:
            return
        ego_x = float(ego.position[0])
        half_win = float(self.config["pack_window"]) / 2.0
        buffer = float(self.config["respawn_buffer"])
        n_lanes = int(self.config["lanes_count"])
        other_vehicles_type = utils.class_from_path(
            self.config["other_vehicles_type"]
        )
        ahead_thresh = ego_x + half_win + buffer
        behind_thresh = ego_x - half_win - buffer

        new_vehicles = []
        for v in self.road.vehicles:
            if v is ego or getattr(v, "is_adversarial", False):
                new_vehicles.append(v)
                continue
            x = float(v.position[0])
            if x > ahead_thresh:
                new_x = ego_x - half_win
            elif x < behind_thresh:
                new_x = ego_x + half_win
            else:
                new_vehicles.append(v)
                continue
            lane_id = int(self.np_random.integers(0, n_lanes))
            speed = float(self.np_random.uniform(22.0, 27.0))
            try:
                pos = self._world_pos(lane_id, new_x)
                heading = self._world_heading(lane_id, new_x)
            except Exception:
                new_vehicles.append(v)
                continue
            v_new = other_vehicles_type(self.road, pos, heading, speed)
            v_new.randomize_behavior()
            new_vehicles.append(v_new)
        self.road.vehicles = new_vehicles


# ---------------------------------------------------------------------------
# Factories + registrations (mirror v3-ts naming)
# ---------------------------------------------------------------------------


def make_v3i_raw(**kwargs):
    kwargs, weights = _split_env_kwargs(kwargs)
    env = InteractionV3HighwayEnv(**kwargs)
    if weights is not None:
        env.config["archetype_weights"] = dict(weights)
    return env


def make_v3i_dict(**kwargs):
    kwargs, weights = _split_env_kwargs(kwargs)
    env = InteractionV3HighwayEnv(**kwargs)
    if weights is not None:
        env.config["archetype_weights"] = dict(weights)
    return DictObsWrapperV3(env)


def make_v3i_h10(**kwargs):
    kwargs, weights = _split_env_kwargs(kwargs)
    env = InteractionV3HighwayEnv(**kwargs)
    if weights is not None:
        env.config["archetype_weights"] = dict(weights)
    return HorizonExpectedObservedDictObsWrapperV3(env, horizon=10)


def make_v3i_kin(**kwargs):
    kwargs, weights = _split_env_kwargs(kwargs)
    env = InteractionV3HighwayEnv(**kwargs)
    if weights is not None:
        env.config["archetype_weights"] = dict(weights)
    return KinHistoryDictObsWrapperV3(env)


gym.register(
    id="adversarial-highway-v3i-raw",
    entry_point="driving.adversarial_v3i:make_v3i_raw",
)
gym.register(
    id="adversarial-highway-v3i-dict",
    entry_point="driving.adversarial_v3i:make_v3i_dict",
)
gym.register(
    id="adversarial-highway-v3i-h10",
    entry_point="driving.adversarial_v3i:make_v3i_h10",
)
gym.register(
    id="adversarial-highway-v3i-kin",
    entry_point="driving.adversarial_v3i:make_v3i_kin",
)

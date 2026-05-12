"""V3 traffic with target-speed reward, N_FORCED_NEAR=1, lane-biased
adversarial spawning, and right_lane_reward disabled (the 'ts' variant).

Same V3 archetype mixture (tailgater/sudden_braker/lane_drifter/erratic_speed),
same DiscreteMetaAction, same observation pipeline as the original V3 envs —
but four changes from AnomalyInputDesign.md "Open environment-design
questions":

  1. ``N_FORCED_NEAR = 1`` (vs the V3 default of 3): the proximity-pick now
     guarantees only one adversary in the ego's nearest-8 pool, dropping
     local adversarial density from ~37% to ~12%. Should let the policy
     learn to interact with traffic instead of treating proximity as
     intrinsically dangerous.

  2. Adds a ``target_speed_reward`` Gaussian to ``_rewards``, weight 0.5,
     centred at 25 m/s (sigma 5 m/s). Penalises both crawling and
     overshooting; complements highway-env's existing ``high_speed_reward``
     (linear in speed) by giving a peaked target.

  3. ``right_lane_reward = 0`` — the highway-env default of 0.1 rewarded
     camping in the rightmost lane regardless of safety. Removed because
     observed policies were hiding in lane 0 or lane 3 to collect this
     bonus + minimise lateral threat surface, which defeats the goal
     of learning to safely engage with adversarial agents.

  4. ``LANE_BIAS_FRACTION = 0.8`` — 80% of adversarial agents are spawned
     in the ego's lane or adjacent lanes (lane_diff ≤ 1); the remaining
     20% come from the rest of the road. Combined with (3), this forces
     the policy to handle adversaries directly rather than evade them.
     The split preserves enough non-adjacent traffic that the env still
     feels naturalistic.

Three env factories matching the existing V3 ones:
  - ``adversarial-highway-v3ts-raw``  (no Dict wrapper — plain OccupancyGrid)
  - ``adversarial-highway-v3ts-dict`` (DictObsWrapperV3, for ViT-only)
  - ``adversarial-highway-v3ts-h10``  (H10 anomaly wrapper, for AnomAttn variants)
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

from highway_env import utils
from highway_env.vehicle.kinematics import Vehicle

from driving.adversarial_v3 import (
    AdversarialHighwayV3Env,
    DictObsWrapperV3,
    HorizonExpectedObservedDictObsWrapperV3,
    KinHistoryDictObsWrapperV3,
    _convert_to_archetype,
    _normalize_weights,
    _proximity_pick,
    _split_env_kwargs,
)


TARGET_SPEED = 25.0  # [m/s], peak of the Gaussian
SPEED_BAND_SIGMA = 5.0  # [m/s], Gaussian width


def _lane_idx(v) -> int:
    """Extract integer lane index from a vehicle's lane_index tuple
    (from_node, to_node, lane_id). Robust to malformed values."""
    li = getattr(v, "lane_index", None)
    if li is None or len(li) < 3 or li[2] is None:
        return -1
    return int(li[2])


class TargetSpeedV3HighwayEnv(AdversarialHighwayV3Env):
    """V3 traffic with N_FORCED_NEAR=1, target-speed reward,
    no right-lane bonus, and lane-biased adversarial spawning."""

    N_FORCED_NEAR = 1
    LANE_BIAS_FRACTION = 0.8  # fraction of adversaries in ego_lane ± 1

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "right_lane_reward": 0.0,
            "target_speed": TARGET_SPEED,
            "target_speed_sigma": SPEED_BAND_SIGMA,
            "target_speed_reward": 0.5,
        })
        return cfg

    def _rewards(self, action) -> dict:
        rewards = super()._rewards(action)
        forward_speed = float(self.vehicle.speed * np.cos(self.vehicle.heading))
        target = self.config.get("target_speed", TARGET_SPEED)
        sigma = self.config.get("target_speed_sigma", SPEED_BAND_SIGMA)
        rewards["target_speed_reward"] = float(
            np.exp(-((forward_speed - target) ** 2) / (2.0 * sigma ** 2))
        )
        return rewards

    def _create_vehicles(self) -> None:
        """Same as V3 _create_vehicles, but partitions the nominal pool
        into ``ego_lane ± 1`` ("near") and the rest ("far"), then assigns
        ``LANE_BIAS_FRACTION`` of adversaries from near and the remainder
        from far. The near subset still uses the proximity guarantee
        (``N_FORCED_NEAR`` in the ``NEAR_POOL_SIZE`` nearest)."""
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
        ego_lane = _lane_idx(ego)
        n_total = len(nominal_others)
        n_adv = int(round(n_total * self.ADVERSARIAL_RATIO))
        if n_adv <= 0:
            return

        # Partition by lane proximity to ego.
        near_idx = [i for i, v in enumerate(nominal_others)
                    if _lane_idx(v) >= 0 and abs(_lane_idx(v) - ego_lane) <= 1]
        far_idx = [i for i in range(n_total) if i not in set(near_idx)]

        # Target counts (clip to pool sizes).
        n_near = min(int(round(self.LANE_BIAS_FRACTION * n_adv)), len(near_idx))
        n_far = min(n_adv - n_near, len(far_idx))
        # If far pool is too small, top up from near.
        if n_near + n_far < n_adv:
            extra = min(n_adv - n_near - n_far, len(near_idx) - n_near)
            n_near += extra

        # Near picks: use proximity guarantee (N_FORCED_NEAR in nearest pool).
        near_picks = np.array([], dtype=int)
        if n_near > 0 and near_idx:
            near_dists = np.array([
                np.linalg.norm(np.asarray(nominal_others[i].position)
                               - np.asarray(ego.position))
                for i in near_idx
            ])
            local_picks = _proximity_pick(
                near_dists,
                n_near,
                self.N_FORCED_NEAR,
                min(self.NEAR_POOL_SIZE, len(near_idx)),
                self.np_random,
            )
            near_picks = np.array([near_idx[i] for i in local_picks], dtype=int)

        # Far picks: uniform random from far pool (no proximity bias).
        far_picks = np.array([], dtype=int)
        if n_far > 0 and far_idx:
            far_picks = self.np_random.choice(
                far_idx, size=n_far, replace=False
            ).astype(int)

        picked = np.concatenate([near_picks, far_picks])
        keys, probs = _normalize_weights(self.config["archetype_weights"])
        for k in picked:
            archetype = str(self.np_random.choice(keys, p=probs))
            _convert_to_archetype(nominal_others[int(k)], archetype, self.np_random)


def make_v3ts_raw(**kwargs):
    """Plain V3-ts env — no observation wrapper. For baseline PPO + MlpPolicy."""
    kwargs, weights = _split_env_kwargs(kwargs)
    env = TargetSpeedV3HighwayEnv(**kwargs)
    if weights is not None:
        env.config["archetype_weights"] = dict(weights)
    return env


def make_v3ts_dict(**kwargs):
    """V3-ts env wrapped with DictObsWrapperV3. For ViT-only experiments."""
    kwargs, weights = _split_env_kwargs(kwargs)
    env = TargetSpeedV3HighwayEnv(**kwargs)
    if weights is not None:
        env.config["archetype_weights"] = dict(weights)
    return DictObsWrapperV3(env)


def make_v3ts_h10(**kwargs):
    """V3-ts env wrapped with HorizonExpectedObservedDictObsWrapperV3.
    For AnomAttn experiments."""
    kwargs, weights = _split_env_kwargs(kwargs)
    env = TargetSpeedV3HighwayEnv(**kwargs)
    if weights is not None:
        env.config["archetype_weights"] = dict(weights)
    return HorizonExpectedObservedDictObsWrapperV3(env, horizon=10)


def make_v3ts_kin(**kwargs):
    """V3-ts env wrapped with KinHistoryDictObsWrapperV3 (no anomaly features)."""
    kwargs, weights = _split_env_kwargs(kwargs)
    env = TargetSpeedV3HighwayEnv(**kwargs)
    if weights is not None:
        env.config["archetype_weights"] = dict(weights)
    return KinHistoryDictObsWrapperV3(env)


gym.register(
    id="adversarial-highway-v3ts-raw",
    entry_point="driving.adversarial_v3_ts:make_v3ts_raw",
)
gym.register(
    id="adversarial-highway-v3ts-dict",
    entry_point="driving.adversarial_v3_ts:make_v3ts_dict",
)
gym.register(
    id="adversarial-highway-v3ts-h10",
    entry_point="driving.adversarial_v3_ts:make_v3ts_h10",
)
gym.register(
    id="adversarial-highway-v3ts-kin",
    entry_point="driving.adversarial_v3_ts:make_v3ts_kin",
)

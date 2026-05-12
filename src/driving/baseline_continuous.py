"""Continuous-action V3 baseline with target-speed reward.

A clean baseline for studying the joint effect of:
  - ContinuousAction (throttle, steering) instead of DiscreteMetaAction.
  - An explicit target-speed Gaussian reward, so the policy is
    discouraged from crawling and from overshooting.
on V3 adversarial traffic (4 original archetypes, no cut_in).

No ViT, no anomaly input/reward, no Dict observation wrapper -
just the raw OccupancyGrid + plain MlpPolicy PPO. Pairs with
experiments/060_baseline_continuous_v3/.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

from driving.adversarial_v3 import AdversarialHighwayV3Env


# Defaults centered on the middle of HighwayEnv's reward_speed_range = [20, 30].
# Sigma=5 gives a soft band: ~0.61 at 20 m/s and 30 m/s, ~0.13 at 15 m/s.
TARGET_SPEED = 25.0  # [m/s]
SPEED_BAND_SIGMA = 5.0  # [m/s]


class BaselineContinuousV3HighwayEnv(AdversarialHighwayV3Env):
    """V3 traffic + ContinuousAction + target-speed Gaussian reward."""

    # Force exactly 1 adversary into the ego's nearest-8 pool (V2/V3
    # default is 3). Reduces local adversarial density so the policy
    # can learn to interact with traffic instead of treating proximity
    # itself as the threat. See AnomalyInputDesign.md "Open environment-
    # design questions" #3.
    N_FORCED_NEAR = 1

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
            },
            "target_speed": TARGET_SPEED,
            "target_speed_sigma": SPEED_BAND_SIGMA,
            # Weight comparable to high_speed_reward (0.4 default) so the
            # peak adds ~0.5 of reward per step at the target speed.
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


def make_baseline_continuous_v3_highway(**kwargs):
    return BaselineContinuousV3HighwayEnv(**kwargs)


gym.register(
    id="baseline-continuous-v3-highway",
    entry_point="driving.baseline_continuous:make_baseline_continuous_v3_highway",
)

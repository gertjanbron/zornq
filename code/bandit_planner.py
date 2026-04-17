#!/usr/bin/env python3
"""
bandit_planner.py - B151 Bandit planner + warm-start cache

Small UCB-style portfolio selector for ambiguous dispatcher families.
Keeps per-family arm statistics and remembers the best assignment seen
for reuse as a warm start on similar follow-up instances.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ArmStats:
    pulls: int = 0
    reward_sum: float = 0.0

    @property
    def mean_reward(self) -> float:
        if self.pulls <= 0:
            return 0.0
        return self.reward_sum / self.pulls


class DispatcherBandit:
    """Per-family UCB portfolio with a tiny warm-start memory."""

    def __init__(self, seed: int = 42, ucb_c: float = 0.8,
                 eps: float = 0.05) -> None:
        self.rng = random.Random(seed)
        self.ucb_c = float(ucb_c)
        self.eps = float(eps)
        self.stats: Dict[str, Dict[str, ArmStats]] = {}
        self.warm_starts: Dict[str, Dict[str, Any]] = {}

    def _family_stats(self, family_key: str,
                      arms: List[str]) -> Dict[str, ArmStats]:
        family = self.stats.setdefault(family_key, {})
        for arm in arms:
            family.setdefault(arm, ArmStats())
        return family

    def choose(self, family_key: str, arms: List[str]) -> str:
        if not arms:
            raise ValueError("Bandit requires at least one arm")
        family = self._family_stats(family_key, list(arms))
        if self.rng.random() < self.eps:
            return self.rng.choice(list(arms))
        for arm in arms:
            if family[arm].pulls == 0:
                return arm
        total_pulls = sum(max(1, family[a].pulls) for a in arms)
        logt = math.log(max(2, total_pulls))
        best_arm = arms[0]
        best_score = -1e100
        for arm in arms:
            stats = family[arm]
            bonus = self.ucb_c * math.sqrt(logt / max(1, stats.pulls))
            score = stats.mean_reward + bonus
            if score > best_score:
                best_score = score
                best_arm = arm
        return best_arm

    def update(self, family_key: str, arm: str, reward: float) -> None:
        family = self._family_stats(family_key, [arm])
        stats = family[arm]
        stats.pulls += 1
        stats.reward_sum += float(reward)

    def get_stats(self, family_key: str) -> Dict[str, Dict[str, float]]:
        family = self.stats.get(family_key, {})
        return {
            arm: {
                'pulls': stats.pulls,
                'reward_sum': stats.reward_sum,
                'mean_reward': stats.mean_reward,
            }
            for arm, stats in family.items()
        }

    def remember_result(self, family_key: str, result: Any) -> bool:
        assignment = getattr(result, 'assignment', None)
        if not assignment:
            return False
        best_cut = float(getattr(result, 'best_cut', 0.0))
        prev = self.warm_starts.get(family_key)
        if prev is not None and best_cut < float(prev.get('best_cut', -1e100)):
            return False
        self.warm_starts[family_key] = {
            'assignment': dict(assignment),
            'best_cut': best_cut,
            'strategy': getattr(result, 'strategy', ''),
            'ratio': float(getattr(result, 'ratio', 0.0) or 0.0),
        }
        return True

    def get_warm_start(self, family_key: str) -> Optional[Dict[str, Any]]:
        cached = self.warm_starts.get(family_key)
        if cached is None:
            return None
        return {
            'assignment': dict(cached.get('assignment', {})),
            'best_cut': float(cached.get('best_cut', 0.0)),
            'strategy': cached.get('strategy', ''),
            'ratio': float(cached.get('ratio', 0.0)),
        }


def family_key_from_info(info: Dict[str, Any]) -> str:
    """Compact feature bucket for portfolio sharing across similar graphs."""
    n = int(info.get('n_nodes', 0))
    density = float(info.get('density', 0.0))
    avg_degree = float(info.get('avg_degree', 0.0))
    max_degree = int(info.get('max_degree', 0))
    cycle_rank = int(info.get('cycle_rank', 0))
    leaf_fraction = float(info.get('leaf_fraction', 0.0))
    hub_fraction = float(info.get('hub_fraction', 0.0))

    if n <= 500:
        size_bucket = 'small'
    elif n <= 2000:
        size_bucket = 'medium'
    elif n <= 5000:
        size_bucket = 'large'
    else:
        size_bucket = 'xlarge'

    if density < 0.05:
        density_bucket = 'sparse'
    elif density < 0.20:
        density_bucket = 'mid'
    else:
        density_bucket = 'dense'

    if max_degree <= 4:
        degree_bucket = 'lowdeg'
    elif max_degree <= 12:
        degree_bucket = 'middeg'
    else:
        degree_bucket = 'highdeg'

    if cycle_rank <= 1:
        cycle_bucket = 'treeish'
    elif cycle_rank <= max(4, n // 20):
        cycle_bucket = 'loopy'
    else:
        cycle_bucket = 'meshy'

    regular_flag = 'regular' if info.get('is_regular') else 'irregular'
    bip_flag = 'bip' if info.get('is_bipartite') else 'nonbip'
    leaf_flag = 'leafy' if leaf_fraction >= 0.10 else 'nonleafy'
    hub_flag = 'hubby' if hub_fraction >= 0.05 or avg_degree >= 10 else 'nonhubby'
    grid_flag = 'grid' if info.get('is_grid') else 'nongrid'

    return '|'.join([
        size_bucket, density_bucket, degree_bucket, cycle_bucket,
        regular_flag, bip_flag, leaf_flag, hub_flag, grid_flag,
    ])


def reward_from_dispatch_result(result: Any) -> float:
    """Quality-first reward with a small time/certification adjustment."""
    ratio = float(getattr(result, 'ratio', 0.0) or 0.0)
    time_s = float(getattr(result, 'time_s', 0.0) or 0.0)
    cert = str(getattr(result, 'certificate', '') or '')
    reward = ratio
    if cert == 'EXACT':
        reward += 0.10
    elif cert == 'NEAR_EXACT':
        reward += 0.04
    reward -= min(0.05, 0.002 * max(0.0, time_s))
    return reward

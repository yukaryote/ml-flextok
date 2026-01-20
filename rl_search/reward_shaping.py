"""
Reward shaping utilities for FlexTok token search.

Provides various reward transformation functions to improve RL training.
"""
import numpy as np
from typing import Optional


class RewardShaper:
    """
    Flexible reward shaping for similarity scores.

    Transforms raw similarity scores (lower = better) into shaped rewards
    that are more suitable for RL training.
    """

    def __init__(
        self,
        mode: str = 'negative',
        previous_score: Optional[float] = None,
        scale: float = 1.0,
        offset: float = 0.0,
    ):
        """
        Initialize reward shaper.

        Args:
            mode: Reward shaping mode. Options:
                - 'negative': reward = -score (simple negation)
                - 'inverse': reward = 1 / (1 + score) (positive, bounded to [0, 1])
                - 'exponential': reward = exp(-score) (positive, bounded to [0, 1])
                - 'improvement': reward = previous_score - current_score (improvement-based)
                - 'normalized_improvement': normalized improvement with moving average
            previous_score: Previous score for improvement-based rewards
            scale: Scale factor for reward
            offset: Offset to add to reward
        """
        self.mode = mode
        self.previous_score = previous_score
        self.scale = scale
        self.offset = offset

        # For normalized improvement
        self.score_history = []
        self.score_mean = 0.0
        self.score_std = 1.0

    def __call__(self, score: float) -> float:
        """
        Transform similarity score to reward.

        Args:
            score: Similarity score (lower = better)

        Returns:
            Shaped reward
        """
        if self.mode == 'negative':
            reward = -score

        elif self.mode == 'inverse':
            reward = 1.0 / (1.0 + score)

        elif self.mode == 'exponential':
            reward = np.exp(-score)

        elif self.mode == 'improvement':
            if self.previous_score is None:
                # First step: use negative score
                reward = -score
            else:
                # Reward based on improvement
                reward = self.previous_score - score
            self.previous_score = score

        elif self.mode == 'normalized_improvement':
            # Track score statistics
            self.score_history.append(score)
            if len(self.score_history) > 100:
                self.score_history.pop(0)

            # Update statistics
            if len(self.score_history) > 1:
                self.score_mean = np.mean(self.score_history)
                self.score_std = np.std(self.score_history) + 1e-8

            # Compute normalized improvement
            if self.previous_score is None:
                reward = -(score - self.score_mean) / self.score_std
            else:
                improvement = self.previous_score - score
                reward = improvement / self.score_std

            self.previous_score = score

        else:
            raise ValueError(f"Unknown reward mode: {self.mode}")

        # Apply scale and offset
        reward = reward * self.scale + self.offset

        return float(reward)

    def reset(self):
        """Reset internal state."""
        self.previous_score = None
        self.score_history = []
        self.score_mean = 0.0
        self.score_std = 1.0


class AdaptiveRewardShaper(RewardShaper):
    """
    Adaptive reward shaper that adjusts based on performance.

    Automatically switches between different reward modes based on
    the agent's progress.
    """

    def __init__(
        self,
        initial_mode: str = 'improvement',
        exploration_mode: str = 'inverse',
        exploitation_mode: str = 'negative',
        switch_threshold: float = 0.3,
    ):
        """
        Initialize adaptive reward shaper.

        Args:
            initial_mode: Initial reward mode
            exploration_mode: Mode to use during exploration (high scores)
            exploitation_mode: Mode to use during exploitation (low scores)
            switch_threshold: Threshold to switch from exploration to exploitation
        """
        super().__init__(mode=initial_mode)
        self.exploration_mode = exploration_mode
        self.exploitation_mode = exploitation_mode
        self.switch_threshold = switch_threshold
        self.best_score = float('inf')

    def __call__(self, score: float) -> float:
        """Transform score to reward with adaptive mode."""
        # Update best score
        if score < self.best_score:
            self.best_score = score

        # Switch mode based on performance
        if self.best_score < self.switch_threshold:
            self.mode = self.exploitation_mode
        else:
            self.mode = self.exploration_mode

        # Use parent class call
        return super().__call__(score)


class MultiObjectiveRewardShaper:
    """
    Combines multiple objectives into a single reward signal.

    Example objectives:
    - Similarity to target (primary)
    - Diversity from previous attempts
    - Computational efficiency
    """

    def __init__(
        self,
        similarity_weight: float = 1.0,
        diversity_weight: float = 0.1,
        efficiency_weight: float = 0.05,
        similarity_shaper: Optional[RewardShaper] = None,
    ):
        """
        Initialize multi-objective reward shaper.

        Args:
            similarity_weight: Weight for similarity objective
            diversity_weight: Weight for diversity objective
            efficiency_weight: Weight for efficiency objective
            similarity_shaper: RewardShaper for similarity scores
        """
        self.similarity_weight = similarity_weight
        self.diversity_weight = diversity_weight
        self.efficiency_weight = efficiency_weight
        self.similarity_shaper = similarity_shaper or RewardShaper(mode='improvement')

        # Track history for diversity
        self.token_history = []

    def __call__(
        self,
        score: float,
        token_values: Optional[np.ndarray] = None,
        num_steps: Optional[int] = None,
    ) -> float:
        """
        Compute multi-objective reward.

        Args:
            score: Similarity score
            token_values: Current token values
            num_steps: Number of steps taken so far

        Returns:
            Combined reward
        """
        # Similarity reward
        similarity_reward = self.similarity_shaper(score)
        total_reward = self.similarity_weight * similarity_reward

        # Diversity reward (encourage exploration)
        if token_values is not None and self.diversity_weight > 0:
            diversity_reward = self._compute_diversity(token_values)
            total_reward += self.diversity_weight * diversity_reward

        # Efficiency reward (encourage fewer steps)
        if num_steps is not None and self.efficiency_weight > 0:
            efficiency_reward = -num_steps * 0.01  # Small penalty per step
            total_reward += self.efficiency_weight * efficiency_reward

        return total_reward

    def _compute_diversity(self, token_values: np.ndarray) -> float:
        """Compute diversity reward based on token history."""
        if len(self.token_history) == 0:
            self.token_history.append(token_values)
            return 0.0

        # Compute average distance to previous tokens
        distances = []
        for prev_tokens in self.token_history[-10:]:  # Last 10
            dist = np.linalg.norm(token_values - prev_tokens)
            distances.append(dist)

        diversity = np.mean(distances) if distances else 0.0

        # Add to history
        self.token_history.append(token_values)
        if len(self.token_history) > 100:
            self.token_history.pop(0)

        return diversity

    def reset(self):
        """Reset internal state."""
        self.similarity_shaper.reset()
        self.token_history = []


def create_reward_shaper(config: dict) -> RewardShaper:
    """
    Factory function to create reward shaper from config.

    Args:
        config: Configuration dictionary with keys:
            - type: 'simple', 'adaptive', or 'multi_objective'
            - mode: reward mode (for simple)
            - ... other parameters

    Returns:
        RewardShaper instance
    """
    shaper_type = config.get('type', 'simple')

    if shaper_type == 'simple':
        return RewardShaper(
            mode=config.get('mode', 'improvement'),
            scale=config.get('scale', 1.0),
            offset=config.get('offset', 0.0),
        )

    elif shaper_type == 'adaptive':
        return AdaptiveRewardShaper(
            initial_mode=config.get('initial_mode', 'improvement'),
            exploration_mode=config.get('exploration_mode', 'inverse'),
            exploitation_mode=config.get('exploitation_mode', 'negative'),
            switch_threshold=config.get('switch_threshold', 0.3),
        )

    elif shaper_type == 'multi_objective':
        similarity_config = config.get('similarity_config', {'mode': 'improvement'})
        similarity_shaper = RewardShaper(**similarity_config)

        return MultiObjectiveRewardShaper(
            similarity_weight=config.get('similarity_weight', 1.0),
            diversity_weight=config.get('diversity_weight', 0.1),
            efficiency_weight=config.get('efficiency_weight', 0.05),
            similarity_shaper=similarity_shaper,
        )

    else:
        raise ValueError(f"Unknown reward shaper type: {shaper_type}")

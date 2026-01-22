"""
Wrapper for training RL agents on multiple target images.

This wrapper enables training a general policy that works across different
target images, preventing overfitting to a single target.
"""
import gymnasium as gym
import numpy as np
from typing import Optional, Callable, Any, Dict, Tuple
from PIL import Image
import torch


class MultiTargetWrapper(gym.Wrapper):
    """
    Wrapper that samples a new target image at the start of each episode.

    This allows training a general policy that can work with any target image,
    rather than overfitting to a single target.

    Args:
        env: The FlexTokSearchEnv to wrap
        target_sampler: Function that returns a new target image (PIL.Image)
                       Called at the start of each episode

    Example:
        >>> # Create a sampler that randomly picks from a dataset
        >>> def sample_target():
        ...     idx = np.random.randint(len(dataset))
        ...     tensor = dataset[idx]
        ...     return convert_to_pil(tensor)
        >>>
        >>> env = FlexTokSearchEnv(...)
        >>> env = MultiTargetWrapper(env, target_sampler=sample_target)
    """

    def __init__(
        self,
        env: gym.Env,
        target_sampler: Callable[[], Image.Image],
    ):
        super().__init__(env)
        self.target_sampler = target_sampler

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset with a new target image."""
        # Sample a new target image
        new_target = self.target_sampler()

        # Pass it to the environment via options
        if options is None:
            options = {}
        options['target_image'] = new_target

        return self.env.reset(seed=seed, options=options)


class DatasetTargetSampler:
    """
    Helper class to create target samplers from datasets.

    Args:
        dataset: PyTorch dataset to sample from
        convert_fn: Function to convert dataset item to PIL Image
        split: Optional split to use ('train', 'val', 'test')
        device: Device for tensor operations

    Example:
        >>> from flextok.utils.dataloader import CelebAHQDataset
        >>> from twenty_questions.twenty_questions import convert_images_to_pil
        >>>
        >>> dataset = CelebAHQDataset(root_dir='data/celebahq', split='train')
        >>> sampler = DatasetTargetSampler(
        ...     dataset=dataset,
        ...     convert_fn=lambda x: convert_images_to_pil(x.unsqueeze(0))[0],
        ...     device='cuda'
        ... )
        >>>
        >>> env = MultiTargetWrapper(env, target_sampler=sampler)
    """

    def __init__(
        self,
        dataset,
        convert_fn: Callable[[torch.Tensor], Image.Image],
        device: str = 'cuda',
        seed: Optional[int] = None,
    ):
        self.dataset = dataset
        self.convert_fn = convert_fn
        self.device = device

        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random

    def __call__(self) -> Image.Image:
        """Sample a random target image from the dataset."""
        idx = self.rng.randint(0, len(self.dataset))
        tensor = self.dataset[idx]

        # Move to device if needed
        if hasattr(tensor, 'to'):
            tensor = tensor.to(self.device)

        # Convert to PIL Image
        pil_image = self.convert_fn(tensor)

        return pil_image


class FixedSetTargetSampler:
    """
    Sampler that cycles through a fixed set of target images.

    Useful for training on a specific set of images with balanced coverage.

    Args:
        target_images: List of PIL Images to sample from
        mode: 'random' to sample randomly, 'cycle' to cycle through in order
        seed: Random seed for reproducibility

    Example:
        >>> target_images = [load_image(path) for path in image_paths]
        >>> sampler = FixedSetTargetSampler(
        ...     target_images=target_images,
        ...     mode='random'
        ... )
        >>> env = MultiTargetWrapper(env, target_sampler=sampler)
    """

    def __init__(
        self,
        target_images: list[Image.Image],
        mode: str = 'random',
        seed: Optional[int] = None,
    ):
        self.target_images = target_images
        self.mode = mode
        self.current_idx = 0

        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random

    def __call__(self) -> Image.Image:
        """Sample a target image."""
        if self.mode == 'random':
            idx = self.rng.randint(0, len(self.target_images))
            return self.target_images[idx]
        elif self.mode == 'cycle':
            img = self.target_images[self.current_idx]
            self.current_idx = (self.current_idx + 1) % len(self.target_images)
            return img
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class CurriculumTargetSampler:
    """
    Sampler that implements curriculum learning by gradually increasing difficulty.

    Starts with easier targets (e.g., images with simpler features) and gradually
    introduces more complex ones as training progresses.

    Args:
        easy_targets: List of easier target images
        hard_targets: List of harder target images
        warmup_episodes: Number of episodes to use only easy targets
        transition_episodes: Number of episodes to gradually mix in hard targets

    Example:
        >>> easy_images = [...]  # Simple faces
        >>> hard_images = [...]  # Complex faces with occlusions
        >>> sampler = CurriculumTargetSampler(
        ...     easy_targets=easy_images,
        ...     hard_targets=hard_images,
        ...     warmup_episodes=1000,
        ...     transition_episodes=2000
        ... )
    """

    def __init__(
        self,
        easy_targets: list[Image.Image],
        hard_targets: list[Image.Image],
        warmup_episodes: int = 1000,
        transition_episodes: int = 2000,
        seed: Optional[int] = None,
    ):
        self.easy_targets = easy_targets
        self.hard_targets = hard_targets
        self.warmup_episodes = warmup_episodes
        self.transition_episodes = transition_episodes
        self.episode_count = 0

        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random

    def __call__(self) -> Image.Image:
        """Sample a target image based on curriculum."""
        self.episode_count += 1

        if self.episode_count <= self.warmup_episodes:
            # Only use easy targets
            idx = self.rng.randint(0, len(self.easy_targets))
            return self.easy_targets[idx]
        elif self.episode_count <= self.warmup_episodes + self.transition_episodes:
            # Gradually increase probability of hard targets
            progress = (self.episode_count - self.warmup_episodes) / self.transition_episodes
            if self.rng.random() < progress:
                # Sample from hard targets
                idx = self.rng.randint(0, len(self.hard_targets))
                return self.hard_targets[idx]
            else:
                # Sample from easy targets
                idx = self.rng.randint(0, len(self.easy_targets))
                return self.easy_targets[idx]
        else:
            # Mix of both
            if self.rng.random() < 0.5:
                idx = self.rng.randint(0, len(self.hard_targets))
                return self.hard_targets[idx]
            else:
                idx = self.rng.randint(0, len(self.easy_targets))
                return self.easy_targets[idx]

"""
Gymnasium environment for FlexTok token search.

The agent sequentially picks token values to minimize perceptual
similarity to a secret target image.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any, List, Callable
from PIL import Image
import torchvision.transforms.functional as TF

from flextok.flextok_wrapper import FlexTok
from flextok.utils.misc import get_bf16_context


class FlexTokSearchEnv(gym.Env):
    """
    FlexTok Token Search Environment.

    The agent sequentially selects token values to minimize perceptual
    similarity (e.g., DreamSim score) to a target image.

    Args:
        flextok_model: The FlexTok model to use for decoding
        target_image: PIL Image or tensor representing the secret target
        similarity_fn: Function that computes similarity score between two images
                      Lower scores = more similar. Signature: (img1_pil, img2_pil) -> float
        fsq_levels: List of FSQ quantization levels (e.g., [8, 8, 8, 5, 5, 5])
        max_tokens: Maximum number of tokens to predict
        history_length: Number of past (token, reward) pairs to include in observation
        enable_bf16: Whether to use bf16 for model inference
        device: torch device to use
        goal_threshold: If similarity score drops below this, episode terminates early
        enable_undo: Whether to allow undo actions
        image_obs: Whether to include decoded images in observations
        reward_shaper: Optional callable to transform raw similarity scores to rewards
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        flextok_model: FlexTok,
        target_image: Image.Image,
        similarity_fn: Callable[[Image.Image, Image.Image], float],
        fsq_levels: List[int] = [8],
        max_tokens: int = 256,
        history_length: int = 5,
        enable_bf16: bool = True,
        device: str = 'cuda',
        goal_threshold: float = 0.1,
        enable_undo: bool = True,
        image_obs: bool = False,
        reward_shaper: Optional[Callable[[float], float]] = None,
    ):
        super().__init__()

        self.flextok_model = flextok_model
        self.target_image = target_image
        self.similarity_fn = similarity_fn
        self.fsq_levels = fsq_levels
        self.max_tokens = max_tokens
        self.history_length = history_length
        self.enable_bf16 = enable_bf16
        self.device = device
        self.goal_threshold = goal_threshold
        self.enable_undo = enable_undo
        self.image_obs = image_obs
        self.reward_shaper = reward_shaper

        # Compute total number of tokens per sequence (256 in default FlexTok)
        self.num_token_positions = max_tokens

        # Current state
        self.current_token_idx = 0
        self.token_sequence = None
        self.history = []  # List of (token_value, reward) tuples
        self.best_score = float('inf')
        self.current_image = None

        # Define action space
        # For each token position, we can pick from fsq_levels[i] values
        # For simplicity with single FSQ level, action is discrete
        if len(fsq_levels) == 1:
            # Single quantization level: discrete action space
            # Action: 0 to (fsq_levels[0] - 1) for token value,
            # + 1 action for undo if enabled
            num_actions = fsq_levels[0]
            if enable_undo:
                num_actions += 1  # Add undo action
            self.action_space = spaces.Discrete(num_actions)
            self.undo_action = fsq_levels[0]  # Last action is undo
        else:
            # Multiple quantization levels: multi-discrete or continuous
            # For now, use Box (continuous) space normalized to [-1, 1]
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(len(fsq_levels),),
                dtype=np.float32
            )
            self.undo_action = None  # Not supported yet for multi-dim

        # Define observation space
        # Observation consists of:
        # 1. Current token position (normalized)
        # 2. Past N token choices (one-hot or normalized)
        # 3. Past N rewards
        # 4. (Optional) Past decoded images

        obs_dict = {}

        # Current position
        obs_dict['position'] = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )

        # History of token choices
        if len(fsq_levels) == 1:
            # Discrete: use one-hot encoding
            obs_dict['token_history'] = spaces.Box(
                low=0, high=1,
                shape=(history_length, fsq_levels[0]),
                dtype=np.float32
            )
        else:
            obs_dict['token_history'] = spaces.Box(
                low=-1, high=1,
                shape=(history_length, len(fsq_levels)),
                dtype=np.float32
            )

        # History of rewards
        obs_dict['reward_history'] = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(history_length,),
            dtype=np.float32
        )

        # Optional: decoded image
        if image_obs:
            # Assume 256x256 RGB image
            obs_dict['image'] = spaces.Box(
                low=0, high=255,
                shape=(history_length, 256, 256, 3),
                dtype=np.uint8
            )

        self.observation_space = spaces.Dict(obs_dict)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Update target image if provided in options
        if options is not None and 'target_image' in options:
            self.target_image = options['target_image']

        # Reset state
        self.current_token_idx = 0
        self.token_sequence = torch.zeros(
            self.num_token_positions,
            dtype=torch.long,
            device=self.device
        )
        self.history = []
        self.best_score = float('inf')
        self.current_image = None

        # Get initial observation
        obs = self._get_observation()
        info = {
            'token_idx': self.current_token_idx,
            'best_score': self.best_score,
        }

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (token value or undo)

        Returns:
            observation: Current observation
            reward: Reward for this step
            terminated: Whether episode is done (goal reached or max tokens)
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Handle undo action
        if self.enable_undo and len(self.fsq_levels) == 1:
            if action == self.undo_action:
                return self._undo_step()

        # Convert action to token value
        if len(self.fsq_levels) == 1:
            # Discrete action
            token_value = action
        else:
            # Continuous action: map from [-1, 1] to quantization levels
            token_value = self._continuous_to_token(action)

        # Set token in sequence
        self.token_sequence[self.current_token_idx] = torch.tensor(token_value, device=self.device)

        # Decode current sequence to image
        current_image = self._decode_tokens(self.token_sequence)
        self.current_image = current_image

        # Compute similarity score
        score = self.similarity_fn(self.target_image, current_image)

        # Update best score
        if score < self.best_score:
            self.best_score = score

        # Compute reward
        if self.reward_shaper is not None:
            reward = self.reward_shaper(score)
        else:
            # Default: negative similarity score
            reward = -score

        # Add to history
        self.history.append((int(token_value), reward))
        if len(self.history) > self.history_length:
            self.history.pop(0)

        # Move to next token
        self.current_token_idx += 1

        # Check termination conditions
        terminated = False
        truncated = False

        # Goal reached
        if score < self.goal_threshold:
            terminated = True
            reward += 10.0  # Bonus for reaching goal

        # Max tokens reached
        if self.current_token_idx >= self.num_token_positions:
            terminated = True

        # Get observation
        obs = self._get_observation()

        info = {
            'token_idx': self.current_token_idx,
            'best_score': self.best_score,
            'current_score': score,
            'token_value': int(token_value),
        }

        return obs, reward, terminated, truncated, info

    def _undo_step(self) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Undo the last action."""
        if self.current_token_idx == 0:
            # Can't undo at start
            reward = -1.0  # Penalty for invalid undo
            obs = self._get_observation()
            return obs, reward, False, False, {'invalid_undo': True}

        # Go back one step
        self.current_token_idx -= 1

        # Remove from history
        if self.history:
            self.history.pop()

        # Zero out the token
        self.token_sequence[self.current_token_idx] = 0

        # Get dreamsim reward of current token + small penalty for undo
        # Compute similarity score
        score = self.similarity_fn(self.target_image, self.current_image)

        # Update best score
        if score < self.best_score:
            self.best_score = score

        # Compute reward
        if self.reward_shaper is not None:
            reward = self.reward_shaper(score) - 0.1  # Small penalty for undo
        else:
            # Default: negative similarity score
            reward = -score - 0.1  # Small penalty for undo

        # Decode current sequence
        if self.current_token_idx > 0:
            current_image = self._decode_tokens(self.token_sequence)
            self.current_image = current_image
        else:
            self.current_image = None

        obs = self._get_observation()
        info = {
            'token_idx': self.current_token_idx,
            'best_score': self.best_score,
            'undo': True,
        }

        return obs, reward, False, False, info

    def _decode_tokens(self, tokens: torch.Tensor) -> Image.Image:
        """Decode token sequence to PIL Image."""
        # tokens shape: (num_tokens,)
        # detokenize expects a list of 2D tensors with shape [1, seq_len]
        # where seq_len is the full sequence length

        # Decode with FlexTok
        with torch.no_grad():
            with get_bf16_context(self.enable_bf16):
                # Pass as a list containing a 2D token sequence [1, seq_len]
                decoded = self.flextok_model.detokenize(
                    [tokens.unsqueeze(0)],  # List of tensors with shape [1, seq_len]
                    timesteps=25,
                    guidance_scale=7.5,
                    perform_norm_guidance=True,
                    generator=None,
                    verbose=False,
                )

        # Convert to PIL Image
        # decoded is (1, C, H, W) in [-1, 1]
        decoded = decoded[0]  # (C, H, W)
        decoded = (decoded + 1) / 2  # Normalize to [0, 1]
        decoded = decoded.clamp(0, 1)
        pil_image = TF.to_pil_image(decoded.cpu())

        return pil_image

    def _continuous_to_token(self, action: np.ndarray) -> torch.Tensor:
        """Convert continuous action to quantized token values."""
        # Map from [-1, 1] to [0, level-1] for each dimension
        token_values = []
        for i, level in enumerate(self.fsq_levels):
            # Map [-1, 1] -> [0, level-1]
            val = (action[i] + 1) / 2 * (level - 1)
            val = int(np.clip(np.round(val), 0, level - 1))
            token_values.append(val)
        return torch.tensor(token_values, dtype=torch.long, device=self.device)

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        obs = {}

        # Current position (normalized)
        obs['position'] = np.array([self.current_token_idx / self.num_token_positions],
                                   dtype=np.float32).copy()

        # Token history
        if len(self.fsq_levels) == 1:
            # One-hot encoding for discrete tokens
            token_history = np.zeros((self.history_length, self.fsq_levels[0]), dtype=np.float32)
            for i, (token, _) in enumerate(self.history[-self.history_length:]):
                token_history[i, token] = 1.0
        else:
            # Normalized continuous values
            token_history = np.zeros((self.history_length, len(self.fsq_levels)), dtype=np.float32)
            for i, (token, _) in enumerate(self.history[-self.history_length:]):
                # TODO: handle multi-dimensional tokens
                pass

        obs['token_history'] = token_history.copy()

        # Reward history
        reward_history = np.zeros(self.history_length, dtype=np.float32)
        for i, (_, reward) in enumerate(self.history[-self.history_length:]):
            reward_history[i] = float(reward)
        obs['reward_history'] = reward_history.copy()

        # Optional: current decoded image
        if self.image_obs and self.current_image is not None:
            img_array = np.array(self.current_image)
            obs['image'] = img_array.copy()
        elif self.image_obs:
            # Placeholder blank image
            obs['image'] = np.zeros((256, 256, 3), dtype=np.uint8)

        return obs

    def render(self):
        """Render the current state."""
        if self.current_image is not None:
            return np.array(self.current_image)
        return None

    def close(self):
        """Clean up resources."""
        pass

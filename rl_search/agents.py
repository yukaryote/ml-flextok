"""
RL agent implementations for FlexTok token search.

Provides wrappers around Stable-Baselines3 agents with custom
features and network architectures.
"""
from typing import Optional, Dict, Any, Union, Type, List
import torch
import torch.nn as nn
from stable_baselines3 import SAC, DQN, PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
import numpy as np
from pathlib import Path

# Import advanced architectures
try:
    from rl_search.architectures import AdvancedFeatureExtractor
    ADVANCED_ARCHITECTURES_AVAILABLE = True
except ImportError:
    ADVANCED_ARCHITECTURES_AVAILABLE = False

# Import WandB callbacks
try:
    import wandb
    from rl_search.wandb_callbacks import WandbCallback, WandbEvalCallback, init_wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class CustomFeatureExtractor(nn.Module):
    """
    Custom feature extractor for FlexTok search observations.

    Processes the dictionary observation space:
    - position: current token position
    - token_history: past token choices
    - reward_history: past rewards
    - image (optional): decoded image
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        use_image: bool = False,
    ):
        super().__init__()
        self.use_image = use_image

        # Get observation space dimensions
        position_dim = observation_space['position'].shape[0]
        token_history_shape = observation_space['token_history'].shape
        reward_history_dim = observation_space['reward_history'].shape[0]

        # Position encoder
        self.position_encoder = nn.Sequential(
            nn.Linear(position_dim, 32),
            nn.ReLU(),
        )

        # Token history encoder
        token_history_dim = np.prod(token_history_shape)
        self.token_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(token_history_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Reward history encoder
        self.reward_encoder = nn.Sequential(
            nn.Linear(reward_history_dim, 64),
            nn.ReLU(),
        )

        # Image encoder (if using images)
        if use_image and 'image' in observation_space.spaces:
            self.image_encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 28 * 28, 256),
                nn.ReLU(),
            )
            combined_dim = 32 + 128 + 64 + 256
        else:
            self.image_encoder = None
            combined_dim = 32 + 128 + 64

        # Combine all features
        self.combiner = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from observations."""
        # Encode each component
        # Note: Stable-Baselines3 already handles tensor conversion,
        # so observations should already be tensors on the correct device
        position_features = self.position_encoder(observations['position'].float())
        token_features = self.token_encoder(observations['token_history'].float())
        reward_features = self.reward_encoder(observations['reward_history'].float())

        # Combine features
        features = [position_features, token_features, reward_features]

        # Add image features if available
        if self.use_image and self.image_encoder is not None:
            if 'image' in observations:
                # Normalize image to [0, 1]
                images = observations['image'].float() / 255.0
                # Permute to (B, C, H, W)
                images = images.permute(0, 3, 1, 2)
                image_features = self.image_encoder(images)
                features.append(image_features)

        combined = torch.cat(features, dim=1)
        output = self.combiner(combined)

        return output


class SACAgent:
    """
    Soft Actor-Critic agent for FlexTok token search.

    SAC is an off-policy algorithm that works well with continuous
    action spaces and includes entropy regularization for exploration.
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        buffer_size: int = 100000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        ent_coef: str = 'auto',
        target_entropy: str = 'auto',
        use_custom_feature_extractor: bool = True,
        features_dim: int = 256,
        use_image: bool = False,
        device: str = 'auto',
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
    ):
        """
        Initialize SAC agent.

        Args:
            env: Gymnasium environment
            learning_rate: Learning rate
            buffer_size: Replay buffer size
            batch_size: Batch size for training
            tau: Soft update coefficient
            gamma: Discount factor
            train_freq: Update frequency
            gradient_steps: Gradient steps per update
            ent_coef: Entropy coefficient ('auto' for automatic tuning)
            target_entropy: Target entropy ('auto' for automatic)
            use_custom_feature_extractor: Whether to use custom feature extractor
            features_dim: Feature dimension for custom extractor
            use_image: Whether to use image observations
            device: Device to use
            verbose: Verbosity level
            tensorboard_log: TensorBoard log directory
        """
        self.env = env
        self.use_custom_feature_extractor = use_custom_feature_extractor

        # Create policy kwargs
        policy_kwargs = {}
        if use_custom_feature_extractor:
            from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

            # Store these in closure for the class
            _features_dim = features_dim
            _use_image = use_image

            class CustomExtractor(BaseFeaturesExtractor):
                def __init__(self, observation_space: gym.spaces.Dict, **kwargs):
                    super().__init__(observation_space, features_dim=_features_dim)
                    self.extractor = CustomFeatureExtractor(
                        observation_space, _features_dim, _use_image
                    )

                def forward(self, observations):
                    return self.extractor(observations)

            policy_kwargs['features_extractor_class'] = CustomExtractor
            policy_kwargs['features_extractor_kwargs'] = {}

        # Create SAC agent
        self.model = SAC(
            'MultiInputPolicy',
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            ent_coef=ent_coef,
            target_entropy=target_entropy,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            tensorboard_log=tensorboard_log,
        )

    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 4,
        callback: Optional[BaseCallback] = None,
        **kwargs
    ):
        """Train the agent."""
        return self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
            callback=callback,
            **kwargs
        )

    def predict(self, observation, deterministic: bool = False):
        """Predict action given observation."""
        return self.model.predict(observation, deterministic=deterministic)

    def save(self, path: str):
        """Save the model."""
        self.model.save(path)

    def load(self, path: str):
        """Load the model."""
        self.model = SAC.load(path, env=self.env)


class DQNAgent:
    """
    Deep Q-Network agent for FlexTok token search.

    DQN works well with discrete action spaces and is sample efficient.
    Best suited for single FSQ level (e.g., [8]).
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 1e-4,
        buffer_size: int = 100000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int = 4,
        gradient_steps: int = 1,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        use_custom_feature_extractor: bool = True,
        features_dim: int = 256,
        use_image: bool = False,
        device: str = 'auto',
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
    ):
        """
        Initialize DQN agent.

        Args:
            env: Gymnasium environment
            learning_rate: Learning rate
            buffer_size: Replay buffer size
            batch_size: Batch size for training
            tau: Soft update coefficient (1.0 for hard updates)
            gamma: Discount factor
            train_freq: Update frequency
            gradient_steps: Gradient steps per update
            target_update_interval: Target network update interval
            exploration_fraction: Fraction of training for epsilon decay
            exploration_initial_eps: Initial epsilon for exploration
            exploration_final_eps: Final epsilon for exploration
            use_custom_feature_extractor: Whether to use custom feature extractor
            features_dim: Feature dimension for custom extractor
            use_image: Whether to use image observations
            device: Device to use
            verbose: Verbosity level
            tensorboard_log: TensorBoard log directory
        """
        self.env = env
        self.use_custom_feature_extractor = use_custom_feature_extractor

        # Create policy kwargs
        policy_kwargs = {}
        if use_custom_feature_extractor:
            from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

            # Store these in closure for the class
            _features_dim = features_dim
            _use_image = use_image

            class CustomExtractor(BaseFeaturesExtractor):
                def __init__(self, observation_space: gym.spaces.Dict, **kwargs):
                    super().__init__(observation_space, features_dim=_features_dim)
                    self.extractor = CustomFeatureExtractor(
                        observation_space, _features_dim, _use_image
                    )

                def forward(self, observations):
                    return self.extractor(observations)

            policy_kwargs['features_extractor_class'] = CustomExtractor
            policy_kwargs['features_extractor_kwargs'] = {}

        # Create DQN agent
        self.model = DQN(
            'MultiInputPolicy',
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            tensorboard_log=tensorboard_log,
        )

    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 4,
        callback: Optional[BaseCallback] = None,
        **kwargs
    ):
        """Train the agent."""
        return self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
            callback=callback,
            **kwargs
        )

    def predict(self, observation, deterministic: bool = False):
        """Predict action given observation."""
        return self.model.predict(observation, deterministic=deterministic)

    def save(self, path: str):
        """Save the model."""
        self.model.save(path)

    def load(self, path: str):
        """Load the model."""
        self.model = DQN.load(path, env=self.env)


class PPOAgent:
    """
    Proximal Policy Optimization agent for FlexTok token search.

    PPO is an on-policy algorithm that is simple and robust, but
    less sample efficient than SAC or DQN.
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_custom_feature_extractor: bool = True,
        features_dim: int = 256,
        use_image: bool = False,
        device: str = 'auto',
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
    ):
        """
        Initialize PPO agent.

        Args:
            env: Gymnasium environment
            learning_rate: Learning rate
            n_steps: Steps per update
            batch_size: Batch size
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm
            use_custom_feature_extractor: Whether to use custom feature extractor
            features_dim: Feature dimension for custom extractor
            use_image: Whether to use image observations
            device: Device to use
            verbose: Verbosity level
            tensorboard_log: TensorBoard log directory
        """
        self.env = env
        self.use_custom_feature_extractor = use_custom_feature_extractor

        # Create policy kwargs
        policy_kwargs = {}
        if use_custom_feature_extractor:
            from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

            # Store these in closure for the class
            _features_dim = features_dim
            _use_image = use_image

            class CustomExtractor(BaseFeaturesExtractor):
                def __init__(self, observation_space: gym.spaces.Dict, **kwargs):
                    super().__init__(observation_space, features_dim=_features_dim)
                    self.extractor = CustomFeatureExtractor(
                        observation_space, _features_dim, _use_image
                    )

                def forward(self, observations):
                    return self.extractor(observations)

            policy_kwargs['features_extractor_class'] = CustomExtractor
            policy_kwargs['features_extractor_kwargs'] = {}

        # Create PPO agent
        self.model = PPO(
            'MultiInputPolicy',
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            tensorboard_log=tensorboard_log,
        )

    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 1,
        callback: Optional[BaseCallback] = None,
        **kwargs
    ):
        """Train the agent."""
        return self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
            callback=callback,
            **kwargs
        )

    def predict(self, observation, deterministic: bool = False):
        """Predict action given observation."""
        return self.model.predict(observation, deterministic=deterministic)

    def save(self, path: str):
        """Save the model."""
        self.model.save(path)

    def load(self, path: str):
        """Load the model."""
        self.model = PPO.load(path, env=self.env)


def create_agent(
    agent_type: str,
    env: gym.Env,
    config: Optional[Dict[str, Any]] = None,
) -> Union[SACAgent, DQNAgent, PPOAgent]:
    """
    Factory function to create RL agent.

    Args:
        agent_type: Type of agent ('sac', 'dqn', or 'ppo')
        env: Gymnasium environment
        config: Configuration dictionary

    Returns:
        RL agent instance
    """
    config = config or {}

    if agent_type.lower() == 'sac':
        return SACAgent(env, **config)
    elif agent_type.lower() == 'dqn':
        return DQNAgent(env, **config)
    elif agent_type.lower() == 'ppo':
        return PPOAgent(env, **config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def create_agent_with_advanced_architecture(
    agent_type: str,
    env: gym.Env,
    architecture: str = 'transformer',
    features_dim: int = 512,
    use_image: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Union[SACAgent, DQNAgent, PPOAgent]:
    """
    Factory function to create RL agent with advanced architectures.

    Args:
        agent_type: Type of agent ('sac', 'dqn', or 'ppo')
        env: Gymnasium environment
        architecture: Architecture type:
            - 'transformer': Transformer for history + CNN + cross-attention (BEST)
            - 'gru': GRU for history + CNN (good balance)
            - 'efficient': Simple MLP + efficient CNN (fastest)
        features_dim: Output feature dimension
        use_image: Whether to use image observations
        config: Additional agent configuration

    Returns:
        RL agent instance with advanced feature extractor

    Example:
        >>> agent = create_agent_with_advanced_architecture(
        ...     agent_type='dqn',
        ...     env=env,
        ...     architecture='transformer',
        ...     use_image=True,
        ...     features_dim=512,
        ... )
    """
    if not ADVANCED_ARCHITECTURES_AVAILABLE:
        raise ImportError(
            "Advanced architectures not available. "
            "Make sure rl_search.architectures is properly installed."
        )

    config = config or {}

    # Override feature extractor settings
    config['use_custom_feature_extractor'] = True
    config['features_dim'] = features_dim
    config['use_image'] = use_image

    # Create custom extractor wrapper
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class AdvancedExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space: gym.spaces.Dict):
            super().__init__(observation_space, features_dim=features_dim)
            self.extractor = AdvancedFeatureExtractor(
                observation_space=observation_space,
                features_dim=features_dim,
                use_image=use_image,
                architecture=architecture,
            )

        def forward(self, observations):
            return self.extractor(observations)

    # Inject custom extractor into config
    if 'policy_kwargs' not in config:
        config['policy_kwargs'] = {}
    config['policy_kwargs']['features_extractor_class'] = AdvancedExtractor
    config['policy_kwargs']['features_extractor_kwargs'] = {}

    # Remove use_custom_feature_extractor flag as we're using policy_kwargs
    config.pop('use_custom_feature_extractor', None)

    # Create agent with advanced architecture
    if agent_type.lower() == 'sac':
        # Create SAC with custom policy_kwargs
        agent = SACAgent.__new__(SACAgent)
        agent.env = env
        agent.use_custom_feature_extractor = True

        # Extract SAC-specific params
        sac_params = {
            'learning_rate': config.get('learning_rate', 3e-4),
            'buffer_size': config.get('buffer_size', 100000),
            'batch_size': config.get('batch_size', 256),
            'tau': config.get('tau', 0.005),
            'gamma': config.get('gamma', 0.99),
            'train_freq': config.get('train_freq', 1),
            'gradient_steps': config.get('gradient_steps', 1),
            'ent_coef': config.get('ent_coef', 'auto'),
            'target_entropy': config.get('target_entropy', 'auto'),
            'verbose': config.get('verbose', 1),
            'device': config.get('device', 'auto'),
            'tensorboard_log': config.get('tensorboard_log', None),
            'policy_kwargs': config['policy_kwargs'],
        }

        agent.model = SAC('MultiInputPolicy', env, **sac_params)
        return agent

    elif agent_type.lower() == 'dqn':
        # Create DQN with custom policy_kwargs
        agent = DQNAgent.__new__(DQNAgent)
        agent.env = env
        agent.use_custom_feature_extractor = True

        dqn_params = {
            'learning_rate': config.get('learning_rate', 1e-4),
            'buffer_size': config.get('buffer_size', 100000),
            'batch_size': config.get('batch_size', 32),
            'tau': config.get('tau', 1.0),
            'gamma': config.get('gamma', 0.99),
            'train_freq': config.get('train_freq', 4),
            'gradient_steps': config.get('gradient_steps', 1),
            'target_update_interval': config.get('target_update_interval', 10000),
            'exploration_fraction': config.get('exploration_fraction', 0.1),
            'exploration_initial_eps': config.get('exploration_initial_eps', 1.0),
            'exploration_final_eps': config.get('exploration_final_eps', 0.05),
            'verbose': config.get('verbose', 1),
            'device': config.get('device', 'auto'),
            'tensorboard_log': config.get('tensorboard_log', None),
            'policy_kwargs': config['policy_kwargs'],
        }

        agent.model = DQN('MultiInputPolicy', env, **dqn_params)
        return agent

    elif agent_type.lower() == 'ppo':
        # Create PPO with custom policy_kwargs
        agent = PPOAgent.__new__(PPOAgent)
        agent.env = env
        agent.use_custom_feature_extractor = True

        ppo_params = {
            'learning_rate': config.get('learning_rate', 3e-4),
            'n_steps': config.get('n_steps', 2048),
            'batch_size': config.get('batch_size', 64),
            'n_epochs': config.get('n_epochs', 10),
            'gamma': config.get('gamma', 0.99),
            'gae_lambda': config.get('gae_lambda', 0.95),
            'clip_range': config.get('clip_range', 0.2),
            'ent_coef': config.get('ent_coef', 0.0),
            'vf_coef': config.get('vf_coef', 0.5),
            'max_grad_norm': config.get('max_grad_norm', 0.5),
            'verbose': config.get('verbose', 1),
            'device': config.get('device', 'auto'),
            'tensorboard_log': config.get('tensorboard_log', None),
            'policy_kwargs': config['policy_kwargs'],
        }

        agent.model = PPO('MultiInputPolicy', env, **ppo_params)
        return agent

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

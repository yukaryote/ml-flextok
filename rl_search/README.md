# RL-based Search for FlexTok Tokens

This module provides reinforcement learning agents for searching over the FlexTok token space to minimize perceptual similarity to a target image.

## Overview

The RL agent sequentially picks token values that minimize perceptual similarity (e.g., DreamSim score) to a secret target image. This is formulated as a sequential decision-making problem:

- **State**: Current token position + history of past choices and rewards
- **Action**: Token value choice (discrete for FSQ [8], continuous for multi-dimensional FSQ)
- **Reward**: Shaped similarity score (various modes available)
- **Transition**: Move to next token position or undo previous choice

## Features

### 1. Gymnasium Environment (`environment.py`)
- Fully compatible with OpenAI Gym/Gymnasium interface
- Supports discrete and continuous action spaces
- Configurable observation space (history, images, rewards)
- Optional undo action for exploration
- Customizable reward shaping

### 2. Reward Shaping (`reward_shaping.py`)
Multiple reward transformation strategies:
- **Negative**: `reward = -score` (simple negation)
- **Inverse**: `reward = 1 / (1 + score)` (positive, bounded)
- **Exponential**: `reward = exp(-score)` (positive, bounded)
- **Improvement**: `reward = previous_score - current_score` (improvement-based)
- **Normalized Improvement**: Normalized with running statistics
- **Adaptive**: Automatically switches between exploration and exploitation modes
- **Multi-Objective**: Combines similarity, diversity, and efficiency objectives

### 3. RL Agents (`agents.py`)
Three agent implementations using Stable-Baselines3:

#### DQN (Recommended for discrete actions)
- Best for single FSQ level (e.g., [8])
- Sample efficient with replay buffer
- Simple and stable

```python
from rl_search.agents import create_agent

agent = create_agent('dqn', env, {
    'learning_rate': 1e-4,
    'buffer_size': 50000,
    'batch_size': 32,
})
```

#### SAC (Recommended for continuous actions)
- Best for multi-dimensional FSQ (e.g., [8, 8, 8, 5, 5, 5])
- Off-policy with entropy regularization
- Great exploration via maximum entropy objective

```python
agent = create_agent('sac', env, {
    'learning_rate': 3e-4,
    'ent_coef': 'auto',
})
```

#### PPO (Alternative)
- On-policy algorithm
- Simple and robust but less sample efficient
- Works for both discrete and continuous

```python
agent = create_agent('ppo', env, {
    'learning_rate': 3e-4,
    'n_steps': 2048,
})
```

### 4. Custom Feature Extractor
Processes dictionary observations:
- Position encoder for current token position
- Token history encoder (MLP for discrete, could use RNN)
- Reward history encoder
- Optional image encoder (CNN) for visual observations

### 5. Training Script (`train.py`)
End-to-end training with:
- Automatic checkpointing
- Evaluation during training
- TensorBoard logging
- Configuration management

```bash
python -m rl_search.train \
    --ckpt_path /path/to/flextok.pt \
    --target_image_path /path/to/target.jpg \
    --agent_type dqn \
    --total_timesteps 100000 \
    --save_dir ./rl_checkpoints
```

### 6. Evaluation Utilities (`eval_utils.py`)
- Episode evaluation with metrics
- Trajectory visualization
- Agent comparison
- Token distribution analysis
- Comprehensive reporting

## Quick Start

### 1. Basic Usage

```python
from rl_search import FlexTokSearchEnv, RewardShaper, create_agent

# Create environment
env = FlexTokSearchEnv(
    flextok_model=model,
    target_image=target_image,
    similarity_fn=similarity_fn,
    fsq_levels=[8],
    max_tokens=256,
)

# Create agent
agent = create_agent('dqn', env)

# Train
agent.learn(total_timesteps=50000)

# Evaluate
obs, info = env.reset()
done = False
while not done:
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

### 2. With Custom Reward Shaping

```python
from rl_search.reward_shaping import create_reward_shaper

reward_config = {
    'type': 'adaptive',
    'exploration_mode': 'inverse',
    'exploitation_mode': 'improvement',
}
reward_shaper = create_reward_shaper(reward_config)

env = FlexTokSearchEnv(
    # ... other args
    reward_shaper=reward_shaper,
)
```

### 3. Complete Example

See `notebooks/rl_search_example.ipynb` for a full interactive example.

## Algorithm Comparison

| Algorithm | Sample Efficiency | Exploration | Action Space | Best For |
|-----------|------------------|-------------|--------------|----------|
| **DQN** | High | ε-greedy | Discrete | FSQ [8] |
| **SAC** | High | Entropy | Continuous | Multi-dim FSQ |
| **PPO** | Medium | Entropy | Both | General purpose |

## Performance Tips

1. **Start with DQN** for discrete actions (FSQ [8])
2. **Use improvement-based rewards** for stable training
3. **Enable custom feature extractor** for better performance
4. **Tune exploration** (epsilon for DQN, entropy for SAC)
5. **Monitor TensorBoard** logs during training
6. **Compare with greedy/beam search** as baselines

## Reward Shaping Advice

### When to use each mode:

- **Negative**: Simple baseline, works well with PPO
- **Inverse**: When you want strictly positive rewards
- **Improvement**: Best for stable training, focuses on progress
- **Adaptive**: When performance varies significantly across episodes
- **Multi-Objective**: When you want to balance similarity, diversity, and efficiency

## FAQ

### Q: Why use RL over greedy/beam search?

RL can:
- Learn from past experiences across multiple images
- Discover non-greedy strategies that lead to better global solutions
- Generalize to new target images with transfer learning
- Handle complex reward functions beyond simple similarity

### Q: Are negative rewards bad for PPO?

No! PPO normalizes advantages during training, so the absolute scale doesn't matter. The relative differences between actions are what matters.

### Q: Which agent should I use?

- **Single FSQ level [8]**: Use DQN (discrete, sample efficient)
- **Multi-dimensional FSQ**: Use SAC (continuous, handles exploration well)
- **Unsure**: Start with DQN for simplicity

### Q: How long does training take?

- DQN: ~50K-100K timesteps for reasonable performance
- SAC: ~100K-200K timesteps (more exploration needed)
- PPO: ~200K-500K timesteps (less sample efficient)

With DreamSim evaluation, expect ~10-30 minutes for 50K timesteps on a GPU.

## Future Enhancements

- [ ] Model-based RL (learn FlexTok dynamics)
- [ ] Multi-task learning (train on multiple images)
- [ ] Curriculum learning (start with easy targets)
- [ ] Hierarchical RL (high-level planning, low-level execution)
- [ ] Imitation learning from beam search trajectories
- [ ] Evolutionary strategies integration

## Citation

If you use this code, please cite the FlexTok paper:

```bibtex
@article{flextok2024,
  title={FlexTok: Flexible Tokenization for Images},
  author={...},
  journal={...},
  year={2024}
}
```

## License

Same as the main FlexTok repository.

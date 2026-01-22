# FlexTok RL Training Configuration

This directory contains Hydra configuration files for training RL agents on FlexTok token search.

## Quick Start

### Basic Training
```bash
# From the ml-flextok root directory
python rl_search/train.py
```

### Using Pre-configured Experiments
```bash
# Quick test run (10k timesteps)
python rl_search/train.py experiment=quick_test

# Full training run (500k timesteps)
python rl_search/train.py experiment=full_training

# Single-target training
python rl_search/train.py experiment=single_target
```

### Changing Agents
```bash
# Train with SAC
python rl_search/train.py agent=sac

# Train with PPO
python rl_search/train.py agent=ppo
```

### Custom Configurations
```bash
# Override specific parameters
python rl_search/train.py \
    training.total_timesteps=200000 \
    agent.config.learning_rate=5e-5 \
    wandb.enabled=true

# Change dataset
python rl_search/train.py \
    dataset.path=/path/to/your/dataset \
    dataset.split=train

# Disable multi-target training
python rl_search/train.py \
    training.multi_target.enabled=false \
    dataset.target_image_path=/path/to/target.jpg
```

## Configuration Structure

```
configs/
├── train.yaml                 # Main configuration file
├── agent/                     # Agent-specific configs
│   ├── dqn.yaml
│   ├── sac.yaml
│   └── ppo.yaml
├── experiment/                # Pre-configured experiments
│   ├── quick_test.yaml
│   ├── full_training.yaml
│   └── single_target.yaml
└── README.md                  # This file
```

## Key Configuration Sections

### Model Configuration
- `model.checkpoint_path`: Path to FlexTok checkpoint
- `model.fsq_levels`: FSQ quantization levels (e.g., [8])

### Dataset Configuration
- `dataset.name`: Dataset name (e.g., "celebahq")
- `dataset.path`: Path to dataset directory
- `dataset.split`: Train/val/test split
- `dataset.target_image_path`: Optional path to specific target image

### Environment Configuration
- `environment.max_tokens`: Maximum tokens per episode (default: 256)
- `environment.history_length`: Number of past (token, reward) pairs in observation
- `environment.goal_threshold`: DreamSim score for early termination
- `environment.enable_undo`: Allow undo actions
- `environment.image_obs`: Include decoded images in observations

### Training Configuration
- `training.total_timesteps`: Total training steps
- `training.checkpoint_freq`: Save checkpoint every N steps
- `training.eval_freq`: Evaluate every N steps
- `training.multi_target.enabled`: Enable multi-target training
- `training.multi_target.sampler_type`: "dataset" or "fixed"

### WandB Configuration
- `wandb.enabled`: Enable WandB logging
- `wandb.project`: WandB project name
- `wandb.log_videos`: Enable video trajectory logging
- `wandb.video_log_freq`: Log video every N episodes
- `wandb.video_fps`: Frame rate for videos

## Logged Metrics

The training script logs the following to WandB:

### Training Metrics
- Episode rewards
- Episode lengths
- Best DreamSim scores per episode
- Final DreamSim scores per episode

### Videos
- Search trajectory videos showing:
  - Decoded image at each step
  - DreamSim score at each step
  - Token value chosen at each step

### Images
- Initial target image
- Final decoded image
- Best decoded image per episode

### Evaluation Metrics
- Mean evaluation reward
- Mean best score
- Mean final score
- Evaluation trajectory videos

## Advanced Usage

### Hydra Multirun (Parameter Sweeps)
```bash
# Sweep over learning rates
python rl_search/train.py -m \
    agent.config.learning_rate=1e-4,5e-5,1e-5

# Sweep over agents and exploration settings
python rl_search/train.py -m \
    agent=dqn,sac \
    agent.config.exploration_fraction=0.1,0.2,0.3
```

### Using Hydra Overrides
```bash
# Override nested parameters
python rl_search/train.py \
    +agent.config.new_param=value \  # Add new parameter
    ~wandb.tags \                     # Delete parameter
    ++agent.config.override=value     # Force override
```

### Configuration Composition
Create your own experiment config:

```yaml
# configs/experiment/my_experiment.yaml
# @package _global_

defaults:
  - /agent: dqn

experiment_name: my_custom_experiment

training:
  total_timesteps: 300000
  multi_target:
    enabled: true
    sampler_type: fixed
    n_fixed_images: 20

agent:
  config:
    learning_rate: 2.0e-4
    features_dim: 768
```

Then run:
```bash
python rl_search/train.py experiment=my_experiment
```

## Output Structure

```
outputs/
└── {experiment_name}/
    └── {date}/
        └── {time}/
            ├── checkpoints/          # Model checkpoints
            ├── wandb_checkpoints/    # WandB-tracked checkpoints
            ├── tensorboard/          # TensorBoard logs
            ├── monitor/              # Episode statistics
            ├── eval_monitor/         # Evaluation statistics
            ├── hydra_config.yaml     # Saved configuration
            └── {agent_type}_final.zip  # Final model
```

## Tips

1. **Quick iteration**: Use `experiment=quick_test` for fast debugging
2. **Multi-target training**: Prevents overfitting to a single target image
3. **Video logging**: Set `video_log_freq` higher for faster training (videos are expensive)
4. **Image observations**: Only enable `environment.image_obs=true` if using image-based policies
5. **WandB offline mode**: Use `wandb.mode=offline` for training without internet

## Requirements

Make sure you have installed:
```bash
pip install hydra-core omegaconf wandb imageio imageio-ffmpeg
```

## Troubleshooting

### WandB not logging
- Check `wandb.enabled=true`
- Verify WandB login: `wandb login`
- Check `wandb.mode=online`

### Out of memory
- Reduce `agent.config.batch_size`
- Reduce `agent.config.buffer_size`
- Set `environment.image_obs=false`

### Slow training
- Reduce `wandb.video_log_freq`
- Disable videos: `wandb.log_videos=false`
- Use smaller `agent.config.features_dim`

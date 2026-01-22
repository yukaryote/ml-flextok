#!/bin/bash
# Example training scripts for FlexTok RL Search

# Make sure we're in the project root
cd "$(dirname "$0")/.." || exit

# ============================================================================
# Quick Test Run (for debugging)
# ============================================================================
echo "Running quick test..."
python rl_search/train.py \
    experiment=quick_test \
    wandb.enabled=true \
    device=cuda

# ============================================================================
# Full Training Run with DQN
# ============================================================================
# echo "Running full DQN training..."
python rl_search/train.py \
    experiment=full_training \
    agent=dqn \
    wandb.enabled=true \
    wandb.project=flextok-rl-search \
    wandb.run_name=dqn_multi_target_v1

# ============================================================================
# Single-Target Training (for overfitting test)
# ============================================================================
# echo "Running single-target training..."
# python rl_search/train.py \
#     experiment=single_target \
#     dataset.target_image_path=/path/to/your/target_image.jpg \
#     wandb.enabled=true

# ============================================================================
# Custom Configuration Example
# ============================================================================
# echo "Running custom training..."
# python rl_search/train.py \
#     experiment_name=custom_run \
#     agent=dqn \
#     training.total_timesteps=200000 \
#     agent.config.learning_rate=5e-5 \
#     agent.config.batch_size=128 \
#     agent.config.features_dim=768 \
#     environment.max_tokens=256 \
#     wandb.enabled=true \
#     wandb.video_log_freq=30 \
#     device=cuda

# ============================================================================
# Hyperparameter Sweep with Hydra Multirun
# ============================================================================
# echo "Running hyperparameter sweep..."
# python rl_search/train.py -m \
#     agent.config.learning_rate=1e-4,5e-5,1e-5 \
#     agent.config.batch_size=32,64,128 \
#     training.total_timesteps=100000 \
#     wandb.enabled=true

# ============================================================================
# Train with SAC Agent
# ============================================================================
# echo "Running SAC training..."
# python rl_search/train.py \
#     agent=sac \
#     experiment_name=sac_multi_target \
#     training.total_timesteps=200000 \
#     wandb.enabled=true

# ============================================================================
# Train with PPO Agent
# ============================================================================
# echo "Running PPO training..."
# python rl_search/train.py \
#     agent=ppo \
#     experiment_name=ppo_multi_target \
#     training.total_timesteps=200000 \
#     wandb.enabled=true

echo "Training complete!"

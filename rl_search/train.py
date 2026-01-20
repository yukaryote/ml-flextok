"""
Training script for FlexTok token search RL agents.

Provides end-to-end training with logging, checkpointing, and evaluation.
"""
import argparse
from pathlib import Path
from typing import Optional, Callable
import yaml
import torch
from PIL import Image

from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from flextok.flextok_wrapper import FlexTok
from twenty_questions.twenty_questions import load_flextok_model, convert_images_to_pil

from rl_search.environment import FlexTokSearchEnv
from rl_search.reward_shaping import create_reward_shaper
from rl_search.agents import create_agent


def create_similarity_fn(similarity_model, preprocess_fn=None):
    """
    Create similarity function wrapper.

    Args:
        similarity_model: The similarity model (e.g., DreamSim)
        preprocess_fn: Optional preprocessing function

    Returns:
        Similarity function: (img1_pil, img2_pil) -> float
    """
    def similarity_fn(img1: Image.Image, img2: Image.Image) -> float:
        """Compute similarity score between two PIL images."""
        # Preprocess if needed
        if preprocess_fn is not None:
            img1_tensor = preprocess_fn(img1).unsqueeze(0)
            img2_tensor = preprocess_fn(img2).unsqueeze(0)
        else:
            # Convert to tensors
            import torchvision.transforms.functional as TF
            img1_tensor = TF.to_tensor(img1).unsqueeze(0)
            img2_tensor = TF.to_tensor(img2).unsqueeze(0)

        # Move to device
        device = next(similarity_model.parameters()).device
        img1_tensor = img1_tensor.to(device)
        img2_tensor = img2_tensor.to(device)

        # Compute similarity
        with torch.no_grad():
            score = similarity_model(img1_tensor, img2_tensor)

        # Convert to float
        if isinstance(score, torch.Tensor):
            score = score.item()

        return float(score)

    return similarity_fn


def train_agent(
    agent_type: str,
    flextok_model: FlexTok,
    target_image: Image.Image,
    similarity_fn: Callable,
    fsq_levels: list = [8],
    max_tokens: int = 256,
    total_timesteps: int = 100000,
    reward_config: Optional[dict] = None,
    agent_config: Optional[dict] = None,
    save_dir: str = './rl_checkpoints',
    experiment_name: str = 'flextok_search',
    eval_freq: int = 10000,
    n_eval_episodes: int = 5,
    checkpoint_freq: int = 10000,
    device: str = 'cuda',
    enable_bf16: bool = True,
    verbose: int = 1,
):
    """
    Train an RL agent for FlexTok token search.

    Args:
        agent_type: Type of agent ('sac', 'dqn', 'ppo')
        flextok_model: FlexTok model
        target_image: Target image (PIL Image)
        similarity_fn: Similarity function
        fsq_levels: FSQ quantization levels
        max_tokens: Maximum number of tokens
        total_timesteps: Total training timesteps
        reward_config: Reward shaping configuration
        agent_config: Agent configuration
        save_dir: Directory to save checkpoints
        experiment_name: Name of experiment
        eval_freq: Evaluation frequency
        n_eval_episodes: Number of evaluation episodes
        checkpoint_freq: Checkpoint save frequency
        device: Device to use
        enable_bf16: Whether to use bf16
        verbose: Verbosity level

    Returns:
        Trained agent
    """
    # Create save directory
    save_path = Path(save_dir) / experiment_name
    save_path.mkdir(parents=True, exist_ok=True)

    # Create reward shaper
    reward_config = reward_config or {'type': 'simple', 'mode': 'improvement'}
    reward_shaper = create_reward_shaper(reward_config)

    # Create environment
    env = FlexTokSearchEnv(
        flextok_model=flextok_model,
        target_image=target_image,
        similarity_fn=similarity_fn,
        fsq_levels=fsq_levels,
        max_tokens=max_tokens,
        enable_bf16=enable_bf16,
        device=device,
        reward_shaper=reward_shaper,
    )

    # Wrap with Monitor for logging
    env = Monitor(env, str(save_path / 'monitor'))

    # Create evaluation environment
    eval_env = FlexTokSearchEnv(
        flextok_model=flextok_model,
        target_image=target_image,
        similarity_fn=similarity_fn,
        fsq_levels=fsq_levels,
        max_tokens=max_tokens,
        enable_bf16=enable_bf16,
        device=device,
        reward_shaper=create_reward_shaper(reward_config),
    )
    eval_env = Monitor(eval_env, str(save_path / 'eval_monitor'))

    # Create agent
    agent_config = agent_config or {}
    agent_config['tensorboard_log'] = str(save_path / 'tensorboard')
    agent_config['device'] = device
    agent_config['verbose'] = verbose

    agent = create_agent(agent_type, env, agent_config)

    # Create callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(save_path / 'checkpoints'),
        name_prefix=f'{agent_type}_model',
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / 'best_model'),
        log_path=str(save_path / 'eval_logs'),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)

    callback_list = CallbackList(callbacks)

    # Train agent
    print(f"Training {agent_type.upper()} agent for {total_timesteps} timesteps...")
    print(f"Save directory: {save_path}")
    print(f"Agent config: {agent_config}")
    print(f"Reward config: {reward_config}")

    agent.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
    )

    # Save final model
    final_model_path = save_path / f'{agent_type}_final.zip'
    agent.save(str(final_model_path))
    print(f"Training complete! Final model saved to {final_model_path}")

    # Save configuration
    config = {
        'agent_type': agent_type,
        'fsq_levels': fsq_levels,
        'max_tokens': max_tokens,
        'total_timesteps': total_timesteps,
        'reward_config': reward_config,
        'agent_config': agent_config,
    }
    config_path = save_path / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    return agent


def main():
    """Main training script with CLI."""
    parser = argparse.ArgumentParser(description='Train RL agent for FlexTok token search')

    # Model arguments
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to FlexTok checkpoint')
    parser.add_argument('--fsq_levels', type=int, nargs='+', default=[8],
                        help='FSQ quantization levels')
    parser.add_argument('--max_tokens', type=int, default=256,
                        help='Maximum number of tokens')

    # Training arguments
    parser.add_argument('--agent_type', type=str, default='dqn',
                        choices=['sac', 'dqn', 'ppo'],
                        help='Type of RL agent')
    parser.add_argument('--total_timesteps', type=int, default=100000,
                        help='Total training timesteps')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')

    # Reward shaping
    parser.add_argument('--reward_mode', type=str, default='improvement',
                        choices=['negative', 'inverse', 'exponential', 'improvement'],
                        help='Reward shaping mode')

    # Environment
    parser.add_argument('--target_image_path', type=str, required=True,
                        help='Path to target image')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    # Logging
    parser.add_argument('--save_dir', type=str, default='./rl_checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--experiment_name', type=str, default='flextok_search',
                        help='Name of experiment')
    parser.add_argument('--checkpoint_freq', type=int, default=10000,
                        help='Checkpoint save frequency')

    args = parser.parse_args()

    # Load FlexTok model
    print(f"Loading FlexTok model from {args.ckpt_path}...")
    flextok_model = load_flextok_model(
        ckpt_path=args.ckpt_path,
        fsq_level=args.fsq_levels
    ).to(args.device)

    # Load target image
    print(f"Loading target image from {args.target_image_path}...")
    target_image = Image.open(args.target_image_path).convert('RGB')
    target_image = target_image.resize((256, 256))

    # Load similarity model (DreamSim)
    print("Loading DreamSim model...")
    from dreamsim import dreamsim
    dreamsim_model, dreamsim_preprocess = dreamsim(pretrained=True, device=args.device)
    similarity_fn = create_similarity_fn(dreamsim_model, dreamsim_preprocess)

    # Create configs
    reward_config = {
        'type': 'simple',
        'mode': args.reward_mode,
    }

    agent_config = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
    }

    # Train agent
    agent = train_agent(
        agent_type=args.agent_type,
        flextok_model=flextok_model,
        target_image=target_image,
        similarity_fn=similarity_fn,
        fsq_levels=args.fsq_levels,
        max_tokens=args.max_tokens,
        total_timesteps=args.total_timesteps,
        reward_config=reward_config,
        agent_config=agent_config,
        save_dir=args.save_dir,
        experiment_name=args.experiment_name,
        checkpoint_freq=args.checkpoint_freq,
        device=args.device,
    )

    print("Training complete!")


if __name__ == '__main__':
    main()

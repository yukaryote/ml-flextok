"""
Training script for FlexTok token search RL agents.

Hydra-compatible training with WandB logging, video trajectories, and checkpointing.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Optional, Callable
import torch
from PIL import Image
import numpy as np

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm

from flextok.flextok_wrapper import FlexTok
from flextok.utils.misc import detect_bf16_support
from flextok.utils.dataloader import CelebAHQDataset
from twenty_questions.twenty_questions import load_flextok_model, convert_images_to_pil

from rl_search.environment import FlexTokSearchEnv
from rl_search.multi_target_wrapper import (
    MultiTargetWrapper,
    DatasetTargetSampler,
    FixedSetTargetSampler,
)
from rl_search.reward_shaping import create_reward_shaper
from rl_search.agents import create_agent

# Import WandB callbacks
try:
    import wandb
    from rl_search.wandb_callbacks import WandbCallback, WandbEvalCallback, init_wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
    print("Warning: WandB not available. Training will proceed without WandB logging.")


class TqdmCallback(BaseCallback):
    """
    Custom callback for displaying a progress bar during training.

    Shows:
    - Overall training progress
    - Current episode statistics (reward, length, best score)
    - Training speed (steps/sec)
    """

    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

        # Episode tracking
        self.episode_count = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.last_best_score = None

    def _on_training_start(self) -> None:
        """Initialize progress bar when training starts."""
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="Training",
            unit="steps",
            ncols=120,
            leave=True,
        )

    def _on_step(self) -> bool:
        """Update progress bar on each step."""
        # Update progress bar
        self.pbar.update(1)

        # Track episode progress
        if len(self.locals.get('rewards', [])) > 0:
            self.current_episode_reward += self.locals['rewards'][0]
            self.current_episode_length += 1

        # Check if episode ended
        dones = self.locals.get('dones', [False])
        if dones[0]:
            self.episode_count += 1

            # Get info from the episode
            infos = self.locals.get('infos', [{}])
            if infos and len(infos) > 0:
                info = infos[0]
                best_score = info.get('best_score', None)
                if best_score is not None:
                    self.last_best_score = best_score

            # Update progress bar description with episode stats
            postfix_dict = {
                'ep': self.episode_count,
                'ep_rew': f'{self.current_episode_reward:.2f}',
                'ep_len': self.current_episode_length,
            }

            if self.last_best_score is not None:
                postfix_dict['best_score'] = f'{self.last_best_score:.4f}'

            self.pbar.set_postfix(postfix_dict)

            # Reset episode tracking
            self.current_episode_reward = 0
            self.current_episode_length = 0

        return True

    def _on_training_end(self) -> None:
        """Close progress bar when training ends."""
        if self.pbar is not None:
            self.pbar.close()


def create_similarity_fn(similarity_model, preprocess_fn, device: str):
    """
    Create similarity function wrapper for DreamSim or other models.

    Args:
        similarity_model: The similarity model (e.g., DreamSim)
        preprocess_fn: Preprocessing function
        device: Device to use

    Returns:
        Similarity function: (img1_pil, img2_pil) -> float
    """
    def similarity_fn(img1: Image.Image, img2: Image.Image) -> float:
        """Compute similarity score between two PIL images."""
        # Preprocess images
        img1_tensor = preprocess_fn(img1).to(device)
        img2_tensor = preprocess_fn(img2).to(device)

        # Compute similarity
        with torch.no_grad():
            score = similarity_model(img1_tensor, img2_tensor)

        return float(score.item())

    return similarity_fn


def load_dataset(cfg: DictConfig, device: str):
    """Load dataset for multi-target training."""
    if cfg.dataset.name in ["celebahq", "celeba"]:
        dataset = CelebAHQDataset(
            root_dir=cfg.dataset.path,
            img_size=cfg.dataset.img_size,
            split=cfg.dataset.split,
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")

    return dataset


def create_target_sampler(cfg: DictConfig, dataset, device: str):
    """Create target sampler based on configuration."""
    if cfg.training.multi_target.enabled:
        # Convert function for dataset items to PIL images
        convert_fn = lambda x: convert_images_to_pil(x.unsqueeze(0).to(device))[0]

        if cfg.training.multi_target.sampler_type == "dataset":
            return DatasetTargetSampler(
                dataset=dataset,
                convert_fn=convert_fn,
                device=device,
                seed=cfg.seed,
            )
        elif cfg.training.multi_target.sampler_type == "fixed":
            # Sample a fixed set of images
            n_images = cfg.training.multi_target.get('n_fixed_images', 10)
            indices = np.random.RandomState(cfg.seed).choice(
                len(dataset), size=n_images, replace=False
            )
            target_images = [
                convert_fn(dataset[idx]) for idx in indices
            ]
            return FixedSetTargetSampler(
                target_images=target_images,
                mode='random',
                seed=cfg.seed,
            )
        else:
            raise ValueError(f"Unknown sampler type: {cfg.training.multi_target.sampler_type}")
    else:
        # Single target - return None
        return None


@hydra.main(version_base=None, config_path="../configs/rl_search", config_name="train")
def main(cfg: DictConfig):
    """Main training function with Hydra configuration."""

    # Print configuration
    print("="*80)
    print("Training Configuration:")
    print("="*80)
    print(OmegaConf.to_yaml(cfg))
    print("="*80)

    # Set device
    device = cfg.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Detect bf16 support
    enable_bf16 = detect_bf16_support() if cfg.get('enable_bf16', True) else False
    print(f"BF16 enabled: {enable_bf16}")

    # Set random seed
    if cfg.get('seed'):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    # Initialize WandB
    if WANDB_AVAILABLE and cfg.wandb.enabled:
        # Create WandB config from Hydra config
        wandb_config = OmegaConf.to_container(cfg, resolve=True)

        init_wandb(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config=wandb_config,
            tags=cfg.wandb.get('tags', []),
            notes=cfg.wandb.get('notes', ''),
            mode=cfg.wandb.get('mode', 'online'),
        )
        print(f"WandB initialized: {wandb.run.name}")

    # Create save directory
    save_path = Path(cfg.training.save_dir) / cfg.experiment_name
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Save directory: {save_path}")

    # Load FlexTok model
    print(f"Loading FlexTok model from {cfg.model.checkpoint_path}...")
    flextok_model = load_flextok_model(
        ckpt_path=cfg.model.checkpoint_path,
        fsq_level=cfg.model.fsq_levels
    ).to(device)
    print(f"Model loaded with FSQ levels: {cfg.model.fsq_levels}")

    # Load similarity model (DreamSim)
    print("Loading DreamSim model...")
    from dreamsim import dreamsim
    dreamsim_model, dreamsim_preprocess = dreamsim(pretrained=True, device=device)
    similarity_fn = create_similarity_fn(dreamsim_model, dreamsim_preprocess, device)
    print("DreamSim model loaded")

    # Load dataset
    print(f"Loading dataset: {cfg.dataset.name}...")
    dataset = load_dataset(cfg, device)
    print(f"Dataset loaded: {len(dataset)} images")

    # Create target sampler
    target_sampler = create_target_sampler(cfg, dataset, device)

    # Get initial target image
    if target_sampler is not None:
        initial_target = target_sampler()
        print("Using multi-target training")
    else:
        # Load single target image
        if cfg.dataset.get('target_image_path'):
            initial_target = Image.open(cfg.dataset.target_image_path).convert('RGB')
            initial_target = initial_target.resize((cfg.dataset.img_size, cfg.dataset.img_size))
        else:
            # Use random image from dataset
            idx = np.random.randint(0, len(dataset))
            initial_target = convert_images_to_pil(dataset[idx].unsqueeze(0).to(device))[0]
        print("Using single-target training")

    # Create reward shaper
    reward_config = OmegaConf.to_container(cfg.reward, resolve=True)
    reward_shaper = create_reward_shaper(reward_config)

    # Create training environment
    print("Creating training environment...")
    env = FlexTokSearchEnv(
        flextok_model=flextok_model,
        target_image=initial_target,
        similarity_fn=similarity_fn,
        fsq_levels=cfg.model.fsq_levels,
        max_tokens=cfg.environment.max_tokens,
        history_length=cfg.environment.history_length,
        enable_bf16=enable_bf16,
        device=device,
        goal_threshold=cfg.environment.goal_threshold,
        enable_undo=cfg.environment.enable_undo,
        image_obs=cfg.environment.image_obs,
        reward_shaper=reward_shaper,
    )

    # Wrap with MultiTargetWrapper if enabled
    if target_sampler is not None:
        env = MultiTargetWrapper(env, target_sampler=target_sampler)

    # Wrap with Monitor
    env = Monitor(env, str(save_path / 'monitor'))

    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = FlexTokSearchEnv(
        flextok_model=flextok_model,
        target_image=initial_target,
        similarity_fn=similarity_fn,
        fsq_levels=cfg.model.fsq_levels,
        max_tokens=cfg.environment.max_tokens,
        history_length=cfg.environment.history_length,
        enable_bf16=enable_bf16,
        device=device,
        goal_threshold=cfg.environment.goal_threshold,
        enable_undo=cfg.environment.enable_undo,
        image_obs=cfg.environment.image_obs,
        reward_shaper=reward_shaper,
    )
    eval_env = Monitor(eval_env, str(save_path / 'eval_monitor'))

    # Create agent
    print(f"Creating {cfg.agent.type.upper()} agent...")
    agent_config = OmegaConf.to_container(cfg.agent.config, resolve=True)
    agent_config['tensorboard_log'] = str(save_path / 'tensorboard')
    agent_config['device'] = device
    agent_config['verbose'] = cfg.get('verbose', 1)

    agent = create_agent(cfg.agent.type, env, agent_config)
    print("Agent created successfully")

    # Create callbacks
    callbacks = []

    # Progress bar callback (add first so it displays at the top)
    tqdm_callback = TqdmCallback(
        total_timesteps=cfg.training.total_timesteps,
        verbose=cfg.get('verbose', 1),
    )
    callbacks.append(tqdm_callback)
    print("Progress bar callback added")

    # WandB callback
    if WANDB_AVAILABLE and cfg.wandb.enabled:
        wandb_callback = WandbCallback(
            env=env,
            log_freq=cfg.wandb.log_freq,
            save_freq=cfg.training.checkpoint_freq,
            save_path=str(save_path / 'wandb_checkpoints'),
            log_images=cfg.wandb.log_images,
            image_log_freq=cfg.wandb.image_log_freq,
            log_videos=cfg.wandb.log_videos,
            video_log_freq=cfg.wandb.video_log_freq,
            video_fps=cfg.wandb.video_fps,
            verbose=cfg.get('verbose', 1),
        )
        callbacks.append(wandb_callback)
        print("WandB callback added")

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.training.checkpoint_freq,
        save_path=str(save_path / 'checkpoints'),
        name_prefix=f'{cfg.agent.type}_model',
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # WandB evaluation callback
    if WANDB_AVAILABLE and cfg.wandb.enabled and cfg.training.eval_freq > 0:
        wandb_eval_callback = WandbEvalCallback(
            eval_env=eval_env,
            n_eval_episodes=cfg.training.n_eval_episodes,
            eval_freq=cfg.training.eval_freq,
            deterministic=True,
            log_images=cfg.wandb.log_images,
            log_videos=cfg.wandb.log_videos,
            video_fps=cfg.wandb.video_fps,
            verbose=cfg.get('verbose', 1),
        )
        callbacks.append(wandb_eval_callback)
        print("WandB evaluation callback added")

    callback_list = CallbackList(callbacks)

    # Train agent
    print("="*80)
    print(f"Starting training for {cfg.training.total_timesteps} timesteps...")
    print("="*80)

    agent.learn(
        total_timesteps=cfg.training.total_timesteps,
        callback=callback_list,
        log_interval=cfg.training.get('log_interval', 10),
    )

    # Save final model
    final_model_path = save_path / f'{cfg.agent.type}_final.zip'
    agent.save(str(final_model_path))
    print(f"\nTraining complete! Final model saved to {final_model_path}")

    # Save Hydra config
    config_path = save_path / 'hydra_config.yaml'
    with open(config_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"Configuration saved to {config_path}")

    # Finish WandB run
    if WANDB_AVAILABLE and cfg.wandb.enabled:
        # Log final model as artifact
        artifact = wandb.Artifact(
            name=f"{cfg.experiment_name}_model",
            type="model",
            description=f"Final {cfg.agent.type} model after {cfg.training.total_timesteps} timesteps"
        )
        artifact.add_file(str(final_model_path))
        wandb.log_artifact(artifact)

        wandb.finish()
        print("WandB run finished")

    return agent


if __name__ == '__main__':
    main()

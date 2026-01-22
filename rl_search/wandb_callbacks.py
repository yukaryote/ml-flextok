"""
Weights & Biases callbacks and utilities for RL training.

This module provides integration with WandB for tracking experiments,
logging metrics, and visualizing RL agent performance.
"""
import wandb
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
from PIL import Image
import io
import imageio
import tempfile


class WandbCallback(BaseCallback):
    """
    Custom callback for logging RL training metrics to Weights & Biases.

    This callback logs:
    - Training metrics (rewards, losses, etc.)
    - Episode statistics (length, best score, etc.)
    - Visualizations (images, trajectories, etc.)
    - Model checkpoints

    Args:
        env: The training environment (should be FlexTokSearchEnv)
        log_freq: Frequency (in timesteps) to log metrics
        save_freq: Frequency (in timesteps) to save model checkpoints
        save_path: Path to save model checkpoints
        log_images: Whether to log decoded images to WandB
        image_log_freq: Frequency (in episodes) to log images
        log_videos: Whether to log video trajectories to WandB
        video_log_freq: Frequency (in episodes) to log videos
        video_fps: Frame rate for video logging
        verbose: Verbosity level
    """

    def __init__(
        self,
        env=None,
        log_freq: int = 1000,
        save_freq: int = 10000,
        save_path: Optional[str] = None,
        log_images: bool = True,
        image_log_freq: int = 10,
        log_videos: bool = True,
        video_log_freq: int = 50,
        video_fps: int = 2,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.env = env
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.save_path = Path(save_path) if save_path else None
        self.log_images = log_images
        self.image_log_freq = image_log_freq
        self.log_videos = log_videos
        self.video_log_freq = video_log_freq
        self.video_fps = video_fps

        # Episode tracking
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_best_scores = []
        self.episode_final_scores = []

        # Current episode tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_episode_scores = []
        self.current_episode_tokens = []
        self.current_episode_images = []

        # Create save directory if needed
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_training_start(self) -> None:
        """Called before training starts."""
        # Log environment configuration
        if hasattr(self.training_env.envs[0], 'unwrapped'):
            env = self.training_env.envs[0].unwrapped
            config = {
                'fsq_levels': env.fsq_levels,
                'max_tokens': env.max_tokens,
                'history_length': env.history_length,
                'goal_threshold': env.goal_threshold,
                'enable_undo': env.enable_undo,
                'image_obs': env.image_obs,
            }
            wandb.config.update(config, allow_val_change=True)

    def _on_step(self) -> bool:
        """
        Called after each environment step.

        Returns:
            bool: If False, training will be stopped.
        """
        # Get info from the environment
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]

            # Track episode progress
            if 'current_score' in info:
                self.current_episode_scores.append(info['current_score'])

            if 'token_value' in info:
                self.current_episode_tokens.append(info['token_value'])

            # Capture current image for video if enabled
            if self.log_videos:
                try:
                    if hasattr(self.training_env.envs[0], 'unwrapped'):
                        env = self.training_env.envs[0].unwrapped
                        if env.current_image is not None:
                            self.current_episode_images.append(np.array(env.current_image))
                except Exception:
                    pass

            self.current_episode_reward += self.locals.get('rewards', [0])[0]
            self.current_episode_length += 1

            # Check if episode is done
            dones = self.locals.get('dones', [False])
            if dones[0]:
                self._on_episode_end(info)

        # Log training metrics at specified frequency
        if self.n_calls % self.log_freq == 0:
            self._log_training_metrics()

        # Save model at specified frequency
        if self.save_path and self.n_calls % self.save_freq == 0:
            self._save_checkpoint()

        return True

    def _on_episode_end(self, info: Dict[str, Any]) -> None:
        """Called when an episode ends."""
        self.episode_count += 1

        # Store episode statistics
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_lengths.append(self.current_episode_length)

        best_score = info.get('best_score', float('inf'))
        self.episode_best_scores.append(best_score)

        final_score = self.current_episode_scores[-1] if self.current_episode_scores else float('inf')
        self.episode_final_scores.append(final_score)

        # Log episode metrics
        wandb.log({
            'episode/reward': self.current_episode_reward,
            'episode/length': self.current_episode_length,
            'episode/best_score': best_score,
            'episode/final_score': final_score,
            'episode/count': self.episode_count,
        }, step=self.num_timesteps)

        # Log images if enabled
        if self.log_images and self.episode_count % self.image_log_freq == 0:
            self._log_episode_images()

        # Log trajectory visualization
        if self.episode_count % self.image_log_freq == 0:
            self._log_trajectory_plot()

        # Log video if enabled
        if self.log_videos and self.episode_count % self.video_log_freq == 0:
            self._log_episode_video()

        # Reset current episode tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_episode_scores = []
        self.current_episode_tokens = []
        self.current_episode_images = []

    def _log_training_metrics(self) -> None:
        """Log training metrics to WandB."""
        # Get training metrics from the model
        if hasattr(self.model, 'logger') and self.model.logger:
            # For DQN, SAC, PPO - they log different metrics
            pass

        # Log recent episode statistics (rolling average)
        if len(self.episode_rewards) > 0:
            recent_window = min(100, len(self.episode_rewards))
            recent_rewards = self.episode_rewards[-recent_window:]
            recent_scores = self.episode_best_scores[-recent_window:]

            wandb.log({
                'train/mean_reward_100': np.mean(recent_rewards),
                'train/mean_best_score_100': np.mean(recent_scores),
                'train/total_episodes': self.episode_count,
            }, step=self.num_timesteps)

    def _log_episode_images(self) -> None:
        """Log episode images to WandB."""
        try:
            # Get current image from environment
            if hasattr(self.training_env.envs[0], 'unwrapped'):
                env = self.training_env.envs[0].unwrapped
                if env.current_image is not None:
                    # Log the current decoded image
                    wandb.log({
                        'images/decoded': wandb.Image(
                            env.current_image,
                            caption=f"Episode {self.episode_count}, Score: {self.current_episode_scores[-1]:.4f}"
                        )
                    }, step=self.num_timesteps)

                # Log target image (only once)
                if self.episode_count == 1 and env.target_image is not None:
                    wandb.log({
                        'images/target': wandb.Image(
                            env.target_image,
                            caption="Target Image"
                        )
                    }, step=self.num_timesteps)
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not log images: {e}")

    def _log_trajectory_plot(self) -> None:
        """Log trajectory visualization to WandB."""
        if not self.current_episode_scores:
            return

        try:
            # Create trajectory plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Plot 1: Similarity scores
            ax1 = axes[0]
            steps = list(range(len(self.current_episode_scores)))
            ax1.plot(steps, self.current_episode_scores, marker='o', markersize=3)
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Similarity Score')
            ax1.set_title(f'Episode {self.episode_count} - Similarity Score')
            ax1.grid(alpha=0.3)

            # Plot 2: Token values
            ax2 = axes[1]
            if self.current_episode_tokens:
                ax2.plot(steps, self.current_episode_tokens, marker='o', markersize=3, color='purple')
                ax2.set_xlabel('Step')
                ax2.set_ylabel('Token Value')
                ax2.set_title(f'Episode {self.episode_count} - Token Choices')
                ax2.grid(alpha=0.3)

            plt.tight_layout()

            # Convert to image and log
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)

            wandb.log({
                'plots/trajectory': wandb.Image(
                    Image.open(buf),
                    caption=f"Episode {self.episode_count}"
                )
            }, step=self.num_timesteps)

            plt.close(fig)
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not log trajectory plot: {e}")

    def _log_episode_video(self) -> None:
        """Log episode video showing the search trajectory."""
        if not self.current_episode_images or len(self.current_episode_images) < 2:
            return

        try:
            # Create a video showing the search trajectory
            # Add text overlays with scores and token values
            frames = []

            for idx, img in enumerate(self.current_episode_images):
                # Convert to RGB if needed
                if img.shape[-1] == 4:  # RGBA
                    img = img[:, :, :3]

                # Create figure with image and info
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(img)
                ax.axis('off')

                # Add text annotations
                score = self.current_episode_scores[idx] if idx < len(self.current_episode_scores) else 0
                token = self.current_episode_tokens[idx] if idx < len(self.current_episode_tokens) else 0

                title = f"Step {idx+1}/{len(self.current_episode_images)}\n"
                title += f"Score: {score:.4f} | Token: {token}"
                ax.set_title(title, fontsize=12, pad=10)

                # Convert figure to image
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=80, bbox_inches='tight', pad_inches=0.1)
                buf.seek(0)
                frame = np.array(Image.open(buf))
                frames.append(frame)
                plt.close(fig)

            # Save video to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                video_path = tmp_file.name

            # Write video
            imageio.mimsave(video_path, frames, fps=self.video_fps)

            # Log to WandB
            wandb.log({
                'videos/search_trajectory': wandb.Video(
                    video_path,
                    fps=self.video_fps,
                    format='mp4',
                    caption=f"Episode {self.episode_count} - Search Trajectory"
                )
            }, step=self.num_timesteps)

            # Clean up temporary file
            import os
            os.unlink(video_path)

            if self.verbose > 0:
                print(f"Logged video for episode {self.episode_count}")

        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not log episode video: {e}")

    def _save_checkpoint(self) -> None:
        """Save model checkpoint."""
        if self.save_path:
            checkpoint_path = self.save_path / f"checkpoint_{self.num_timesteps}.zip"
            self.model.save(checkpoint_path)

            # Log to WandB
            wandb.save(str(checkpoint_path))

            if self.verbose > 0:
                print(f"Saved checkpoint to {checkpoint_path}")


class WandbEvalCallback(BaseCallback):
    """
    Callback for periodic evaluation and logging to WandB.

    Runs evaluation episodes at regular intervals and logs:
    - Evaluation metrics
    - Comparison with training performance
    - Visualization of best/worst episodes

    Args:
        eval_env: Environment for evaluation
        n_eval_episodes: Number of episodes to run for evaluation
        eval_freq: Frequency (in timesteps) to run evaluation
        deterministic: Whether to use deterministic actions during evaluation
        log_images: Whether to log images from evaluation episodes
        log_videos: Whether to log videos from best evaluation episode
        video_fps: Frame rate for video logging
        verbose: Verbosity level
    """

    def __init__(
        self,
        eval_env,
        n_eval_episodes: int = 10,
        eval_freq: int = 10000,
        deterministic: bool = True,
        log_images: bool = True,
        log_videos: bool = True,
        video_fps: int = 2,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.log_images = log_images
        self.log_videos = log_videos
        self.video_fps = video_fps
        self.eval_count = 0

    def _on_step(self) -> bool:
        """Called after each step."""
        if self.n_calls % self.eval_freq == 0:
            self._run_evaluation()
        return True

    def _run_evaluation(self) -> None:
        """Run evaluation episodes."""
        self.eval_count += 1

        episode_rewards = []
        episode_lengths = []
        episode_best_scores = []
        episode_final_scores = []
        best_trajectory = None
        worst_trajectory = None
        best_reward = -float('inf')
        worst_reward = float('inf')

        for episode in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            trajectory = {'scores': [], 'tokens': [], 'images': []}

            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

                # Track trajectory
                if 'current_score' in info:
                    trajectory['scores'].append(info['current_score'])
                if 'token_value' in info:
                    trajectory['tokens'].append(info['token_value'])

                # Store image
                if self.log_images:
                    img = self.eval_env.render()
                    if img is not None:
                        trajectory['images'].append(img)

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_best_scores.append(info.get('best_score', float('inf')))
            episode_final_scores.append(trajectory['scores'][-1] if trajectory['scores'] else float('inf'))

            # Track best/worst episodes
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_trajectory = trajectory
            if episode_reward < worst_reward:
                worst_reward = episode_reward
                worst_trajectory = trajectory

        # Log evaluation metrics
        wandb.log({
            'eval/mean_reward': np.mean(episode_rewards),
            'eval/std_reward': np.std(episode_rewards),
            'eval/mean_best_score': np.mean(episode_best_scores),
            'eval/mean_final_score': np.mean(episode_final_scores),
            'eval/mean_length': np.mean(episode_lengths),
            'eval/count': self.eval_count,
        }, step=self.num_timesteps)

        # Log best episode visualization
        if best_trajectory:
            self._log_trajectory(
                best_trajectory,
                f"Best Episode (Reward: {best_reward:.2f})",
                prefix='eval/best'
            )

        if self.verbose > 0:
            print(f"\nEvaluation {self.eval_count}:")
            print(f"  Mean reward: {np.mean(episode_rewards):.2f}")
            print(f"  Mean best score: {np.mean(episode_best_scores):.4f}")

    def _log_trajectory(self, trajectory: Dict, title: str, prefix: str) -> None:
        """Log trajectory visualization."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Plot scores
            ax1 = axes[0]
            steps = list(range(len(trajectory['scores'])))
            ax1.plot(steps, trajectory['scores'], marker='o', markersize=3)
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Similarity Score')
            ax1.set_title(f'{title} - Similarity')
            ax1.grid(alpha=0.3)

            # Plot tokens
            ax2 = axes[1]
            if trajectory['tokens']:
                ax2.plot(steps, trajectory['tokens'], marker='o', markersize=3, color='purple')
                ax2.set_xlabel('Step')
                ax2.set_ylabel('Token Value')
                ax2.set_title(f'{title} - Tokens')
                ax2.grid(alpha=0.3)

            plt.tight_layout()

            # Convert to image and log
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)

            wandb.log({
                f'{prefix}_trajectory': wandb.Image(Image.open(buf), caption=title)
            }, step=self.num_timesteps)

            plt.close(fig)

            # Log final image if available
            if self.log_images and trajectory['images']:
                final_image = trajectory['images'][-1]
                wandb.log({
                    f'{prefix}_final_image': wandb.Image(
                        final_image,
                        caption=f"{title} - Final Image"
                    )
                }, step=self.num_timesteps)

            # Log video if available
            if self.log_videos and trajectory['images'] and len(trajectory['images']) >= 2:
                self._log_trajectory_video(trajectory, title, prefix)

        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not log trajectory: {e}")

    def _log_trajectory_video(self, trajectory: Dict, title: str, prefix: str) -> None:
        """Log trajectory video."""
        try:
            frames = []

            for idx, img in enumerate(trajectory['images']):
                # Ensure numpy array
                if isinstance(img, Image.Image):
                    img = np.array(img)

                # Convert to RGB if needed
                if img.shape[-1] == 4:  # RGBA
                    img = img[:, :, :3]

                # Create figure with image and info
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(img)
                ax.axis('off')

                # Add text annotations
                score = trajectory['scores'][idx] if idx < len(trajectory['scores']) else 0
                token = trajectory['tokens'][idx] if idx < len(trajectory['tokens']) else 0

                info_text = f"Step {idx+1}/{len(trajectory['images'])}\n"
                info_text += f"Score: {score:.4f} | Token: {token}"
                ax.set_title(info_text, fontsize=12, pad=10)

                # Convert figure to image
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=80, bbox_inches='tight', pad_inches=0.1)
                buf.seek(0)
                frame = np.array(Image.open(buf))
                frames.append(frame)
                plt.close(fig)

            # Save video to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                video_path = tmp_file.name

            # Write video
            imageio.mimsave(video_path, frames, fps=self.video_fps)

            # Log to WandB
            wandb.log({
                f'{prefix}_video': wandb.Video(
                    video_path,
                    fps=self.video_fps,
                    format='mp4',
                    caption=title
                )
            }, step=self.num_timesteps)

            # Clean up temporary file
            import os
            os.unlink(video_path)

            if self.verbose > 0:
                print(f"Logged video for {title}")

        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not log trajectory video: {e}")


def init_wandb(
    project: str,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
    mode: str = 'online',
    **kwargs
) -> wandb.run:
    """
    Initialize Weights & Biases for RL experiment tracking.

    Args:
        project: WandB project name
        name: Run name (optional)
        config: Configuration dictionary
        tags: List of tags for the run
        notes: Notes about the run
        mode: 'online', 'offline', or 'disabled'
        **kwargs: Additional arguments for wandb.init()

    Returns:
        WandB run object

    Example:
        >>> run = init_wandb(
        ...     project='flextok-rl-search',
        ...     name='dqn-fsq8-experiment1',
        ...     config={'learning_rate': 1e-4, 'batch_size': 32},
        ...     tags=['dqn', 'fsq-8'],
        ...     notes='Testing DQN with FSQ level 8'
        ... )
    """
    run = wandb.init(
        project=project,
        name=name,
        config=config,
        tags=tags,
        notes=notes,
        mode=mode,
        **kwargs
    )

    return run


def log_environment_info(env) -> None:
    """
    Log environment information to WandB.

    Args:
        env: FlexTokSearchEnv instance
    """
    env_info = {
        'env/fsq_levels': env.fsq_levels,
        'env/max_tokens': env.max_tokens,
        'env/history_length': env.history_length,
        'env/goal_threshold': env.goal_threshold,
        'env/enable_undo': env.enable_undo,
        'env/image_obs': env.image_obs,
        'env/action_space': str(env.action_space),
        'env/observation_space': str(env.observation_space),
    }

    wandb.config.update(env_info, allow_val_change=True)


def log_agent_config(agent_type: str, agent_config: Dict[str, Any]) -> None:
    """
    Log agent configuration to WandB.

    Args:
        agent_type: Type of agent ('dqn', 'sac', 'ppo')
        agent_config: Agent configuration dictionary
    """
    config = {
        'agent/type': agent_type,
        **{f'agent/{k}': v for k, v in agent_config.items()}
    }

    wandb.config.update(config, allow_val_change=True)


def log_comparison_table(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Log comparison table of multiple agents to WandB.

    Args:
        results: Dictionary mapping agent names to their evaluation results

    Example:
        >>> results = {
        ...     'DQN': {'mean_reward': 10.5, 'mean_best_score': 0.15},
        ...     'SAC': {'mean_reward': 12.3, 'mean_best_score': 0.12},
        ... }
        >>> log_comparison_table(results)
    """
    # Create comparison table
    columns = ['Agent', 'Mean Reward', 'Std Reward', 'Mean Best Score', 'Mean Final Score']
    data = []

    for name, result in results.items():
        data.append([
            name,
            result.get('mean_reward', 0),
            result.get('std_reward', 0),
            result.get('mean_best_score', 0),
            result.get('mean_final_score', 0),
        ])

    table = wandb.Table(columns=columns, data=data)
    wandb.log({'comparison/agents': table})


def log_token_distribution(trajectories: List[List[Dict]], fsq_levels: List[int]) -> None:
    """
    Log token distribution histogram to WandB.

    Args:
        trajectories: List of trajectories from evaluation
        fsq_levels: FSQ quantization levels
    """
    # Collect all tokens
    all_tokens = []
    for traj in trajectories:
        tokens = [step.get('token_value', 0) for step in traj]
        all_tokens.extend(tokens)

    # Create histogram
    if len(fsq_levels) == 1:
        # Discrete tokens
        token_counts = {i: 0 for i in range(fsq_levels[0])}
        for token in all_tokens:
            if token in token_counts:
                token_counts[token] += 1

        # Log as WandB histogram
        wandb.log({
            'analysis/token_distribution': wandb.Histogram(all_tokens, num_bins=fsq_levels[0])
        })

        # Also log as table
        data = [[token, count] for token, count in token_counts.items()]
        table = wandb.Table(columns=['Token Value', 'Count'], data=data)
        wandb.log({'analysis/token_counts': table})

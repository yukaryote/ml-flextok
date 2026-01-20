"""
Evaluation and visualization utilities for FlexTok token search RL agents.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Callable
from PIL import Image
import torch
from pathlib import Path

from flextok.flextok_wrapper import FlexTok
from rl_search.environment import FlexTokSearchEnv
from rl_search.agents import SACAgent, DQNAgent, PPOAgent


def evaluate_agent(
    agent,
    env: FlexTokSearchEnv,
    n_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Evaluate an RL agent on the environment.

    Args:
        agent: RL agent
        env: Environment
        n_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions
        render: Whether to render
        verbose: Whether to print progress

    Returns:
        Dictionary with evaluation metrics:
        - episode_rewards: List of episode rewards
        - episode_lengths: List of episode lengths
        - best_scores: List of best similarity scores per episode
        - final_scores: List of final similarity scores per episode
        - mean_reward: Mean episode reward
        - std_reward: Std episode reward
        - mean_best_score: Mean best similarity score
    """
    episode_rewards = []
    episode_lengths = []
    best_scores = []
    final_scores = []
    all_trajectories = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        trajectory = []

        while not done:
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # Store trajectory info
            trajectory.append({
                'token_idx': info.get('token_idx', 0),
                'token_value': info.get('token_value', 0),
                'score': info.get('current_score', 0),
                'reward': reward,
            })

            if render:
                img = env.render()
                if img is not None:
                    plt.imshow(img)
                    plt.title(f"Episode {episode}, Step {episode_length}")
                    plt.show()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        best_scores.append(info.get('best_score', float('inf')))
        final_scores.append(trajectory[-1]['score'] if trajectory else float('inf'))
        all_trajectories.append(trajectory)

        if verbose:
            print(f"Episode {episode+1}/{n_episodes}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Length={episode_length}, "
                  f"Best Score={info.get('best_score', 0):.4f}")

    results = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'best_scores': best_scores,
        'final_scores': final_scores,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_best_score': np.mean(best_scores),
        'mean_final_score': np.mean(final_scores),
        'trajectories': all_trajectories,
    }

    return results


def visualize_episode(
    agent,
    env: FlexTokSearchEnv,
    deterministic: bool = True,
    save_path: Optional[str] = None,
):
    """
    Visualize a single episode with plots.

    Args:
        agent: RL agent
        env: Environment
        deterministic: Whether to use deterministic actions
        save_path: Path to save visualization

    Returns:
        Figure
    """
    # Run episode
    obs, info = env.reset()
    done = False

    scores = []
    rewards = []
    token_values = []
    images = []

    while not done:
        action, _ = agent.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if 'current_score' in info:
            scores.append(info['current_score'])
            rewards.append(reward)
            token_values.append(info.get('token_value', 0))

            # Get current image
            img = env.render()
            if img is not None:
                images.append(img)

    # Create visualization
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: Similarity scores over time
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(scores, marker='o', markersize=3)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Similarity Score (lower = better)')
    ax1.set_title('Similarity Score Over Time')
    ax1.grid(alpha=0.3)

    # Plot 2: Rewards over time
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(rewards, marker='o', markersize=3, color='green')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Reward')
    ax2.set_title('Reward Over Time')
    ax2.grid(alpha=0.3)

    # Plot 3: Token values over time
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(token_values, marker='o', markersize=3, color='purple')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Token Value')
    ax3.set_title('Token Choices Over Time')
    ax3.grid(alpha=0.3)

    # Plot 4-6: Show key images
    # Show first, middle, and last images
    if len(images) >= 3:
        indices = [0, len(images)//2, -1]
        titles = ['First Image', 'Middle Image', 'Final Image']

        for i, (idx, title) in enumerate(zip(indices, titles)):
            ax = plt.subplot(2, 3, 4 + i)
            ax.imshow(images[idx])
            ax.set_title(f"{title}\nScore: {scores[idx]:.4f}")
            ax.axis('off')

    plt.suptitle(f'Episode Visualization (Final Score: {scores[-1]:.4f})', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    return fig


def compare_agents(
    agents: List[Tuple[str, any]],
    env: FlexTokSearchEnv,
    n_episodes: int = 10,
    save_path: Optional[str] = None,
):
    """
    Compare multiple agents on the same environment.

    Args:
        agents: List of (name, agent) tuples
        env: Environment
        n_episodes: Number of episodes per agent
        save_path: Path to save comparison plot

    Returns:
        Dictionary with comparison results
    """
    results = {}

    for name, agent in agents:
        print(f"\nEvaluating {name}...")
        eval_results = evaluate_agent(agent, env, n_episodes=n_episodes, verbose=False)
        results[name] = eval_results

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Episode rewards
    ax1 = axes[0, 0]
    for name in results.keys():
        rewards = results[name]['episode_rewards']
        ax1.plot(rewards, marker='o', label=name, alpha=0.7)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards Comparison')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Best scores
    ax2 = axes[0, 1]
    for name in results.keys():
        scores = results[name]['best_scores']
        ax2.plot(scores, marker='o', label=name, alpha=0.7)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Best Similarity Score')
    ax2.set_title('Best Scores Comparison')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Plot 3: Mean reward bar chart
    ax3 = axes[1, 0]
    names = list(results.keys())
    mean_rewards = [results[name]['mean_reward'] for name in names]
    std_rewards = [results[name]['std_reward'] for name in names]
    ax3.bar(names, mean_rewards, yerr=std_rewards, alpha=0.7)
    ax3.set_ylabel('Mean Episode Reward')
    ax3.set_title('Mean Reward Comparison')
    ax3.grid(alpha=0.3, axis='y')

    # Plot 4: Mean best score bar chart
    ax4 = axes[1, 1]
    mean_scores = [results[name]['mean_best_score'] for name in names]
    ax4.bar(names, mean_scores, alpha=0.7, color='green')
    ax4.set_ylabel('Mean Best Score')
    ax4.set_title('Mean Best Score Comparison')
    ax4.grid(alpha=0.3, axis='y')

    plt.suptitle('Agent Comparison', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to {save_path}")

    return results


def visualize_search_trajectory(
    trajectory: List[dict],
    target_image: Optional[Image.Image] = None,
    save_path: Optional[str] = None,
):
    """
    Visualize the search trajectory with scores and token choices.

    Args:
        trajectory: List of trajectory dictionaries from evaluation
        target_image: Optional target image to display
        save_path: Path to save visualization
    """
    scores = [step['score'] for step in trajectory]
    rewards = [step['reward'] for step in trajectory]
    token_values = [step['token_value'] for step in trajectory]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Plot 1: Scores
    ax1 = axes[0, 0]
    ax1.plot(scores, marker='o', markersize=4, linewidth=2)
    min_idx = np.argmin(scores)
    ax1.scatter(min_idx, scores[min_idx], color='red', s=100, zorder=5)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Similarity Score')
    ax1.set_title(f'Similarity Score (Best: {min(scores):.4f} at step {min_idx})')
    ax1.grid(alpha=0.3)

    # Plot 2: Rewards
    ax2 = axes[0, 1]
    ax2.plot(rewards, marker='o', markersize=4, linewidth=2, color='green')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Reward')
    ax2.set_title('Reward Over Time')
    ax2.grid(alpha=0.3)

    # Plot 3: Token values
    ax3 = axes[1, 0]
    ax3.plot(token_values, marker='o', markersize=4, linewidth=2, color='purple')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Token Value')
    ax3.set_title('Token Choices')
    ax3.grid(alpha=0.3)

    # Plot 4: Target image
    ax4 = axes[1, 1]
    if target_image is not None:
        ax4.imshow(target_image)
        ax4.set_title('Target Image')
    else:
        ax4.text(0.5, 0.5, 'No target image', ha='center', va='center')
    ax4.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Trajectory visualization saved to {save_path}")

    return fig


def analyze_token_distribution(
    trajectories: List[List[dict]],
    fsq_levels: List[int] = [8],
    save_path: Optional[str] = None,
):
    """
    Analyze the distribution of token choices across episodes.

    Args:
        trajectories: List of trajectories from multiple episodes
        fsq_levels: FSQ levels
        save_path: Path to save visualization
    """
    # Collect all token values
    all_tokens = []
    for traj in trajectories:
        tokens = [step['token_value'] for step in traj]
        all_tokens.extend(tokens)

    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    if len(fsq_levels) == 1:
        # Discrete tokens
        bins = np.arange(fsq_levels[0] + 1) - 0.5
        ax.hist(all_tokens, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_xticks(range(fsq_levels[0]))
        ax.set_xlabel('Token Value')
    else:
        # Continuous tokens
        ax.hist(all_tokens, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Token Value')

    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Token Choices')
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Token distribution saved to {save_path}")

    return fig


def create_evaluation_report(
    eval_results: dict,
    save_path: str,
    agent_name: str = 'Agent',
):
    """
    Create a comprehensive evaluation report.

    Args:
        eval_results: Results from evaluate_agent
        save_path: Path to save report
        agent_name: Name of agent
    """
    report_path = Path(save_path)
    report_path.mkdir(parents=True, exist_ok=True)

    # Save text report
    text_report = f"""
Evaluation Report for {agent_name}
{'=' * 50}

Summary Statistics:
- Number of episodes: {len(eval_results['episode_rewards'])}
- Mean episode reward: {eval_results['mean_reward']:.4f} ± {eval_results['std_reward']:.4f}
- Mean best score: {eval_results['mean_best_score']:.4f}
- Mean final score: {eval_results['mean_final_score']:.4f}
- Mean episode length: {np.mean(eval_results['episode_lengths']):.1f}

Episode-by-Episode Results:
"""

    for i, (reward, length, score) in enumerate(zip(
        eval_results['episode_rewards'],
        eval_results['episode_lengths'],
        eval_results['best_scores']
    )):
        text_report += f"Episode {i+1}: Reward={reward:.2f}, Length={length}, Best Score={score:.4f}\n"

    with open(report_path / 'evaluation_report.txt', 'w') as f:
        f.write(text_report)

    print(f"Evaluation report saved to {report_path / 'evaluation_report.txt'}")

    return text_report

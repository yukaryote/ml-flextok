#!/usr/bin/env python3
"""
Wandb Sweep Runner for FlexTok Fine-tuning

This script runs a hyperparameter sweep to find optimal learning rates and warmup steps.

Usage:
    # Initialize sweep (run once)
    python run_sweep.py --init

    # Run sweep agent (run on each machine/GPU)
    python run_sweep.py --agent SWEEP_ID

    # Or run both in one command
    python run_sweep.py --init --count 10
"""

import argparse
import os
import sys
import yaml
import subprocess

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


def load_sweep_config(config_path='sweep_config.yaml'):
    """Load sweep configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def init_sweep(config_path='sweep_config.yaml', project='flextok-finetuning'):
    """Initialize a new wandb sweep."""
    if not WANDB_AVAILABLE:
        print("Error: wandb is required to run sweeps")
        sys.exit(1)

    config = load_sweep_config(config_path)

    # Create sweep
    sweep_id = wandb.sweep(config, project=project)

    print("=" * 80)
    print("SWEEP INITIALIZED")
    print("=" * 80)
    print(f"Sweep ID: {sweep_id}")
    print(f"Project: {project}")
    print()
    print("To run the sweep agent, use:")
    print(f"  python run_sweep.py --agent {sweep_id}")
    print()
    print("Or on this machine:")
    print(f"  wandb agent {project}/{sweep_id}")
    print("=" * 80)

    return sweep_id


def train_function():
    """Training function to be called by wandb agent."""
    # Initialize wandb run
    run = wandb.init()

    # Get hyperparameters from sweep
    config = wandb.config

    # Build command line arguments
    cmd = [
        'python', 'train_flextok.py',
        '--config', config.config,
    ]

    # Add hyperparameters as overrides
    overrides = []
    if 'learning_rate' in config:
        overrides.append(f'learning_rate={config.learning_rate}')
    if 'encoder_lr' in config:
        overrides.append(f'encoder_lr={config.encoder_lr}')
    if 'decoder_lr' in config:
        overrides.append(f'decoder_lr={config.decoder_lr}')
    if 'warmup_steps' in config:
        overrides.append(f'warmup_steps={config.warmup_steps}')
    if 'max_grad_norm' in config:
        overrides.append(f'max_grad_norm={config.max_grad_norm}')
    if 'weight_decay' in config:
        overrides.append(f'weight_decay={config.weight_decay}')

    # Set wandb run name based on hyperparameters
    run_name = f"lr{config.get('learning_rate', 0):.2e}_warm{config.get('warmup_steps', 0)}"
    overrides.append(f'wandb_run_name={run_name}')

    if overrides:
        cmd.extend(overrides)

    print("=" * 80)
    print("STARTING TRAINING RUN")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print()
    print("Hyperparameters:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 80)

    # Run training
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        return e.returncode


def run_agent(sweep_id, project='flextok-finetuning', count=None):
    """Run a wandb sweep agent."""
    if not WANDB_AVAILABLE:
        print("Error: wandb is required to run sweeps")
        sys.exit(1)

    print("=" * 80)
    print("STARTING SWEEP AGENT")
    print("=" * 80)
    print(f"Sweep ID: {sweep_id}")
    print(f"Project: {project}")
    if count:
        print(f"Number of runs: {count}")
    print("=" * 80)
    print()

    # Run agent
    wandb.agent(
        f"{project}/{sweep_id}",
        function=train_function,
        count=count,
    )


def main():
    parser = argparse.ArgumentParser(description='Run wandb sweep for FlexTok fine-tuning')
    parser.add_argument('--init', action='store_true', help='Initialize a new sweep')
    parser.add_argument('--agent', type=str, help='Run sweep agent with given sweep ID')
    parser.add_argument('--config', type=str, default='sweep_config.yaml',
                        help='Path to sweep config YAML file')
    parser.add_argument('--project', type=str, default='flextok-finetuning',
                        help='Wandb project name')
    parser.add_argument('--count', type=int, help='Number of runs to execute (default: unlimited)')

    args = parser.parse_args()

    if not WANDB_AVAILABLE:
        print("Error: wandb is not installed. Install with: pip install wandb")
        sys.exit(1)

    if args.init:
        # Initialize sweep
        sweep_id = init_sweep(args.config, args.project)

        # Optionally run agent immediately
        if args.count or args.agent is None:
            print("\nStarting sweep agent...")
            run_agent(sweep_id, args.project, args.count)

    elif args.agent:
        # Run agent with existing sweep
        run_agent(args.agent, args.project, args.count)

    else:
        parser.print_help()
        print("\nError: Must specify either --init or --agent SWEEP_ID")
        sys.exit(1)


if __name__ == '__main__':
    main()

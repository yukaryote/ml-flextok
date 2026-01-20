# Quick Start Guide - RL Search for FlexTok

This guide will get you up and running with RL-based token search in 5 minutes.

## Step 1: Install Dependencies

```bash
conda activate flextok  # or your FlexTok environment
pip install gymnasium stable-baselines3 tensorboard
```

Or install all at once:
```bash
pip install -r rl_search/requirements.txt
```

## Step 2: Verify Installation

```bash
python rl_search/test_installation.py
```

You should see:
```
✅ All required packages installed!
✅ All RL modules imported successfully!
✅ Environment creation test passed!
✅ Agent creation test passed!
🎉 All tests passed! RL search is ready to use.
```

## Step 3: Try the Interactive Notebook

```bash
cd notebooks
jupyter notebook rl_search_example.ipynb
```

Run through the cells to:
1. Load your FlexTok model
2. Create the environment
3. Train a DQN agent
4. Evaluate and visualize results

## Step 4: Train Your First Agent (Command Line)

```bash
python -m rl_search.train \
    --ckpt_path checkpoints/celeba_d18_arcface_fsq_8/20260108/checkpoint_best.pt \
    --target_image_path data/celebahq/val/00000.jpg \
    --agent_type dqn \
    --total_timesteps 50000 \
    --reward_mode improvement \
    --save_dir ./rl_checkpoints/my_first_agent
```

This will:
- Load your FlexTok model
- Load a target image
- Train a DQN agent for 50K timesteps (~15-30 min on GPU)
- Save checkpoints to `./rl_checkpoints/my_first_agent/`
- Log to TensorBoard

## Step 5: Monitor Training

While training is running, open another terminal:

```bash
tensorboard --logdir ./rl_checkpoints/my_first_agent/tensorboard
```

Open browser to http://localhost:6006 to see:
- Episode rewards
- Episode lengths
- Loss curves
- Exploration rate

## Step 6: Evaluate the Trained Agent

Use the evaluation utilities:

```python
from rl_search.agents import DQNAgent
from rl_search.eval_utils import evaluate_agent, visualize_episode

# Load trained agent
agent = DQNAgent(env)
agent.load('./rl_checkpoints/my_first_agent/dqn_final.zip')

# Evaluate
results = evaluate_agent(agent, env, n_episodes=10)
print(f"Mean best score: {results['mean_best_score']:.4f}")

# Visualize
visualize_episode(agent, env)
```

## Example Usage in Code

```python
from PIL import Image
import torch
from twenty_questions.twenty_questions import load_flextok_model
from dreamsim import dreamsim
from rl_search import FlexTokSearchEnv, create_agent
from rl_search.reward_shaping import RewardShaper

# 1. Load models
device = 'cuda'
model = load_flextok_model(
    ckpt_path='path/to/checkpoint.pt',
    fsq_level=[8]
).to(device)

dreamsim_model, preprocess = dreamsim(pretrained=True, device=device)

# 2. Load target image
target = Image.open('target.jpg').convert('RGB').resize((256, 256))

# 3. Create similarity function
def similarity_fn(img1, img2):
    t1 = preprocess(img1).unsqueeze(0).to(device)
    t2 = preprocess(img2).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(dreamsim_model(t1, t2).item())

# 4. Create environment
reward_shaper = RewardShaper(mode='improvement')
env = FlexTokSearchEnv(
    flextok_model=model,
    target_image=target,
    similarity_fn=similarity_fn,
    fsq_levels=[8],
    max_tokens=256,
    reward_shaper=reward_shaper,
)

# 5. Create and train agent
agent = create_agent('dqn', env, {
    'learning_rate': 1e-4,
    'buffer_size': 50000,
    'batch_size': 32,
    'verbose': 1,
})

agent.learn(total_timesteps=50000)

# 6. Use the trained agent
obs, _ = env.reset()
done = False
while not done:
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(f"Step {info['token_idx']}: Score {info.get('current_score', 0):.4f}")
```

## Common Issues

### Import Error: No module named 'gymnasium'
```bash
pip install gymnasium
```

### Import Error: No module named 'stable_baselines3'
```bash
pip install stable-baselines3
```

### CUDA out of memory
Reduce batch size:
```python
agent_config = {'batch_size': 16}  # Default is 32
```

### Training is slow
- Use GPU: `device='cuda'`
- Reduce total_timesteps: Start with 10K for testing
- Disable image observations: `image_obs=False` (default)

## Next Steps

1. **Read the full documentation**: [rl_search/README.md](README.md)
2. **Review implementation details**: [../RL_IMPLEMENTATION_SUMMARY.md](../RL_IMPLEMENTATION_SUMMARY.md)
3. **Experiment with different agents**: Try SAC or PPO
4. **Tune hyperparameters**: Learning rate, buffer size, etc.
5. **Compare with baselines**: Run greedy/beam search for comparison

## Tips for Best Results

1. **Start with DQN** for discrete FSQ [8]
2. **Use improvement-based rewards** for stable training
3. **Train for at least 50K timesteps** for reasonable performance
4. **Monitor TensorBoard** to track progress
5. **Evaluate on multiple images** to test generalization
6. **Compare with greedy search** as a baseline

## Help

If you encounter issues:
1. Run `python rl_search/test_installation.py` to check setup
2. Check the example notebook for working code
3. Review error messages carefully
4. Consult Stable-Baselines3 docs: https://stable-baselines3.readthedocs.io/

## Performance Expectations

After 50K timesteps of training with DQN:
- **Training time**: 15-30 minutes on GPU
- **Performance**: Should match or exceed greedy search
- **Best score**: Expect similarity < 0.3 for most images
- **Sample efficiency**: Better than PPO, comparable to SAC

Enjoy exploring RL-based token search! 🚀

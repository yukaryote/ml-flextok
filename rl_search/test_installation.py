"""
Test script to verify RL search installation.

Run this to check if all dependencies are installed correctly.
"""
import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")

    tests = {
        'gymnasium': 'gymnasium',
        'stable_baselines3': 'stable-baselines3',
        'torch': 'torch',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib.pyplot',
        'PIL': 'Pillow',
        'yaml': 'pyyaml',
    }

    failed = []
    for module_name, package_name in tests.items():
        try:
            if '.' in module_name:
                parts = module_name.split('.')
                mod = __import__(parts[0])
                for part in parts[1:]:
                    mod = getattr(mod, part)
            else:
                __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError as e:
            print(f"  ✗ {package_name}: {e}")
            failed.append(package_name)

    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        print(f"Install with: pip install {' '.join(failed)}")
        return False
    else:
        print("\n✅ All required packages installed!")
        return True


def test_rl_modules():
    """Test if RL search modules can be imported."""
    print("\nTesting RL search modules...")

    try:
        from rl_search import FlexTokSearchEnv, RewardShaper, create_agent
        print("  ✓ rl_search.environment")
        print("  ✓ rl_search.reward_shaping")
        print("  ✓ rl_search.agents")

        from rl_search.eval_utils import evaluate_agent, visualize_episode
        print("  ✓ rl_search.eval_utils")

        print("\n✅ All RL modules imported successfully!")
        return True
    except ImportError as e:
        print(f"\n❌ Failed to import RL modules: {e}")
        return False


def test_environment_creation():
    """Test if environment can be created."""
    print("\nTesting environment creation...")

    try:
        import torch
        from unittest.mock import MagicMock
        from PIL import Image
        import numpy as np
        from rl_search import FlexTokSearchEnv

        # Create mock objects
        mock_model = MagicMock()
        mock_model.decode_tokens = MagicMock(return_value=torch.randn(1, 3, 256, 256))

        target_image = Image.new('RGB', (256, 256))

        def mock_similarity(img1, img2):
            return 0.5

        # Try to create environment
        env = FlexTokSearchEnv(
            flextok_model=mock_model,
            target_image=target_image,
            similarity_fn=mock_similarity,
            fsq_levels=[8],
            max_tokens=10,
        )

        # Test reset
        obs, info = env.reset()
        print(f"  ✓ Environment created")
        print(f"  ✓ Observation space: {env.observation_space}")
        print(f"  ✓ Action space: {env.action_space}")

        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  ✓ Step executed (reward: {reward:.2f})")

        print("\n✅ Environment creation test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_creation():
    """Test if agents can be created."""
    print("\nTesting agent creation...")

    try:
        import torch
        from unittest.mock import MagicMock
        from PIL import Image
        from rl_search import FlexTokSearchEnv, create_agent

        # Create mock environment
        mock_model = MagicMock()
        mock_model.decode_tokens = MagicMock(return_value=torch.randn(1, 3, 256, 256))
        target_image = Image.new('RGB', (256, 256))

        env = FlexTokSearchEnv(
            flextok_model=mock_model,
            target_image=target_image,
            similarity_fn=lambda x, y: 0.5,
            fsq_levels=[8],
            max_tokens=10,
        )

        # Test DQN
        dqn_agent = create_agent('dqn', env, {'verbose': 0})
        print(f"  ✓ DQN agent created")

        # Test SAC (if action space is continuous)
        # For discrete, SAC won't work, so we skip

        # Test PPO
        ppo_agent = create_agent('ppo', env, {'verbose': 0})
        print(f"  ✓ PPO agent created")

        print("\n✅ Agent creation test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("RL Search Installation Test")
    print("="*60)

    results = []

    # Test imports
    results.append(("Package imports", test_imports()))

    # Test RL modules
    results.append(("RL modules", test_rl_modules()))

    # Test environment
    results.append(("Environment creation", test_environment_creation()))

    # Test agents
    results.append(("Agent creation", test_agent_creation()))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 All tests passed! RL search is ready to use.")
        print("\nNext steps:")
        print("  1. Check out notebooks/rl_search_example.ipynb")
        print("  2. Read rl_search/README.md for documentation")
        print("  3. Try training with: python -m rl_search.train --help")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nTo install missing packages:")
        print("  pip install -r rl_search/requirements.txt")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

"""
Quick verification script to ensure multi-bird training setup is working.
Run this to test that all components are properly installed and functioning.
"""

import sys


def check_imports():
    """Check if all required modules can be imported."""
    print("\n" + "="*70)
    print("CHECKING IMPORTS")
    print("="*70)
    
    modules = {
        'pygame': 'Game rendering',
        'numpy': 'Numerical operations',
        'torch': 'Deep learning (PyTorch)',
        'flappy_bird_env': 'Flappy Bird environment',
        'vector_env': 'Vectorized environment (NEW)',
        'bird_ai': 'DQN training functions',
        'main': 'Interactive menu',
    }
    
    all_ok = True
    for module, description in modules.items():
        try:
            __import__(module)
            print(f"✓ {module:20s} - {description}")
        except ImportError as e:
            print(f"✗ {module:20s} - MISSING! ({description})")
            print(f"  Error: {e}")
            all_ok = False
    
    return all_ok


def check_functions():
    """Check if required functions exist."""
    print("\n" + "="*70)
    print("CHECKING FUNCTIONS")
    print("="*70)
    
    try:
        from bird_ai import train_dqn, train_dqn_vectorized, DQN
        print("✓ train_dqn() - Single-bird training (original)")
        print("✓ train_dqn_vectorized() - Multi-bird training (NEW)")
        print("✓ DQN - Neural network class")
        return True
    except ImportError as e:
        print(f"✗ Failed to import training functions: {e}")
        return False


def check_vectorized_env():
    """Test the vectorized environment."""
    print("\n" + "="*70)
    print("TESTING VECTORIZED ENVIRONMENT")
    print("="*70)
    
    try:
        from vector_env import VectorFlappyBirdEnv
        import numpy as np
        
        print("Creating vectorized environment with 4 birds...")
        vec_env = VectorFlappyBirdEnv(num_envs=4, render_mode=False)
        print(f"✓ Environment created: {vec_env.num_envs} birds")
        
        print("Resetting environment...")
        states = vec_env.reset()
        print(f"✓ Reset successful, states shape: {states.shape}")
        
        print("Taking random actions...")
        actions = np.random.choice([0, 1], size=4)
        next_states, rewards, dones, infos = vec_env.step(actions)
        print(f"✓ Step successful")
        print(f"  - Next states shape: {next_states.shape}")
        print(f"  - Rewards: {rewards}")
        print(f"  - Dones: {dones}")
        
        vec_env.close()
        print("✓ Environment closed successfully")
        
        return True
    except Exception as e:
        print(f"✗ Vectorized environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_device():
    """Check if PyTorch can use GPU."""
    print("\n" + "="*70)
    print("CHECKING COMPUTE DEVICE")
    print("="*70)
    
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PyTorch device: {device}")
        
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print("\n💡 Recommendation: Use 16-32 birds for faster training")
        else:
            print("⚠ No GPU detected, using CPU")
            print("\n💡 Recommendation: Use 4-8 birds on CPU (slower but works)")
        
        return True
    except Exception as e:
        print(f"✗ Device check failed: {e}")
        return False


def quick_training_test():
    """Run a very quick training test (10 steps)."""
    print("\n" + "="*70)
    print("QUICK TRAINING TEST")
    print("="*70)
    print("Running a 10-step training test with 4 birds...")
    
    try:
        from bird_ai import train_dqn_vectorized, DQN, device, state_dim, action_dim
        from vector_env import VectorFlappyBirdEnv
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import numpy as np
        from collections import deque
        import random
        
        # Create small test environment
        vec_env = VectorFlappyBirdEnv(num_envs=4, render_mode=False)
        nn_model = DQN(state_dim, action_dim).to(device)
        optimizer = optim.Adam(nn_model.parameters(), lr=0.001)
        memory = deque(maxlen=100)
        
        state = vec_env.reset()
        print(f"✓ Initial states shape: {state.shape}")
        
        # Run 10 steps
        for step in range(10):
            actions = np.random.choice([0, 1], size=4)
            next_state, rewards, dones, infos = vec_env.step(actions)
            
            for i in range(4):
                memory.append((state[i], int(actions[i]), float(rewards[i]), next_state[i], bool(dones[i])))
            
            state = next_state
        
        print(f"✓ Completed 10 steps, collected {len(memory)} transitions")
        
        # Try one training step
        if len(memory) >= 32:
            batch = random.sample(memory, 32)
            states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*batch)
            
            states_b = torch.tensor(np.array(states_b), dtype=torch.float32).to(device)
            actions_b = torch.tensor(actions_b, dtype=torch.long).unsqueeze(1).to(device)
            
            q_values = nn_model(states_b).gather(1, actions_b)
            loss = q_values.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"✓ Training step successful, loss: {loss.item():.4f}")
        
        vec_env.close()
        print("✓ Quick training test PASSED!")
        
        return True
    except Exception as e:
        print(f"✗ Quick training test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("\n" + "="*70)
    print("MULTI-BIRD TRAINING SETUP VERIFICATION")
    print("="*70)
    print("\nThis script will verify that your multi-bird training setup is")
    print("working correctly. It will check imports, functions, and run")
    print("a quick test of the vectorized environment.")
    print("="*70)
    
    checks = [
        ("Imports", check_imports),
        ("Functions", check_functions),
        ("Compute Device", check_device),
        ("Vectorized Environment", check_vectorized_env),
        ("Quick Training", quick_training_test),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} check crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} - {name}")
    
    print("="*70)
    print(f"Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 SUCCESS! Your multi-bird training setup is ready!")
        print("\nYou can now run:")
        print("  python main.py")
        print("\nThen choose option 2 for multi-bird training.")
    else:
        print("\n⚠ WARNING! Some checks failed.")
        print("\nPlease fix the issues above before training.")
        print("Common fixes:")
        print("  - Run: pip install -r requirements.txt")
        print("  - Check that all files are in the correct directory")
        print("  - Make sure bird_ai.py was modified correctly")
    
    print("="*70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(1)

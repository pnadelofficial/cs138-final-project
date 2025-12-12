import gymnasium as gym
import bluesky_gym
from stable_baselines3 import TD3
import numpy as np
import time
from datetime import datetime

bluesky_gym.register_envs()

# Configuration
timesteps = 5e5  # 500k timesteps per environment
n_eval_episodes = 20

# All 7 BlueSky-Gym environments
environments = [
    'DescentEnv-v0',
    'VerticalCREnv-v0',
    'PlanWaypointEnv-v0',
    'HorizontalCREnv-v0',
    'SectorCREnv-v0',
    'StaticObstacleEnv-v0',
    'MergeEnv-v0'
]

print("TD3 TRAINING ON ALL 7 BLUESKY-GYM ENVIRONMENTS")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Timesteps per environment: {int(timesteps):,}")
print(f"Total environments: {len(environments)}")
print(f"Estimated total time: ~{len(environments) * 2.5:.1f} hours")

all_results = []
start_time_total = time.time()

# Train on each environment
for env_idx, env_name in enumerate(environments, 1):
    print(f"[{env_idx}/{len(environments)}] ENVIRONMENT: {env_name}")
    
    start_time_env = time.time()
    
    print(f"\nTraining TD3 on {env_name}")
    
    try:
        env = gym.make(env_name, render_mode=None)
        
        model = TD3(
            "MultiInputPolicy", 
            env, 
            verbose=1, 
            learning_rate=3e-4, 
            tensorboard_log="./tensorboard_logs/"
        )
        
        model.learn(
            total_timesteps=int(timesteps), 
            reset_num_timesteps=False, 
            tb_log_name=f"TD3_{env_name}"
        )
        
        model_path = f"td3_{env_name.lower()}"
        model.save(model_path)
        env.close()
        
        training_time = time.time() - start_time_env
        print(f"\nTraining complete! Time: {training_time/60:.1f} minutes")
        print(f"\nEvaluating TD3 on {env_name}")
        
        env = gym.make(env_name, render_mode=None)
        model = TD3.load(model_path, env=env)
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_eval_episodes):
            obs, info = env.reset()
            done = truncated = False
            episode_reward = 0
            episode_length = 0
            
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        env.close()
        
        # Store results
        result = {
            'environment': env_name,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'training_time_min': training_time / 60
        }
        all_results.append(result)
        
        print(f"\nResults for {env_name}:")
        print(f"   Mean Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"   Mean Length: {result['mean_length']:.2f} ± {result['std_length']:.2f}")
        print(f"   Training Time: {result['training_time_min']:.1f} minutes")
        
    except Exception as e:
        print(f"\nERROR on {env_name}: {str(e)}")
        all_results.append({
            'environment': env_name,
            'error': str(e)
        })
    
    # Progress update
    elapsed_total = time.time() - start_time_total
    remaining_envs = len(environments) - env_idx
    avg_time_per_env = elapsed_total / env_idx
    estimated_remaining = remaining_envs * avg_time_per_env
    
    print(f"\n⏱Progress: {env_idx}/{len(environments)} environments complete")
    print(f"   Elapsed time: {elapsed_total/3600:.2f} hours")
    print(f"   Estimated remaining: {estimated_remaining/3600:.2f} hours")

total_time = time.time() - start_time_total

print("FINAL RESULTS - TD3 ON ALL ENVIRONMENTS (1M TIMESTEPS)")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total time: {total_time/3600:.2f} hours")
print(f"{'Environment':<25} {'Mean Reward':<20} {'Mean Length':<20}")

for result in all_results:
    if 'error' in result:
        print(f"{result['environment']:<25} ERROR: {result['error']}")
    else:
        reward_str = f"{result['mean_reward']:>7.2f} ± {result['std_reward']:<5.2f}"
        length_str = f"{result['mean_length']:>7.2f} ± {result['std_length']:<5.2f}"
        print(f"{result['environment']:<25} {reward_str:<20} {length_str:<20}")


# Save results to file
import json
with open('td3_all_environments_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\nResults saved to: td3_all_environments_results.json")

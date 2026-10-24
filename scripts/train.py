import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from robot_simulator.envs import RobotNavEnv


def main():
    os.makedirs("models", exist_ok=True)
    
    env = RobotNavEnv(randomize=True)
    env = Monitor(env)
    timesteps = 200_000

    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    print(f"Starting training for {timesteps} steps...")
    model.learn(total_timesteps=timesteps)

    model_name = "models/ppo_robot_nav_smooth"
    model.save(model_name)
    print(f"Model saved to {model_name}.zip")


if __name__ == '__main__':
    main()
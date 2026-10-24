import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from robot_simulator.envs import RobotNavEnv


def main():
    os.makedirs("models", exist_ok=True)
    
    env = RobotNavEnv()
    env = Monitor(env)

    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    print("Starting training for 100,000 steps...")
    model.learn(total_timesteps=100_000)

    model.save("models/ppo_robot_nav")
    print("Model saved to models/ppo_robot_nav.zip")


if __name__ == '__main__':
    main()
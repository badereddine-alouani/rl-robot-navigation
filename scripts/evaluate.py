from stable_baselines3 import PPO
from robot_simulator.envs import RobotNavEnv
from robot_simulator.simulation import plot_trajectory
from robot_simulator.obstacles import CircularObstacle
import numpy as np

env = RobotNavEnv(randomize=True)
model = PPO.load("models/ppo_robot_nav")
obs, info = env.reset()
x_history = [env.robot_x]
y_history = [env.robot_y]
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    x_history.append(env.robot_x)
    y_history.append(env.robot_y)
    done = terminated or truncated

print(f"Evaluation finished! Outcome: {'Goal Reached!' if env.prev_distance <= env.goal_tolerance else 'Did not reach goal'}")
plot_trajectory(x_history, y_history, env.goal, env.obstacles, theta=env.robot_theta)
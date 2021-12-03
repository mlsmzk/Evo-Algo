import gym
env = gym.make('MountainCar-v0')
for i_episode in range(5):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
    
        if done:
            print(f"Episode finished after {t+1} timesteps")
            break
env.close()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
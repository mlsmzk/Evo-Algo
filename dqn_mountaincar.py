import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque
import gym

class DQN:
    def __init__(self,env):
        self.memory = deque(maxlen=200)
        self.env = env

        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.tau = .125

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential() 
        state_shape = self.env.observation_space.shape
        model.add(Dense(2,input_dim = state_shape[0], activation="relu"))
        model.add(Dense(3, activation="relu"))
        model.add(Dense(4, name="layer3"))
        model.compile(loss='mean_squared_error',
            optimizer = Adam(learning_rate = self.learning_rate))
        return model
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state,action,reward,new_state,done])
    
    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return
        
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def act(self,state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

def main():
    env = gym.make("MountainCar-v0")
    gamma = 0.9
    epsilon = 0.95

    trials = 100
    trial_len = 500

    updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    steps = []
    for trial in range(trials):
        cur_state = env.reset().reshape(1,2)
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            env.render()
            new_state, reward, done, _ = env.step(action)

            reward = reward if not done else -20
            print(reward)
            new_state = new_state.reshape(1,2)
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()
            dqn_agent.target_train()

            cur_state = new_state
            if done:
                break
        if step >= 199:
            print('Failed to complete trial')
        else:
            print(f"Completed in {trial} trials")
            break

if __name__ == "__main__":
    main()
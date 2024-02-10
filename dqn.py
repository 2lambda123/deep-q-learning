# -*- coding: utf-8 -*-
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import secrets

EPISODES = 1000

class DQNAgent:
    def __init__(self, state_size, action_size):
        """This function initializes the parameters for the agent's neural network model and sets the exploration and discount rates.
        Parameters:
            - state_size (int): The size of the state space.
            - action_size (int): The size of the action space.
        Returns:
            - None: This function does not return any values.
        Processing Logic:
            - Sets the state and action sizes.
            - Initializes a memory deque.
            - Sets the discount and exploration rates.
            - Sets the learning rate.
            - Builds the neural network model."""
        
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        """Builds a neural network model for deep Q-learning.
        Parameters:
            - self (object): The object that calls the function.
        Returns:
            - model (object): The compiled neural network model.
        Processing Logic:
            - Add 2 dense layers with 24 nodes each.
            - Use ReLU activation function.
            - Add a final dense layer with the number of actions as nodes.
            - Use linear activation function.
            - Compile the model with mean squared error loss and Adam optimizer.
            - Return the compiled model."""
        
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        """"Stores the state, action, reward, next_state, and done in a memory list."
        Parameters:
            - state (object): The current state of the environment.
            - action (object): The action taken by the agent.
            - reward (float): The reward received for taking the action.
            - next_state (object): The resulting state after taking the action.
            - done (bool): Indicates if the episode is complete.
        Returns:
            - None: This function does not return anything.
        Processing Logic:
            - Adds the state, action, reward, next_state, and done to the memory list."""
        
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """"Performs an action based on the given state and returns the chosen action. If the random number generated is less than or equal to the epsilon value, a random action is chosen. Otherwise, the model predicts the action values for the given state and returns the action with the highest value.
        Parameters:
            - state (array): The current state of the environment.
        Returns:
            - action (int): The chosen action based on the given state.
        Processing Logic:
            - Randomly choose an action with probability epsilon.
            - Predict the action values for the given state.
            - Return the action with the highest value.
            - If multiple actions have the same value, the first one is returned.""""
        
        if np.random.rand() <= self.epsilon:
            return secrets.SystemRandom().randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        """Returns:
            - None: No return value.
        Processing Logic:
            - Randomly sample a minibatch.
            - Calculate target based on reward.
            - Update target based on next state.
            - Fit model with state and target.
            - Update epsilon value if applicable."""
        
        minibatch = secrets.SystemRandom().sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """Loads weights for a given model.
        Parameters:
            - name (str): Name of the weights file to be loaded.
        Returns:
            - None: Does not return any value.
        Processing Logic:
            - Load weights from given file.
            - Use self.model to access model.
            - Use .load_weights() method.
            - Use name parameter as input."""
        
        self.model.load_weights(name)

    def save(self, name):
        """Saves the weights of the model.
        Parameters:
            - name (str): Name of the file to save the weights to.
        Returns:
            - None: Does not return anything.
        Processing Logic:
            - Saves the weights of the model.
            - Uses the name provided to save the weights.
            - Does not return anything.
            - Only works for models."""
        
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")

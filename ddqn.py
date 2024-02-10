# -*- coding: utf-8 -*-
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf
import secrets

EPISODES = 5000

class DQNAgent:
    def __init__(self, state_size, action_size):
        """Function to initialize the Deep Q-Network agent.
        Parameters:
            - state_size (int): The dimension of the state space.
            - action_size (int): The dimension of the action space.
        Returns:
            - None
        Processing Logic:
            - Initialize the agent's memory.
            - Set the discount rate, exploration rate, minimum exploration rate, and exploration rate decay.
            - Set the learning rate.
            - Build the agent's model and target model.
            - Update the target model."""
        
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    """Huber loss for Q Learning

    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        """Calculates the Huber loss between y_true and y_pred.
        Parameters:
            - y_true (tensor): True values.
            - y_pred (tensor): Predicted values.
            - clip_delta (float): Threshold for clipping the error.
        Returns:
            - tensor: Huber loss between y_true and y_pred.
        Processing Logic:
            - Calculate the absolute error.
            - Check if the error is within the threshold.
            - Calculate the squared loss if within threshold.
            - Calculate the quadratic loss if outside threshold.
            - Return the mean of the two losses."""
        
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        """Builds a neural network model for deep Q-learning.
        Parameters:
            - self (object): The object calling the function.
        Returns:
            - model (object): The built neural network model.
        Processing Logic:
            - Create a sequential model.
            - Add two dense layers with ReLU activation.
            - Add a dense layer with linear activation.
            - Compile the model with a custom loss function and Adam optimizer."""
        
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        """Copies weights from model to target_model.
        Parameters:
            - self (class): The class object.
        Returns:
            - None: No return value.
        Processing Logic:
            - Copy weights from model to target_model."""
        
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        """This function adds the given state, action, reward, next_state, and done values to the memory list.
        Parameters:
            - state (object): The current state of the environment.
            - action (object): The action taken in the current state.
            - reward (float): The reward received for taking the action.
            - next_state (object): The resulting state after taking the action.
            - done (bool): Indicates if the episode is complete.
        Returns:
            - None: This function does not return any values.
        Processing Logic:
            - Appends the given values to the memory list.
            - Memory list stores tuples of (state, action, reward, next_state, done).
            - Used for experience replay in reinforcement learning.
            - Helps improve learning by breaking correlations between consecutive samples."""
        
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """"Returns an action based on the given state using epsilon-greedy algorithm.
        Parameters:
            - state (numpy array): Current state of the environment.
        Returns:
            - action (int): Action to be taken by the agent.
        Processing Logic:
            - Randomly selects an action with probability epsilon.
            - Otherwise, uses the model to predict the best action.
            - Returns the index of the highest predicted action.
            - Uses numpy.argmax() to handle multiple actions.
            - Uses secrets.SystemRandom() for secure random number generation.""""
        
        if np.random.rand() <= self.epsilon:
            return secrets.SystemRandom().randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        """Docstring:
        Replays a batch of experiences from the agent's memory and updates the model's weights based on the Bellman equation.
        Parameters:
            - batch_size (int): Number of experiences to replay.
        Returns:
            - None: This function does not return any values.
        Processing Logic:
            - Sample experiences from memory randomly.
            - Update the target value for the chosen action.
            - Fit the model to the updated target value.
            - Update the exploration rate.
        Example:
            replay(32)"""
        
        minibatch = secrets.SystemRandom().sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """Loads the weights of a model.
        Parameters:
            - name (str): Name of the file containing the weights to be loaded.
        Returns:
            - None: The function does not return anything.
        Processing Logic:
            - Load weights from file.
            - Update the model with the loaded weights."""
        
        self.model.load_weights(name)

    def save(self, name):
        """Saves the weights of the model.
        Parameters:
            - name (str): Name of the file to save the weights to.
        Returns:
            - None: No return value.
        Processing Logic:
            - Save weights to specified file.
            - Model must be trained before saving.
            - Name must include file extension.
            - Example: save("model_weights.h5")"""
        
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-ddqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            #reward = reward if not done else -10
            x,x_dot,theta,theta_dot = next_state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2
            
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-ddqn.h5")

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
        """Function:
        Initializes the Deep Q-Network Agent.
        Parameters:
            - state_size (int): Number of dimensions in the state space.
            - action_size (int): Number of possible actions.
        Returns:
            - None: This function does not return any value.
        Processing Logic:
            - Initializes memory buffer.
            - Sets discount rate, exploration rate, minimum exploration rate, and exploration rate decay.
            - Sets learning rate.
            - Builds the Deep Q-Network model."""
        
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
            - self (object): The object calling the function.
        Returns:
            - model (object): The built neural network model.
        Processing Logic:
            - Uses the Sequential() function to create a sequential model.
            - Adds 3 dense layers with 24, 24, and self.action_size nodes respectively.
            - Uses 'relu' activation function for the first two layers and 'linear' for the last layer.
            - Compiles the model using 'mse' loss function and Adam optimizer with learning rate self.learning_rate."""
        
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        """"Adds the given state, action, reward, next_state, and done to the memory list."
        Parameters:
            - state (any): The current state of the environment.
            - action (any): The action taken in the current state.
            - reward (any): The reward received for taking the action.
            - next_state (any): The resulting state after taking the action.
            - done (bool): Indicates if the episode is finished.
        Returns:
            - None: This function does not return anything.
        Processing Logic:
            - Appends the given state, action, reward, next_state, and done to the memory list."""
        
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Purpose:
            This function chooses an action based on the given state, using the epsilon-greedy algorithm.
        Parameters:
            - state (array): The current state of the environment.
        Returns:
            - action (int): The chosen action based on the state.
        Processing Logic:
            - Choose random action with probability epsilon.
            - Predict action values using model.
            - Return action with highest value."""
        
        if np.random.rand() <= self.epsilon:
            return secrets.SystemRandom().randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        """Replays a batch of experiences from the agent's memory and trains the neural network model.
        Parameters:
            - batch_size (int): Number of experiences to replay from the agent's memory.
        Returns:
            - loss (float): The loss value from training the neural network model.
        Processing Logic:
            - Randomly samples a batch of experiences from the agent's memory.
            - Calculates the target value for each experience using the Bellman equation.
            - Updates the target value for the chosen action in the neural network model.
            - Stores the states and corresponding target values for training.
            - Trains the neural network model for one epoch using the stored states and target values.
            - Updates the agent's exploration rate.
            - Returns the loss value from training the neural network model."""
        
        minibatch = secrets.SystemRandom().sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target 
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
        history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def load(self, name):
        """Loads weights for a model.
        Parameters:
            - name (str): Name of the weights file to be loaded.
        Returns:
            - None: No return value.
        Processing Logic:
            - Load weights from file.
            - Use name to specify file.
            - Weights are for a model."""
        
        self.model.load_weights(name)

    def save(self, name):
        """Saves the weights of the model to the specified file name.
        Parameters:
            - name (str): The name of the file to save the weights to.
        Returns:
            - None: The function does not return anything.
        Processing Logic:
            - Save model weights to file.
            - Use specified file name.
            - No return value.
            - 1 parameter."""
        
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
                loss = agent.replay(batch_size)
                # Logging training loss every 10 timesteps
                if time % 10 == 0:
                    print("episode: {}/{}, time: {}, loss: {:.4f}"
                        .format(e, EPISODES, time, loss))  
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")

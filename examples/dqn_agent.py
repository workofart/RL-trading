"""
In this example we demonstrate how to implement a DQN agent and
train it to trade optimally on a periodic price signal.
Training time is short and results are unstable.
Do not hesitate to run several times and/or tweak parameters to get better results.
Inspired from https://github.com/keon/deep-q-learning
"""
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import json
from keras.layers import Dense
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
from envs.trading_spread import SpreadTrading
from gens.deterministic import WavySignal, RandomGenerator
from gens.csvstream import CSVStreamer
from envs.trading_tick import TickTrading


plt.interactive(False)


class DQNAgent:
    def __init__(self,
                 state_size,
                 episodes,
                 episode_length,
                 action_size=len(TickTrading._actions),
                 memory_size=2000,
                 train_interval=100,
                 gamma=0.95,
                 learning_rate=0.001,
                 batch_size=64,
                 epsilon_min=0.01
                 ):
        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = memory_size
        self.memory = [None] * memory_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decrement = (self.epsilon - epsilon_min)\
            * train_interval / (episodes * episode_length)  # linear decrease rate
        self.learning_rate = learning_rate
        self.train_interval = train_interval
        self.batch_size = batch_size
        self.brain = self._build_brain()
        self.i = 0

    def exportModel(self):
        # serialize model to JSON
        model_json = self.brain.to_json()
        with open("../models/model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.brain.save_weights("../models/model.h5")
        print("Saved model to disk")

    def importModel(self):
        json_file = open('../models/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.brain = model_from_json(loaded_model_json)
        self.brain.load_weights("model.h5")
        print("Loaded model from disk")

    def _build_brain(self):
        """Build the agent's brain
        """
        brain = Sequential()
        neurons_per_layer = 24
        activation = "relu"
        brain.add(Dense(neurons_per_layer,
                        input_dim=self.state_size,
                        activation=activation))
        brain.add(Dense(neurons_per_layer, activation=activation))
        brain.add(Dense(self.action_size, activation='linear'))
        brain.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return brain

    def act(self, state):
        """Acting Policy of the DQNAgent
        """
        action = np.zeros(self.action_size)
        if np.random.rand() == self.epsilon:
            action[random.randrange(self.action_size)] = 1
        else:
            state = state.reshape(1, self.state_size)
            act_values = self.brain.predict(state)
            action[np.argmax(act_values[0])] = 1
        return action

    def observe(self, state, action, reward, next_state, done, warming_up=False):
        """Memory Management and training of the agent
        """
        self.i = (self.i + 1) % self.memory_size
        self.memory[self.i] = (state, action, reward, next_state, done)
        if (not warming_up) and (self.i % self.train_interval) == 0:
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decrement
            state, action, reward, next_state, done = self._get_batches()
            reward += (self.gamma
                       * np.logical_not(done)
                       * np.amax(self.brain.predict(next_state),
                                 axis=1))
            q_target = self.brain.predict(state)
            q_target[action[0], action[1]] = reward
            # print('Expected value: {} | Actual value: {}'.format(q_target, state))
            return self.brain.fit(state, q_target,
                                  batch_size=self.batch_size,
                                  epochs=1,
                                  verbose=False)

    def _get_batches(self):
        """Selecting a batch of memory
           Split it into categorical subbatches
           Process action_batch into a position vector
        """
        batch = np.array(random.sample(self.memory, self.batch_size))
        state_batch = np.concatenate(batch[:, 0])\
            .reshape(self.batch_size, self.state_size)
        action_batch = np.concatenate(batch[:, 1])\
            .reshape(self.batch_size, self.action_size)
        reward_batch = batch[:, 2]
        next_state_batch = np.concatenate(batch[:, 3])\
            .reshape(self.batch_size, self.state_size)
        done_batch = batch[:, 4]
        # action processing
        action_batch = np.where(action_batch == 1)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


def getDataGenerator(gen_type):
    # Instantiating the environmnent
    if gen_type == 'W':
        generator = WavySignal(period_1=25, period_2=50,
                               epsilon=0, ba_spread=0)
    elif gen_type == 'R':
        generator = RandomGenerator(
            spread=0.0001, range_low=1.0, range_high=2.0)
    elif gen_type == 'C':
        filename = r'nw.csv'
        generator = CSVStreamer(filename=filename)
    return generator


def trainAgent(episodes, environment, agent):
    # Training the agent
    for ep in range(episodes):
        ms = time.time()
        state = environment.reset()
        rew = 0
        for _ in range(episode_length):
            action = agent.act(state)
            next_state, reward, done, _ = environment.step(action)
            loss = agent.observe(state, action, reward, next_state, done)
            # environment.render()
            state = next_state
            rew += reward
        # print(loss)
        print("Ep:" + str(ep)
              + "| rew:" + str(round(rew, 2))
              + "| eps:" + str(round(agent.epsilon, 2))
              # + "| loss:" + str(round(loss.history["loss"][0], 8))
              + "| runtime:" + str(time.time() - ms))


def envSetup(trading_type, generator, trading_fee, time_fee, history_length, episode_length, profit_taken, stop_loss):
    if trading_type == 'T':
        environment = TickTrading(data_generator=generator,
                                  trading_fee=trading_fee,
                                  time_fee=time_fee,
                                  history_length=history_length,
                                  episode_length=episode_length,
                                  profit_taken=profit_taken,
                                  stop_loss=stop_loss)
        action_size = len(TickTrading._actions)
    else:
        environment = SpreadTrading(spread_coefficients=[1],
                                    data_generator=generator,
                                    trading_fee=trading_fee,
                                    time_fee=time_fee,
                                    history_length=history_length,
                                    episode_length=episode_length)
        action_size = len(SpreadTrading._actions)
    print('Completed Environment Setup for Trading Type: [%c]' % trading_type)
    return environment, action_size


def warmupAgent(memory_size, state):
    # Warming up the agent
    for _ in range(memory_size):
        action = agent.act(state)
        next_state, reward, done, _ = environment.step(action)
        agent.observe(state, action, reward, next_state, done, warming_up=True)
    print('completed mem allocation: ', time.time() - start)


def runAgent(environment, agent):
    # Running the agent
    done = False
    state = environment.reset()
    while not done:
        action = agent.act(state)
        state, _, done, info = environment.step(action)
        if 'status' in info and info['status'] == 'Closed plot':
            done = True
        else:
            environment.render()

if __name__ == "__main__":
    with open('../config/config.json') as f:
        config = json.load(f)

    agentConfig = config['agent']
    envConfig = config['env']
    runConfig = config['run']

    start = time.time()
    pluck = lambda dict, *args: (dict[arg] for arg in args)

    episodes, \
    episode_length, \
    trading_fee, \
    time_fee, \
    history_length, \
    profit_taken, \
    stop_loss = pluck(envConfig,
                      'episodes',
                      'episode_length',
                      'trading_fee',
                      'time_fee',
                      'history_length',
                      'profit_taken',
                      'stop_loss')

    memory_size,\
    gamma,\
    epsilon_min,\
    batch_size,\
    train_interval,\
    learning_rate = pluck(agentConfig,
                          'memory_size',
                          'gamma',
                          'epsilon_min',
                          'batch_size',
                          'train_interval',
                          'learning_rate')

    render_show, useExistingModel, trading_type, dataGenType = pluck(runConfig, 'renderGraph', 'useExistingModel', 'tradingType', 'dataGenType')

    generator = getDataGenerator(dataGenType)
    environment, action_size = envSetup(trading_type, generator, trading_fee, time_fee, history_length, episode_length, profit_taken, stop_loss)
    state = environment.reset()
    # Instantiating the agent
    state_size = len(state)

    agent = DQNAgent(state_size=state_size,
                     action_size=action_size,
                     memory_size=memory_size,
                     episodes=episodes,
                     episode_length=episode_length,
                     train_interval=train_interval,
                     gamma=gamma,
                     learning_rate=learning_rate,
                     batch_size=batch_size,
                     epsilon_min=epsilon_min)

    if not useExistingModel:
        warmupAgent(memory_size, state)
        trainAgent(episodes, environment, agent)
        agent.exportModel()
    else:
        agent.importModel()

    runAgent(environment, agent)

    print("All done:", str(time.time()-start))

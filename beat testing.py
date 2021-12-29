

import tensorflow as tf
import gym
from gym.wrappers import Monitor
from collections import deque
import tensorflow as tf
import numpy as np
import random
import math
import time
import glob
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
from Environment import beat_detection

env = beat_detection() # ENV Instance Call

num_features = env.observation_space.shape[0]  # NN_Input Features 
num_actions = env.action_space.n  # NN_Output Desired work to perform by the RL agent
print('Number of state features: {}'.format(num_features))
print('Number of possible actions: {}'.format(num_actions))


class DQN(tf.keras.Model):
  #Dense neural network class
  def __init__(self):
    super(DQN, self).__init__()
    self.dense1 = tf.keras.layers.Dense(32, activation="relu") # 32, 64, 128, 612
    self.dense2 = tf.keras.layers.Dense(32, activation="relu") # Hyperparameter Tunning 
    self.dense3 = tf.keras.layers.Dense(32, activation="relu")
    self.dense4 = tf.keras.layers.Dense(32, activation="relu")
    self.reshape = tf.keras.layers.Reshape((32,1), input_shape=(32,))
    self.lstm =tf.keras.layers.LSTM(64) # LSTM 
    self.dense5 = tf.keras.layers.Dense(32, activation="relu") # LSTM 

    self.dense6 = tf.keras.layers.Dense(num_actions, dtype=tf.float32) # No activation # Linear activation
    
  def call(self, x): # X = input data set  
    #Forward pass.
    x = self.dense1(x) # Input coming from outside

    x = self.dense2(x) # Input from previous layer
    x = self.dense3(x)
    x = self.dense4(x)
    x = self.reshape(x)
    x = self.lstm(x)
    x = self.dense5(x)
    
    return self.dense6(x)

main_nn = DQN() # Actual network that is learning 

target_nn = DQN() # Helper network 

optimizer = tf.keras.optimizers.Adam(1e-5) # SGD , 1e-6,  
mse = tf.keras.losses.MeanSquaredError()   # MSE => No other option, Linaer output 

main_nn.build(input_shape=(None,num_features)) # input shape(batch_size, features)
target_nn.build(input_shape=(None,num_features)) # None bacth size mean you can feed arbiterary batch

main_nn.load_weights("TF_MODEL_Conditional_7.h5")
#target_nn.load_weights("TF_MODEL_Conditional_7.h5")
#main_nn.load_weights("TF_MODEL_Conditional.h5")
# Data set of the DQN Algorithm (State, action, reward, next_state, done)
class ReplayBuffer(object):
  
  def __init__(self, size):
    self.buffer = deque(maxlen=size) # Deque 

  def add(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))

  def __len__(self):
    return len(self.buffer)

  def sample(self, num_samples): # batch size = num_samples 
    states, actions, rewards, next_states, dones = [], [], [], [], []
    idx = np.random.choice(len(self.buffer), num_samples) # LSTM --> Episodic sequence maintain
    for i in idx:
      elem = self.buffer[i]
      state, action, reward, next_state, done = elem
      states.append(state)
      actions.append(action)
      rewards.append(reward)
      next_states.append(next_state)
      dones.append(done)


    states = np.array(states, dtype=np.float32) 
    actions = np.array(actions, dtype=np.int32) # It must be in integer type
    rewards = np.array(rewards, dtype=np.float32)  
    next_states = np.array(next_states, dtype=np.float32)
    dones = np.array(dones, dtype=np.float32) # We have to conver it to float otherwise it will not work
    return states, actions, rewards, next_states, dones


# In[5]:
 # Exploration  ==> Random actions are taken by RL agent //training
    # Initially epsilon is higher 

 # Exploitation ==> Deterministic actions (NN's extimated actions) //testing
    # Epsilon is minimum - 0 
def select_epsilon_greedy_action(state, epsilon):
 
  result = tf.random.uniform((1,))
  if result < epsilon:          
    return env.action_space.sample() # Random action (left or right).
  else:
    return tf.argmax(main_nn(state)[0]).numpy() # Greedy action for state.

@tf.function
def train_step(states, actions, rewards, next_states, dones):
 
  # Calculate targets.
  next_qs = target_nn(next_states)
  max_next_qs = tf.reduce_max(next_qs, axis=-1) # We want to maximize the reward
  target = rewards + (1. - dones) * discount * max_next_qs # Learning Y data
  with tf.GradientTape() as tape:
    qs = main_nn(states)  
    action_masks = tf.one_hot(actions, num_actions)  # [1,1] == [1,0] // [0,1]
    masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)  #   
    loss = mse(target, masked_qs) # (target - predicted)
  grads = tape.gradient(loss, main_nn.trainable_variables)
  optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))
  return loss


# Hyperparameters.
# Full dataset modifications 
num_episodes = 1 
number_songs = 1  

epsilon = 1.0
start = 0.89
end = 0.001
decay1 = 0.06
decay2 = 0.4



batch_size = 32
discount = 0.98 # 1.0 Instant reward is important 
buffer = ReplayBuffer(100000)
cur_frame = 0
min_reward = 1360

# Start training. Play game once and then train with a batch.
last_100_ep_rewards = []
loss_value = []
loss_value_1 = []
rew = []
ep_reward = 0 
correctList = []

epsilonList=[]

song_number = 121
for  episode in range (num_episodes):

      done = False
      env.data.get_beats_and_features(song_number)
      print("Current Song Number", song_number )
      state = env.reset()
      ep_reward, done = 0, False
      loss = 0
      
      for t in range(1723):
        state_in = tf.expand_dims(state, axis=0) # (None, Features) (features[None,...])
        action = select_epsilon_greedy_action(state_in, 0.0)
        #action = np.random.randint(0,2)
        next_state, reward, done, info = env.step(action, t)
        ep_reward += reward
        # Save to experience replay.
        state = next_state
        #print("Step Reward", reward)
print("RL Agent Reward from Song      : ",ep_reward)
print("Actual Beat Count from Dataset : ", len(env.data.beats))


print("Beats in the Dataset: ", env.data.beats)

print("\nRL Predicted Beats: ", env.predicted_beat)

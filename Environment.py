# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 15:06:35 2021

@author: Suleman_Sahib
"""




import os
import numpy as np
from tqdm.notebook import tqdm
import librosa
import gym
from gym import spaces, Env


audioFiles="SMC_MIREX/SMC_MIREX_Audio"    
annotations="SMC_MIREX/SMC_MIREX_Annotations_05_08_2014"
class data_:
  def __init__(self):
    self.features = []
    self.beats = []
    
    self.all_wav = []
    self.all_anno =[]
    FrameNumbers=[]
    for root, dirnames, filenames in os.walk(audioFiles):
        self.all_wav += [os.path.join(root, f) for f in filenames if f[-4:] == '.wav']
    for root, dirnames, filenames in os.walk(annotations):
        self.all_anno+= [os.path.join(root, f) for f in filenames if f[-4:] == '.txt']

    self.all_wav.sort()
    self.all_anno.sort()
  def get_beats_and_features(self,song_number):
    #y,sr= librosa.load(os.path.join(LOC,files[0]))
    feature_loc = self.all_wav[song_number]
    groud_truth_loc = self.all_anno[song_number]
    y,sr= librosa.load(feature_loc)
    self.features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    #onset_env = librosa.onset.onset_strength(y, sr=sr,
    #                                     aggregate=np.median)
    #tempo, self.beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    self.beats = librosa.time_to_frames(np.loadtxt(groud_truth_loc, dtype=float)) # updating the ground truth
    
    
  def feature_at_t(self, t):
    feat = []
    for f in self.features:
      feat.append(f[t])
    return feat #self.features[t]
  def beat_at_t(self,t):
    if t in self.beats:
      return True
    else:
      return False
#data = data_()
#data.get_beats_and_features(0)
#d = data.features[0]
#print(d)

class beat_detection(Env):
  def __init__(self):
    self.action_space = spaces.Discrete(2)
    self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape=(40,), dtype = np.float32)
    self.data = data_()
    self.data.get_beats_and_features(0)

    self.predicted_beat = []
    self.predicted_non_beat = [] 

  def step(self, action, t):
    reward = 0
    if action == 0:  # NON BEAT
      self.predicted_non_beat.append(t)
      
      if t not in self.data.beats:
        reward += 0 #1 # Correct detection Reward
      else:
        reward += -1 # Wrong detection Penelaise

    else:  #Beat
      
      if t in self.data.beats:
        reward += 1 # Correct detection Reward
        self.predicted_beat.append(t)
      else:
        reward += 0 # Wrong detection Reward
    # True Positive(Actual Beat, Detected Beat), False Positive(Actual Non Beat, Detected Beat) 
    # True Negative (Actual NON-Beat, Detected Beat) , False Negative (Actual Beat, Detected NON-Beat)
    info = [self.predicted_beat, self.predicted_non_beat]
    if t >= 1722:
      done = True
      next_state = np.zeros(40)
    else:
      done = False
      next_state = self.next_state(t)  

    return next_state , reward, done, info
    
  def next_state(self, t):
    return self.data.feature_at_t(t+1)
  def reset(self):
    return self.data.feature_at_t(0)
"""
env = beat_detection()
env.reset()
 
for i in range(200):
    env.data.get_beats_and_features(i)
    print("Song_Number : ", i)
    env.reset()
    done = False
    for t in range(1725):
      action = env.action_space.sample()
      next_state, reward, done, info = env.step(action, t)
      #print(len(next_state), t)
      if done:
        break 
"""
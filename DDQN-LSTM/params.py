#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 21:19:09 2017

@author: shengx
"""
import tensorflow as tf
import numpy as np
import random
import gym
import pickle
import os




Params = {

        'LEARNING_RATE': 0.00025,
        
        'BATCH_SIZE':  32,
        
        'REWARD_DISCOUNT': 0.99,
        
        'RANDOM_ACTION_PROB_START': 0.9,
        
        'RANDOM_ACTION_PROB_END': 0.1,
        
        'ANNEALING_STEP': 50000,
        
        'FRAME_SKIP': 2,
        
        'SYNC_FREQ': 2000,
        
        'UPDATE_FREQ': 4,
        
        'SAVE_FREQ': 200,
        
        'MEMORY_BUFFER_SIZE': 200,
        
        'SAVE_PATH': './log/',
        
        'IMG_X': 105,
        
        'IMG_Y': 80,
        
        'IMG_Z': 4,
        
        'RNN_H_UNIT': 512,
        
        'RNN_SEQUENCE_LENGTH': 8}

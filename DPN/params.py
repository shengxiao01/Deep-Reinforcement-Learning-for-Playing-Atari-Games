#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 21:19:09 2017

@author: shengx
"""
import tensorflow as tf
import numpy as np



Params = {

        'GAME': 'Pong-v0',

        'LEARNING_RATE': 0.00025,
        
        'REWARD_DISCOUNT': 0.99,
        
        'FRAME_SKIP': 2,
        
        'SYNC_FREQ': 2000,
        
        'UPDATE_FREQ': 4,
        
        'SAVE_FREQ': 2000,
        
        'SAVE_PATH': './log/',
        
        'IMG_X': 105,
        
        'IMG_Y': 80,
        
        'IMG_Z': 1,
        
        'ENTROPY_PENALTY': 0,
        
        'MIN_POLICY': 0.02}

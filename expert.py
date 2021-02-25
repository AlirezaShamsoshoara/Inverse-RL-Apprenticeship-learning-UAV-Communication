"""
#################################
# Expert Operation functions
#################################
"""

#########################################################
# import libraries
import numpy as np
from random import randint
from config import Config_IRL
import matplotlib.pyplot as plt

#########################################################
# General Parameters
num_features = Config_IRL.get('NUM_FEATURES')
num_trajectories = Config_IRL.get('NUM_TRAJECTORIES')
trajectory_length = Config_IRL.get('TRAJECTORY_LENGTH')

#########################################################
# Function definition


def expert_policy(uav, ues_objects, ax_ues, cell_objects):
    episode = 0
    tmp_value = 1
    while episode <= num_trajectories:
        ax_ues.patches[tmp_value].set_color('g')
        rand_val = randint(1, 23)
        tmp_value = rand_val
        ax_ues.patches[rand_val].set_color('b')
        plt.pause(0.0000001)
        episode += 1
    pass


def get_features(state):
    pass

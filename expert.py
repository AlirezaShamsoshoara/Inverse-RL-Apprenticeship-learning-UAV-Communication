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
from config import Config_requirement

#########################################################
# General Parameters
num_features = Config_IRL.get('NUM_FEATURES')
dist_limit = Config_requirement.get('dist_limit')
num_trajectories = Config_IRL.get('NUM_TRAJECTORIES')
trajectory_length = Config_IRL.get('TRAJECTORY_LENGTH')


#########################################################
# Function definition


def expert_policy(uav, ues_objects, ax_ues, cell_objects):
    episode = 0
    tmp_value = 1
    state = 0
    while episode <= num_trajectories:
        get_features(state=state, cell_objects=cell_objects, uav=uav)
        ax_ues.patches[tmp_value].set_color('g')
        rand_val = randint(1, 23)
        tmp_value = rand_val
        ax_ues.patches[rand_val].set_color('b')
        plt.pause(0.0000001)
        episode += 1
    pass


def get_features(state, cell_objects, uav):
    num_neighbors_ues = 0
    phi_distance = 1 - np.power((cell_objects[state].get_distance()) / dist_limit, 2.)
    phi_hop = 1 - np.power((uav.get_hop()) / dist_limit, 2.)
    for neighbor in cell_objects[state].get_neighbor():
        num_neighbors_ues += len(cell_objects[neighbor].get_ues_idx())
    phi_ues = np.exp(num_neighbors_ues)
    phi_throughput = np.log2(1 + uav.get)
    phi_interference =
    return phi_distance, phi_hop, phi_ues

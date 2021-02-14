"""
#################################
# Random Policy functions
#################################
"""

#########################################################
# import libraries
from random import randint
import matplotlib.pyplot as plt
from config import Config_General
from config import Config_requirement

#########################################################
# General Parameters
num_cells = Config_General.get('NUM_CELLS')
dist_limit = Config_requirement.get('dist_limit')
cell_source = 0
cell_destination = num_cells - 1

#########################################################
# Function definition


def random_action(uav, ues_objects, ax_ues, cell_objects):
    print(" ****** Mode: Random policy by the drone ")
    tmp_cell = 1
    distance = 0
    # while True:
    while distance <= dist_limit:
        # TODO: Set the transmission power for higher throughput
        # TODO: Calculate the interference from other neighbor UEs on the UAV base station
        # TODO: Calculate the UAV's interference effect on neighbor UEs because of the power allocation

        ax_ues.patches[tmp_cell].set_color('g')
        ax_ues.patches[cell_source].set_color('r')
        ax_ues.patches[cell_destination].set_color('r')
        cell = uav.get_cell_id()

        avail_neighbors = cell_objects[cell].get_neighbor()
        avail_actions = cell_objects[cell].get_actions()
        idx_rand = randint(0, len(avail_actions)-1)
        action_rand = avail_actions[idx_rand]
        neighbor_rand = avail_neighbors[idx_rand]
        uav.set_action_movement(action_rand)
        uav.set_cell_id(neighbor_rand)
        uav.set_location(cell_objects[neighbor_rand].get_location())
        ax_ues.patches[neighbor_rand].set_color('b')
        tmp_cell = neighbor_rand
        distance += 1
        plt.pause(0.00000001)

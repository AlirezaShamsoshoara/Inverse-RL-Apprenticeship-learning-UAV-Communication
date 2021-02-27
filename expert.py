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
from config import Config_Power
import matplotlib.pyplot as plt
from location import update_axes
from config import Config_General
from config import Config_requirement
from config import movement_actions_list
from utils import multi_actions_to_action

#########################################################
# General Parameters
num_cells = Config_General.get('NUM_CELLS')
cell_source = 0
cell_destination = num_cells - 1
tx_powers = Config_Power.get('UAV_Tr_power')
num_features = Config_IRL.get('NUM_FEATURES')
dist_limit = Config_requirement.get('dist_limit')
num_trajectories = Config_IRL.get('NUM_TRAJECTORIES')
trajectory_length = Config_IRL.get('TRAJECTORY_LENGTH')

#########################################################
# Function definition


def expert_policy(uav, ues_objects, ax_objects, cell_objects):
    episode = 0
    prev_cell = 1

    while episode <= num_trajectories:
        cell = uav.get_cell_id()
        state = uav.get_cell_id()
        expert_action_mov = int(input("Please select the cell to move" + str(movement_actions_list) + ": "))
        expert_action_power = float(input("Please select the TX Power" + str(tx_powers) + ":"))
        expert_action = multi_actions_to_action(expert_action_mov, expert_action_power)
        # expert_action_mov_index = np.where(expert_action_mov == np.array(movement_actions_list))[0]
        # expert_action_power_index = np.where(expert_action_power == np.array(tx_powers))[0]

        avail_actions_mov = cell_objects[cell].get_actions()
        avail_neighbors = cell_objects[cell].get_neighbor()
        new_state = avail_neighbors[np.where(expert_action_mov == np.array(avail_actions_mov))[0][0]]
        new_cell = new_state
        uav.set_cell_id(cid=new_cell)
        uav.set_location(loc=cell_objects[new_cell].get_location())
        uav.set_hop(hop=uav.get_hop()+1)

        uav.set_power(tr_power=expert_action_power)
        interference = uav.calc_interference(cell_objects, ues_objects)
        sinr, snr = uav.calc_sinr(cell_objects)
        throughput = uav.calc_throughput()
        interference_ues = uav.calc_interference_ues(cell_objects, ues_objects)

        phi_1, phi_2, phi_3, phi_4, phi_5 = get_features(state=state, cell_objects=cell_objects, uav=uav,
                                                         ues_objects=ues_objects)

        update_axes(ax_objects, prev_cell, cell_source, cell_destination, new_cell, expert_action_power,
                    cell_objects[new_cell].get_location(), expert_action_mov, cell_objects[cell].get_location())

        plt.pause(0.001)
        prev_cell = new_cell

        episode += 1


def get_features(state, cell_objects, uav, ues_objects):
    num_neighbors_ues = 0
    phi_distance = 1 - np.power((cell_objects[state].get_distance()) / dist_limit, 2.)
    phi_hop = 1 - np.power((uav.get_hop()) / dist_limit, 2.)
    for neighbor in cell_objects[state].get_neighbor():
        num_neighbors_ues += len(cell_objects[neighbor].get_ues_idx())
    phi_ues = np.exp(-num_neighbors_ues)
    phi_throughput = np.power((uav.calc_throughput()) / uav.calc_max_throughput(cell_objects=cell_objects), 2)
    phi_interference = np.exp(-uav.calc_interference_ues(cells_objects=cell_objects, ues_objects=ues_objects))
    return phi_distance, phi_hop, phi_ues, phi_throughput, phi_interference

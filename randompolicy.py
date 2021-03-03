"""
#################################
# Random Policy functions
#################################
"""

#########################################################
# import libraries
from random import seed
from random import randint
import matplotlib.pyplot as plt
from config import Config_Power
from location import update_axes
from config import Config_General
from config import Config_requirement

#########################################################
# General Parameters
num_cells = Config_General.get('NUM_CELLS')
cell_source = 0
cell_destination = num_cells - 1
tx_powers = Config_Power.get('UAV_Tr_power')
dist_limit = Config_requirement.get('dist_limit')

#########################################################
# Function definition


def random_action(uav, ues_objects, ax_objects, cell_objects):
    print(" ****** Mode: Random policy by the drone ")
    seed(1732)
    prev_cell = 1
    distance = 0
    while True:
    # while distance <= dist_limit:
        # TODO: Set the transmission power for the throughput (Done!)
        # TODO: Calculate the interference from other neighbor UEs on the UAV's base station (Done!)
        # TODO: Calculate the UAV's interference effect on neighbor UEs because of the transmission power
        #  allocation (Done!)

        cell = uav.get_cell_id()
        avail_neighbors = cell_objects[cell].get_neighbor()
        avail_actions = cell_objects[cell].get_actions()

        idx_rand = randint(0, len(avail_actions)-1)
        action_rand = avail_actions[idx_rand]
        neighbor_rand = avail_neighbors[idx_rand]
        uav.set_action_movement(action=action_rand)
        uav.set_cell_id(cid=neighbor_rand)
        uav.set_location(loc=cell_objects[neighbor_rand].get_location())

        tx_index = randint(0, len(tx_powers)-1)
        tx_power = tx_powers[tx_index]
        uav.set_power(tr_power=tx_power)
        update_axes(ax_objects, prev_cell, cell_source, cell_destination, neighbor_rand, tx_power,
                    cell_objects[neighbor_rand].get_location(), action_rand, cell_objects[cell].get_location())

        interference = uav.calc_interference(cell_objects, ues_objects)
        sinr, snr = uav.calc_sinr(cell_objects)
        throughput = uav.calc_throughput()
        interference_ues = uav.calc_interference_ues(cell_objects, ues_objects)

        # Should remove these above lines
        uav.uav_perform_task(cell_objects, ues_objects)

        prev_cell = neighbor_rand
        distance += 1
        plt.pause(0.00000001)
        # plt.pause(0.05)

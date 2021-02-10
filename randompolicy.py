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

#########################################################
# Function definition


def random_action(uav, ues_objects, ax_ues):
    print(" ****** Mode: Random policy by the drone ")
    tmp_value = 1
    while True:
        ax_ues.patches[tmp_value].set_color('g')
        rand_val = randint(0+1, num_cells-2)
        tmp_value = rand_val
        ax_ues.patches[rand_val].set_color('b')
        # plt.pause(0.000000000001)
        plt.pause(0.00000001)
    pass

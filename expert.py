"""
#################################
# Expert Operation functions
#################################
"""

#########################################################
# import libraries
import numpy as np
from random import randint

#########################################################
# General Parameters

#########################################################
# Function definition


def expert_policy(uav, ues_objects, ax_ues):
    tmp_value = 1
    while True:
        ax_ues.patches[tmp_value].set_color('g')
        rand_val = randint(1, 23)
        tmp_value = rand_val
        ax_ues.patches[rand_val].set_color('b')
    pass

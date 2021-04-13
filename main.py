"""
Created on January 26th, 2020
@author:    Alireza Shamsoshoara
@Project:   UAV communication using apprenticeship learning via Inverse Reinforcement Learning (IRL)
            Paper: ### TODO
            Arxiv: ### TODO
            Github: https://github.com/AlirezaShamsoshoara/Inverse-RL-Apprenticeship-learning-UAV-Communication
            YouTube Link: ### TODO
@Northern Arizona University
This project is developed and tested with Python 3.6 using pycharm on Ubuntu 18.04 LTS machine
"""

#################################
# Main File
#################################

# ############# import libraries
# General Modules

# Customized Modules
from utils import UAV
from config import Mode
from qlearning import qrl
from deeprl import deep_rl
from location import plotues
from utils import create_ues
from utils import create_cells
from inverserlSGD import inverse_rl
from expert import expert_policy
from location import plothexagon
from config import Config_General
from utils import find_closest_cell
from shortestpath import short_path
from randompolicy import random_action
from behavioral import behavioral_cloning

#########################################################
# General Parameters
Altitude = Config_General.get('Altitude')

#########################################################
# Main


def main():
    print(" ..... Running:")
    print("UAV communication using apprenticeship learning via Inverse Reinforcement Learning (IRL)")
    v_coord_cells, h_coord_cells, cell_ids, fig_cells, ax_cells, coordinates = plothexagon()
    fig_ues, ax_ues, x_coord_ues, y_coord_ues = plotues(fig_cells, ax_cells, cell_ids, h_coord_cells, v_coord_cells)
    ue_cell_ids = find_closest_cell(h_coord_cells, v_coord_cells, x_coord_ues, y_coord_ues)
    ues_objects = create_ues(x_coord_ues, y_coord_ues, ue_cell_ids)
    cells_objects = create_cells(h_coord_cells, v_coord_cells, cell_ids, ue_cell_ids, coordinates)
    uav = UAV(x_loc=0, y_loc=0, z_loc=Altitude, cell_id=0)
    return uav, ues_objects, ax_ues, cells_objects


if __name__ == "__main__":
    uav_main, ues_objects_main, ax_ues_main, cells_objects_main = main()
    if Mode == "Expert":
        expert_policy(uav_main, ues_objects_main, ax_ues_main, cells_objects_main)
    elif Mode == "IRL_SGD" or Mode == "IRL_DQN":
        inverse_rl(uav_main, ues_objects_main, ax_ues_main, cells_objects_main)
    elif Mode == "DRL":
        deep_rl()
    elif Mode == "QRL":
        qrl()
    elif Mode == "BC":
        behavioral_cloning(uav_main, ues_objects_main, ax_ues_main, cells_objects_main)
    elif Mode == "Shortest":
        short_path(uav_main, ues_objects_main, ax_ues_main, cells_objects_main)
    elif Mode == "Random":
        random_action(uav_main, ues_objects_main, ax_ues_main, cells_objects_main)
    elif Mode == "ResultsIRL":
        pass
    elif Mode == "ResultsDRL":
        pass
    elif Mode == "ResultsQRL":
        pass
    elif Mode == "ResultsRand":
        pass
    else:
        print("Mode is not correct")

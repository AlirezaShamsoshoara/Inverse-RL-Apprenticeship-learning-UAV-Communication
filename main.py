"""
Created on January 26th, 2020
@author:    Alireza Shamsoshoara
@Project:   UAV communication using apprenticeship learning via Inverse Reinforcement Learning (IRL)
            Paper: ### TODO
            Arxiv: ### TODO
            YouTUbeLink: ### TODO
@Northern Arizona University
This project is developed and tested with Python 3.6 using pycharm on Ubuntu 18.04 LTS machine
"""

#################################
# Main File
#################################

# ############# import libraries
# General Modules

# Customized Modules
from qrl import q_rl
from config import Mode
from deeprl import deep_rl
from location import plotues
from config import Config_Power
from expert import expert_policy
from inverserl import inverse_rl
from location import plothexagon
from randompolicy import random_action


def main():
    print(" ..... Running:")
    print("UAV communication using apprenticeship learning via Inverse Reinforcement Learning (IRL)")
    v_coord_cells, h_coord_cells, cell_ids, fig_cells, ax_cells = plothexagon()
    fig_ues, ax_ues = plotues(fig_cells, ax_cells, cell_ids, h_coord_cells, v_coord_cells)


if __name__ == "__main__":
    if Mode == "Expert":
        main()
        expert_policy()
    elif Mode == "IRL":
        inverse_rl()
    elif Mode == "DRL":
        deep_rl()
    elif Mode == "QRL":
        q_rl()
    elif Mode == "Random":
        random_action()
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

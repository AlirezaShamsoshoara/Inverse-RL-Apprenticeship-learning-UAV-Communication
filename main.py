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
from config import Config_Power
from expert import expert_policy
from inverserl import inverse_rl
from location import plothexagon


def main():
    print(" ..... Running:")
    print("UAV communication using apprenticeship learning via Inverse Reinforcement Learning (IRL)")
    v_coord_cells, h_coord_cells, cell_ids, fig_cells, ax_cells = plothexagon()
    print("UAV Transmit Power Options: ", Config_Power.get('UAV_Tr_power'))


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
    else:
        print("Mode is not correct")

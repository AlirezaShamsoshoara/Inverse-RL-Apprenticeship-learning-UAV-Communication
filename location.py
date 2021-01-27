"""
#################################
# Location and allocation functions and modules
#################################
"""

#########################################################
# import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

#########################################################
# General Parameters

#########################################################
# Function and definition


def plothexagon():
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    hexagon = RegularPolygon((0, 0), numVertices=6, radius=1, edgecolor='k', facecolor='g', alpha=0.4)
    ax.add_patch(hexagon)

    plt.show()


class Cell:

    def __init__(self, x_loc, y_loc, num_ues, uid):
        pass

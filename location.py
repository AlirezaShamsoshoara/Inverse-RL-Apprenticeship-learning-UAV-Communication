"""
#################################
# Location and allocation functions and modules
#################################
"""

#########################################################
# import libraries
import numpy as np
import matplotlib.pyplot as plt
from hexalattice.hexalattice import *
from matplotlib.patches import RegularPolygon

#########################################################
# General Parameters

#########################################################
# Function and definition


def plothexagon():
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    hexagon = RegularPolygon((0, 0), numVertices=6, radius=1, edgecolor='k', facecolor='g', alpha=0.3,
                             orientation=np.float(np.radians(30)))
    ax.add_patch(hexagon)

    hex_centers, _ = create_hex_grid(nx=5,
                                     ny=5,
                                     do_plot=True)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    plt.show()


class Cell:

    def __init__(self, x_loc, y_loc, num_ues, uid):
        pass

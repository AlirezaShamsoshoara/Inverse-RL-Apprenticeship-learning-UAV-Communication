"""
#################################
# Location and allocation functions and modules
#################################
"""

#########################################################
# import libraries
from config import Config_General
import matplotlib.pyplot as plt
from hexalattice.hexalattice import *
from matplotlib.patches import RegularPolygon
import numpy as np

#########################################################
# General Parameters
radius = Config_General.get('Radius')
cells = Config_General.get('NUM_CELLS')

#########################################################
# Function and definition


def plothexagon():
    fig_cells, ax_cells = plt.subplots(1, figsize=(12, 8))
    ax_cells.set_aspect('equal')
    coordinates = [[None, float] for i in range(0, cells)]
    cell = 0
    for ind_x in range(0, np.int(np.sqrt(cells))):
        for ind_y in range(0, np.int(np.sqrt(cells))):
            coordinates[cell][:] = [ind_x, ind_y]
            cell += 1
    cell_ids = [np.int(np.sqrt(cells))*coord[0] + coord[1] for coord in coordinates]
    hcoord = [(3./2.) * radius * coord[0] for coord in coordinates]
    vcoord = [-(np.mod(coord[0], 2))*((np.sqrt(3.)/2.) * radius) + ((coord[1]) * (np.sqrt(3.)) * radius) for coord
              in coordinates]

    for x, y, cid in zip(hcoord, vcoord, cell_ids):
        hexagon = RegularPolygon((x, y), numVertices=6, radius=radius, edgecolor='k', facecolor='g', alpha=0.25,
                                 orientation=np.float(np.radians(30)), linewidth=1.5)
        ax_cells.add_patch(hexagon)
        ax_cells.text(x, y+((np.sqrt(3.)/8.) * radius), cid, ha='center', va='center', size=12)

    # hex_centers, _ = create_hex_grid(n=40,
    #                                  do_plot=True,
    #                                  rotate_deg=30.0,
    #                                  face_color=[0, 0.6, 0.4])
    ax_cells.scatter(hcoord, vcoord, color='b', alpha=0.5)
    ax_cells.set_xlim([min(hcoord) - 2*radius, max(hcoord) + 2*radius])
    ax_cells.set_ylim([min(vcoord) - 2*radius, max(vcoord) + 2*radius])
    ax_cells.grid(True)
    plt.show()
    return vcoord, hcoord, cell_ids, fig_cells, ax_cells


class Cell:

    def __init__(self, x_loc, y_loc, num_ues, uid):
        pass

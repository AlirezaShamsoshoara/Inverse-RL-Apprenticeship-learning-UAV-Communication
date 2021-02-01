"""
#################################
# Location and allocation functions and modules
#################################
"""

#########################################################
# import libraries
import numpy as np
import matplotlib.pyplot as plt
from config import Config_General
from hexalattice.hexalattice import *
from matplotlib.patches import RegularPolygon


#########################################################
# General Parameters
radius = Config_General.get('Radius')
cells = Config_General.get('NUM_CELLS')
num_ues = Config_General.get('NUM_UEs')
loc_delta = Config_General.get('Loc_delta')

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
    ax_cells.scatter(hcoord, vcoord, color='b', alpha=0.5, marker='^', s=50)
    ax_cells.patches[0].set_color('r')
    ax_cells.patches[cells-1].set_color('r')
    ax_cells.set_xlim([min(hcoord) - 2 * radius, max(hcoord) + 2 * radius])
    ax_cells.set_ylim([min(vcoord) - 2 * radius, max(vcoord) + 2 * radius])
    ax_cells.grid(True)
    # plt.show(block=False)
    return vcoord, hcoord, cell_ids, fig_cells, ax_cells


def plotues(fig_cells, ax_cells, cell_ids, hcoord, vcoord):
    x_coord_ues = np.zeros([num_ues], dtype=float)
    y_coord_ues = np.zeros([num_ues], dtype=float)
    x_coord_ues[0], y_coord_ues[0] = hcoord[1] - 3 * loc_delta, vcoord[1] + 0 * loc_delta
    x_coord_ues[1], y_coord_ues[1] = hcoord[1] + 0 * loc_delta, vcoord[1] + 3 * loc_delta

    x_coord_ues[2], y_coord_ues[2] = hcoord[2] - 2 * loc_delta, vcoord[2] - 2 * loc_delta
    x_coord_ues[3], y_coord_ues[3] = hcoord[2] - 0 * loc_delta, vcoord[2] + 2 * loc_delta

    x_coord_ues[4], y_coord_ues[4] = hcoord[3] - 1 * loc_delta, vcoord[3] - 3 * loc_delta
    x_coord_ues[5], y_coord_ues[5] = hcoord[3] + 1 * loc_delta, vcoord[3] + 3 * loc_delta

    x_coord_ues[6], y_coord_ues[6] = hcoord[4] - 0 * loc_delta, vcoord[4] + 3 * loc_delta
    x_coord_ues[7], y_coord_ues[7] = hcoord[4] + 0.5 * loc_delta, vcoord[4] - 3 * loc_delta
    x_coord_ues[8], y_coord_ues[8] = hcoord[4] - 3 * loc_delta, vcoord[4] + 1 * loc_delta

    x_coord_ues[9], y_coord_ues[9] = hcoord[5] - 2 * loc_delta, vcoord[5] - 2 * loc_delta
    x_coord_ues[10], y_coord_ues[10] = hcoord[5] + 1 * loc_delta, vcoord[5] - 3 * loc_delta
    x_coord_ues[11], y_coord_ues[11] = hcoord[5] + 2 * loc_delta, vcoord[5] + 2 * loc_delta

    x_coord_ues[12], y_coord_ues[12] = hcoord[6] - 3 * loc_delta, vcoord[6] - 3 * loc_delta
    x_coord_ues[13], y_coord_ues[13] = hcoord[6] + 2 * loc_delta, vcoord[6] + 2 * loc_delta
    x_coord_ues[14], y_coord_ues[14] = hcoord[6] - 1 * loc_delta, vcoord[6] + 2 * loc_delta

    x_coord_ues[15], y_coord_ues[15] = hcoord[7] + 3 * loc_delta, vcoord[7] + 2.5 * loc_delta
    x_coord_ues[16], y_coord_ues[16] = hcoord[7] - 3 * loc_delta, vcoord[7] + 1 * loc_delta
    x_coord_ues[17], y_coord_ues[17] = hcoord[7] - 2 * loc_delta, vcoord[7] - 3 * loc_delta

    x_coord_ues[18], y_coord_ues[18] = hcoord[8] + 0 * loc_delta, vcoord[8] + 3 * loc_delta
    x_coord_ues[19], y_coord_ues[19] = hcoord[8] - 2 * loc_delta, vcoord[8] + 1 * loc_delta
    x_coord_ues[20], y_coord_ues[20] = hcoord[8] - 1 * loc_delta, vcoord[8] - 2 * loc_delta

    x_coord_ues[21], y_coord_ues[21] = hcoord[9] - 2.5 * loc_delta, vcoord[9] + 2.5 * loc_delta
    x_coord_ues[22], y_coord_ues[22] = hcoord[9] + 3 * loc_delta, vcoord[9] + 2 * loc_delta

    x_coord_ues[23], y_coord_ues[23] = hcoord[10] - 2 * loc_delta, vcoord[10] + 2 * loc_delta
    x_coord_ues[24], y_coord_ues[24] = hcoord[10] + 3 * loc_delta, vcoord[10] + 1 * loc_delta
    x_coord_ues[25], y_coord_ues[25] = hcoord[10] - 2.5 * loc_delta, vcoord[10] - 2.5 * loc_delta
    x_coord_ues[26], y_coord_ues[26] = hcoord[10] + 0 * loc_delta, vcoord[10] - 3 * loc_delta
    x_coord_ues[27], y_coord_ues[27] = hcoord[10] + 1 * loc_delta, vcoord[10] - 2.7 * loc_delta

    x_coord_ues[28], y_coord_ues[28] = hcoord[11] - 0 * loc_delta, vcoord[11] - 2.5 * loc_delta
    x_coord_ues[29], y_coord_ues[29] = hcoord[11] + 2.5 * loc_delta, vcoord[11] - 1.7 * loc_delta
    x_coord_ues[30], y_coord_ues[30] = hcoord[11] + 2.5 * loc_delta, vcoord[11] + 1.8 * loc_delta
    x_coord_ues[31], y_coord_ues[31] = hcoord[11] + 1 * loc_delta, vcoord[11] + 2.9 * loc_delta

    x_coord_ues[32], y_coord_ues[32] = hcoord[12] + 1 * loc_delta, vcoord[12] - 2.8 * loc_delta
    x_coord_ues[33], y_coord_ues[33] = hcoord[12] + 3 * loc_delta, vcoord[12] + 0 * loc_delta
    x_coord_ues[34], y_coord_ues[34] = hcoord[12] - 1 * loc_delta, vcoord[12] + 1 * loc_delta

    x_coord_ues[35], y_coord_ues[35] = hcoord[13] + 1 * loc_delta, vcoord[13] + 2 * loc_delta
    x_coord_ues[36], y_coord_ues[36] = hcoord[13] - 2.5 * loc_delta, vcoord[13] + 0 * loc_delta

    x_coord_ues[37], y_coord_ues[37] = hcoord[14] + 0 * loc_delta, vcoord[14] + 2 * loc_delta
    x_coord_ues[38], y_coord_ues[38] = hcoord[14] - 2.5 * loc_delta, vcoord[14] + 1.3 * loc_delta

    x_coord_ues[39], y_coord_ues[39] = hcoord[15] + 0 * loc_delta, vcoord[15] - 3 * loc_delta
    x_coord_ues[40], y_coord_ues[40] = hcoord[15] + 2.7 * loc_delta, vcoord[15] - 1 * loc_delta
    x_coord_ues[41], y_coord_ues[41] = hcoord[15] + 1.5 * loc_delta, vcoord[15] + 2.7 * loc_delta
    x_coord_ues[42], y_coord_ues[42] = hcoord[15] - 1 * loc_delta, vcoord[15] + 2.7 * loc_delta
    x_coord_ues[43], y_coord_ues[43] = hcoord[15] - 3 * loc_delta, vcoord[15] + 0 * loc_delta

    x_coord_ues[44], y_coord_ues[44] = hcoord[16] + 2.7 * loc_delta, vcoord[16] - 1 * loc_delta
    x_coord_ues[45], y_coord_ues[45] = hcoord[16] + 1.5 * loc_delta, vcoord[16] + 2.7 * loc_delta
    x_coord_ues[46], y_coord_ues[46] = hcoord[16] - 1 * loc_delta, vcoord[16] + 2.7 * loc_delta
    x_coord_ues[47], y_coord_ues[47] = hcoord[16] - 3 * loc_delta, vcoord[16] + 0 * loc_delta

    ax_cells.scatter(x_coord_ues[:], y_coord_ues[:], color='m', edgecolors='none', marker='o')
    # ax_cells.scatter(x_coord_ues, y_coord_ues, color='m', alpha=0.01)
    plt.show(block=True)
    return fig_cells, ax_cells

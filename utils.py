"""
#################################
# Utility functions
#################################
"""

#########################################################
# import libraries
import numpy as np
from config import Config_General
from scipy.spatial.distance import cdist

#########################################################
# General Parameters
num_ues = Config_General.get("NUM_UEs")
num_cells = Config_General.get("NUM_CELLS")
ue_tr_power = Config_General.get("UE_Tr_power")

#########################################################
# Class and Function definitions


class Cell:

    def __init__(self, x_loc=None, y_loc=None, num_ues_cell=-1, unique_id=-1):
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.num_ues_cell = num_ues_cell
        self.cell_id = unique_id
        self.location = [self.x_loc, self.y_loc]

    def set_location(self, loc):
        self.x_loc = loc[0]
        self.y_loc = loc[1]
        self.location = [self.x_loc, self.y_loc]

    def set_num_ues(self, num_ues_cell):
        self.num_ues_cell = num_ues_cell

    def set_id(self, uid):
        self.cell_id = uid

    def get_location(self):
        return self.location

    def get_num_ues(self):
        return self.num_ues_cell

    def get_id(self):
        return self.cell_id


class UAV:

    def __init__(self, x_loc=None, y_loc=None, cell_id=-1, tr_power=0):
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.cell_id = cell_id
        self.power = tr_power
        self.location = [self.x_loc, self.y_loc]

    def set_location(self, loc):
        self.x_loc = loc[0]
        self.y_loc = loc[1]
        self.location = [self.x_loc, self.y_loc]

    def set_id(self, cid):
        self.cell_id = cid

    def set_power(self, tr_power):
        self.power = tr_power

    def get_location(self):
        return self.location

    def get_cell_id(self):
        return self.cell_id

    def get_tr_power(self):
        return self.power

    def send_pkt(self):
        pass

    def calc_throughput(self):
        pass

    def calc_interference(self):
        pass

    def calc_snir(self):
        pass


class UE:

    def __init__(self, x_loc=None, y_loc=None, ue_id=-1, cell_id=-1, tr_power=0):
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.ue_id = ue_id
        self.cell_id = cell_id
        self.power = tr_power
        self.location = [self.x_loc, self.y_loc]

    def set_location(self, loc):
        self.x_loc = loc[0]
        self.y_loc = loc[1]
        self.location = [self.x_loc, self.y_loc]

    def set_ue_id(self, ue_id):
        self.ue_id = ue_id

    def set_cell_id(self, cell_id):
        self.cell_id = cell_id

    def set_power(self, tr_power):
        self.power = tr_power

    def get_location(self):
        return self.location

    def get_ue_id(self):
        return self.ue_id

    def get_cell_id(self):
        return self.cell_id

    def get_power(self):
        return self.power

    def calc_interference(self):
        pass


def find_closest_cell(h_coord_cells, v_coord_cells, x_coord_ues, y_coord_ues):
    ue_cell_ids = np.zeros([num_ues], dtype=np.int16) - 1
    cell_coord_pairs = np.concatenate((h_coord_cells.reshape(-1, 1), v_coord_cells.reshape(-1, 1)), axis=1)
    for index in range(0, num_ues):
        dist = cdist(np.array([[x_coord_ues[index], y_coord_ues[index]]]), cell_coord_pairs, 'euclidean')
        min_index = np.argmin(dist)
        ue_cell_ids[index] = min_index
    return ue_cell_ids


def create_ues(x_coord_ues, y_coord_ues, ue_cell_ids):
    ues_objects = np.empty((num_ues, 1), dtype=object)
    for ue in range(0, num_ues):
        ues_objects[ue] = UE(x_loc=x_coord_ues[ue], y_loc=y_coord_ues[ue])
        ues_objects[ue, 0].set_ue_id(ue)
        ues_objects[ue, 0].set_cell_id(ue_cell_ids[ue])
        ues_objects[ue, 0].set_power(ue_tr_power)
    return ues_objects

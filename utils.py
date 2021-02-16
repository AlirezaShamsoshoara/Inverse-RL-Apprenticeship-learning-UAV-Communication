"""
#################################
# Utility functions
#################################
"""

#########################################################
# import libraries
import numpy as np
from config import Config_Power
from config import Config_General
from config import Config_interference
from config import config_movement_step
from config import movement_actions_list
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean

#########################################################
# General Parameters
num_ues = Config_General.get("NUM_UEs")
num_cells = Config_General.get("NUM_CELLS")
ue_tr_power = Config_Power.get("UE_Tr_power")
float_acc = Config_General.get('FLOAT_ACCURACY')
antenna_gain = Config_interference.get('AntennaGain')


#########################################################
# Class and Function definitions


class Cell:

    def __init__(self, x_loc=None, y_loc=None, num_ues_cell=-1, unique_id=-1):
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.z_loc = 0.0
        self.num_ues_cell = num_ues_cell
        self.cell_id = unique_id
        self.location = [self.x_loc, self.y_loc, self.z_loc]
        self.ues_idx = None
        self.coordinate = None
        self.neighbors = None
        self.available_actions = None

    def print_info(self):
        print("Cell ID = ", self.cell_id, "\n",
              "Location = ", self.location, "\n",
              "Num UEs = ", self.num_ues_cell, "\n",
              "UEs index = ", self.ues_idx, "\n",
              "Neighbors = ", self.neighbors, "\n",
              "Actions = ", self.available_actions, "\n")

    def set_location(self, loc):
        self.x_loc = loc[0]
        self.y_loc = loc[1]
        self.location = [self.x_loc, self.y_loc, self.z_loc]

    def set_num_ues(self, num_ues_cell):
        self.num_ues_cell = num_ues_cell

    def set_id(self, uid):
        self.cell_id = uid

    def set_ues_ids(self, ues_idx):
        self.ues_idx = ues_idx

    def set_coord(self, coord):
        self.coordinate = coord

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors

    def set_available_actions(self, actions):
        self.available_actions = actions

    def get_location(self):
        return self.location

    def get_num_ues(self):
        return self.num_ues_cell

    def get_id(self):
        return self.cell_id

    def get_ues_idx(self):
        return self.ues_idx

    def get_coord(self):
        return self.coordinate

    def get_neighbor(self):
        return self.neighbors

    def get_actions(self):
        return self.available_actions


class UAV:

    def __init__(self, x_loc=None, y_loc=None, z_loc=None, cell_id=-1, tr_power=0):
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.z_loc = z_loc
        self.cell_id = cell_id
        self.power = tr_power
        self.location = [self.x_loc, self.y_loc, self.z_loc]
        self.action_movement = 0
        self.interference = 0

    def set_location(self, loc):
        self.x_loc = loc[0]
        self.y_loc = loc[1]
        self.location = [self.x_loc, self.y_loc, self.z_loc]

    def set_cell_id(self, cid):
        self.cell_id = cid

    def set_power(self, tr_power):
        self.power = tr_power

    def set_action_movement(self, action):
        self.action_movement = action

    def get_location(self):
        return self.location

    def get_cell_id(self):
        return self.cell_id

    def get_tr_power(self):
        return self.power

    def get_action_movement(self):
        return self.action_movement

    def send_pkt(self):
        pass

    def calc_throughput(self):
        pass

    def calc_interference(self, cells_objects, ues_objects):
        current_cell = self.get_cell_id()
        neighbors = cells_objects[current_cell].get_neighbor()
        interference = 0
        for neighbor in neighbors:
            ues = cells_objects[neighbor].get_ues_idx()
            for ue in ues:
                csi = get_csi(ues_objects[ue].get_location(), cells_objects[current_cell].get_location())
                interference += (ues_objects[ue].get_power()) * ((abs(csi))**2)
                print(interference)
        self.interference = interference
        return self.interference

    def calc_sinr(self, cell_objects):
        sinr =


class UE:

    def __init__(self, x_loc=None, y_loc=None, ue_id=-1, cell_id=-1, tr_power=0):
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.z_loc = 0.0
        self.ue_id = ue_id
        self.cell_id = cell_id
        self.power = tr_power
        self.location = [self.x_loc, self.y_loc, self.z_loc]

    def set_location(self, loc):
        self.x_loc = loc[0]
        self.y_loc = loc[1]
        self.location = [self.x_loc, self.y_loc, self.z_loc]

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
    ues_objects = np.empty(num_ues, dtype=object)
    for ue in range(0, num_ues):
        ues_objects[ue] = UE(x_loc=x_coord_ues[ue], y_loc=y_coord_ues[ue])
        ues_objects[ue].set_ue_id(ue)
        ues_objects[ue].set_cell_id(ue_cell_ids[ue])
        ues_objects[ue].set_power(ue_tr_power)
    return ues_objects


def create_cells(h_coord_cells, v_coord_cells, cell_ids, ue_cell_ids, coordinates):
    cells_objects = np.empty(num_cells, dtype=object)
    counts = np.zeros(num_cells, dtype=np.int16)
    _, counts[0+1:num_cells-1] = np.unique(ue_cell_ids, return_counts=True)
    for cell in range(0, num_cells):
        cells_objects[cell] = Cell(h_coord_cells[cell], v_coord_cells[cell])
        cells_objects[cell].set_id(cell_ids[cell])
        cells_objects[cell].set_num_ues(counts[cell])
        cells_objects[cell].set_ues_ids(np.where(ue_cell_ids == cell)[0])
        cells_objects[cell].set_coord(coordinates[cell])

    for cell in range(0, num_cells):
        available_neighbor, available_action = find_neighbors(cells_objects[cell], cells_objects)
        cells_objects[cell].set_neighbors(available_neighbor)
        cells_objects[cell].set_available_actions(available_action)
    return cells_objects


def check_neighbor_availability(location, cells_objects):
    for cell in range(0, len(cells_objects)):
        if round(cells_objects[cell].get_location()[0], float_acc) == round(location[0], float_acc) and \
                round(cells_objects[cell].get_location()[1], float_acc) == round(location[1], float_acc):
            return True, cells_objects[cell].get_id()
    return False, None


def find_neighbors(cell_object, cell_objects):
    available_action = []
    available_neighbor = []
    x_cell = cell_object.get_location()[0]
    y_cell = cell_object.get_location()[1]
    for action in movement_actions_list:
        x_change, y_change = action_to_location(action)
        new_x = x_cell + x_change
        new_y = y_cell + y_change
        check_flag, neighbor = check_neighbor_availability([new_x, new_y], cell_objects)
        if check_flag:
            available_action.append(action)
            available_neighbor.append(neighbor)
    return available_neighbor, available_action


def action_to_location(action):
    x_change, y_change = None, None
    x_step, y_step = config_movement_step.get('x_step'), config_movement_step.get('y_step')
    if action == 1:
        x_change = 0
        y_change = y_step
    elif action == 2:
        x_change = x_step
        y_change = (1./2.) * y_step
    elif action == 3:
        x_change = x_step
        y_change = (-1./2.) * y_step
    elif action == 4:
        x_change = 0
        y_change = -y_step
    elif action == 5:
        x_change = -x_step
        y_change = (-1./2.) * y_step
    elif action == 6:
        x_change = -x_step
        y_change = (1./2.) * y_step
    else:
        exit('Error: Not a defined action for the movement')
    return x_change, y_change


def get_csi(loc_source, loc_destination):
    distance = euclidean(loc_source, loc_destination)
    csi = antenna_gain * (1/distance**2) * (1 + 1j)
    return csi

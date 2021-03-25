"""
#################################
# Configuration File
#################################
"""

#########################################################
# import libraries
import numpy as np


#########################################################
# Configuration
Mode = 'IRL'
# Different Modes {"Expert", "IRL", "DRL", "QRL", "BC", "Shortest", "Random", "ResultsIRL", "ResultsDRL", "ResultsQRL",
# "ResultsBC", "ResultShortest", "ResultsRand"}

# Possible number of UEs: 75
# Possible number of Cells: 25
Config_General = {'NUM_UAV': 1, 'Size': 5, 'NUM_CELLS': 25, 'NUM_UEs': 75, 'Radius': 10, 'Loc_delta': 2,
                  'FLOAT_ACCURACY': 6, 'Altitude': 50.0}
Config_requirement = {'dist_limit': Config_General.get('Size') + 3}
movement_actions_list = [1, 2, 3, 4, 5, 6]  # 1: North, 2: North East, 3: South East, 4: South, 5: South West,
# 6: North West
Number_of_neighbor_UEs = {'Min': 0, 'Max': 0}


config_movement_step = {'x_step': (Config_General.get('Radius')) * (3./2.),
                        'y_step': (Config_General.get('Radius')) * np.sqrt(3)}

Config_FLags = {'SAVE_path': True, 'Display_map': True, 'SingleArrow': False, 'SAVE_IRL_DATA': False,
                'SAVE_EXPERT_DATA': True, 'SAVE_IRL_WEIGHT': True, 'SAVE_MODEL_IRL_SGD': True, 'PLOT_RESULTS': True,
                'SAVE_PLOT_PDF': True, 'SAVE_PLOT_FIG': True, 'PRINT_INFO': True}

Config_interference = {'AntennaGain': 100, 'Bandwidth': 50}
Config_Power = {'UE_Tr_power': 2.0, 'UAV_Tr_power': [50.0, 60.0, 80.0, 100.0, 150.0, 200.0], 'UAV_init_energy': 400.0,
                'UAV_mobility_consumption': 10.0}  # Tr power: mW, Energy, Jule
# [50.0, 60.0, 80.0, 100.0, 150.0, 200.0]
# [50.0, 80.0, 100.0, 150.0]

Config_IRL = {'NUM_FEATURES': 4, 'NUM_EPOCHS': 100000, 'NUM_PLAY': 1, 'NUM_TRAJECTORIES_EXPERT': 2,
              'TRAJECTORY_LENGTH': Config_requirement.get('dist_limit'), 'GAMMA_FEATURES': 0.9,
              'EPSILON_OPTIMIZATION': 0.1, 'LEARNING_RATE': 1e-3, 'BATCH_SIZE': 100, 'EPSILON_GREEDY': 0.1,
              'GAMMA_DISCOUNT': 0.9}

Config_QRL = {}
Config_DRL = {}

pathDist = 'ConfigData/Cells_%d_Size_%d_UEs_%d' % (Config_General.get('NUM_CELLS'), Config_General.get('Size'),
                                                   Config_General.get('NUM_UEs'))

ExpertPath = "Data/ExpertDemo/"
WeightPath = "Data/Weights/"
InverseRLPath = "Data/InverseRL/"
ResultPathPDF = "Results/PDF/"
ResultPathFIG = "Results/FIG/"
SGDModelPath = "Data/InverseRL/SGDModel/"
Config_Path = {'PathDist': pathDist, 'ExpertPath': ExpertPath, 'WeightPath': WeightPath, 'InverseRLPath': InverseRLPath,
               'ResultPathPDF': ResultPathPDF, 'ResultPathFIG': ResultPathFIG, 'SGDModelPath': SGDModelPath}

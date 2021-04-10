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
Mode = 'BC'
# Different Modes {"Expert", "IRL_SGD", "IRL_DQN", "DRL", "QRL", "BC", "Shortest", "Random", "ResultsIRL", "ResultsDRL",
# "ResultsQRL", "ResultsBC", "ResultShortest", "ResultsRand"}

# Possible number of UEs Cluster: 75
# Possible number of Cells: 25
Config_General = {'NUM_UAV': 1, 'Size': 5, 'NUM_CELLS': 25, 'NUM_UEs': 75, 'Radius': 10, 'Loc_delta': 2,
                  'FLOAT_ACCURACY': 6, 'Altitude': 50.0}
Config_requirement = {'dist_limit': Config_General.get('Size') + 3, 'MAX_DISTANCE': 6, 'MIN_UE_NEIGHBORS': 4,
                      'MAX_UE_NEIGHBORS': 29, 'MIN_INTERFERENCE': 0.5123281666343314,
                      'MAX_INTERFERENCE': 14.621335028196711}

movement_actions_list = [1, 2, 3, 4, 5, 6]  # 1: North, 2: North East, 3: South East, 4: South, 5: South West,
# 6: North West
Number_of_neighbor_UEs = {'Min': 0, 'Max': 0}


config_movement_step = {'x_step': (Config_General.get('Radius')) * (3./2.),
                        'y_step': (Config_General.get('Radius')) * np.sqrt(3)}

Config_Flags = {'SAVE_path': True, 'Display_map': False, 'SingleArrow': False, 'SAVE_IRL_DATA': False,
                'SAVE_EXPERT_DATA': True, 'SAVE_IRL_WEIGHT': False, 'SAVE_MODEL_IRL_SGD': False, 'PLOT_RESULTS': True,
                'SAVE_PLOT_PDF': False, 'SAVE_PLOT_FIG': False, 'PRINT_INFO': False, 'LOAD_IRL': False,
                'SAVE_DATA_BC_EXPERT': True, 'SAVE_MODEL_BC': True}

Config_interference = {'AntennaGain': 100, 'Bandwidth': 50}
Config_Power = {'UE_Tr_power': 2.0, 'UAV_Tr_power': [50.0, 60.0, 80.0, 100.0, 150.0, 200.0], 'UAV_init_energy': 400.0,
                'UAV_mobility_consumption': 10.0}  # Tr power: mW, Energy, Jule
# [50.0, 60.0, 80.0, 100.0, 150.0, 200.0]
# [50.0, 80.0, 100.0, 150.0]

Config_IRL = {'NUM_FEATURES': 5, 'NUM_EPOCHS': 10002, 'NUM_PLAY': 1, 'NUM_TRAJECTORIES_EXPERT': 1,
              'TRAJECTORY_LENGTH': Config_requirement.get('dist_limit'), 'GAMMA_FEATURES': 0.999,
              'EPSILON_OPTIMIZATION': 0.01, 'LEARNING_RATE': 1e-3, 'BATCH_SIZE': 100, 'EPSILON_GREEDY': 0.1,
              'GAMMA_DISCOUNT': 0.9}

Config_BehavioralCloning = {'NUM_TRAJECTORIES_EXPERT': 100000}

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
ExpertPath_BC = "Data/BehavioralCloning/"
Config_Path = {'PathDist': pathDist, 'ExpertPath': ExpertPath, 'WeightPath': WeightPath, 'InverseRLPath': InverseRLPath,
               'ResultPathPDF': ResultPathPDF, 'ResultPathFIG': ResultPathFIG, 'SGDModelPath': SGDModelPath,
               'ExpertPath_BC': ExpertPath_BC}

"""
#################################
# Configuration File
#################################
"""

Mode = 'Random'
# Different Modes {"Expert", "IRL", "DRL", "QRL", "BC", "Random", "ResultsIRL", "ResultsDRL", "ResultsQRL",
# "ResultsRand"}

# Possible number of UEs: 75
# Possible number of Cells: 25
Config_General = {'NUM_UAV': 1, 'Size': 3, 'NUM_CELLS': 25, 'NUM_UEs': 75, 'Radius': 10, 'Loc_delta': 2}
Config_FLags = {'SAVE_path': True, 'Display_map': True}
Config_Power = {'UE_Tr_power': 10, 'UAV_Tr_power': [10, 20, 40, 50, 80, 100], 'UAV_init_energy': 400,
                'UAV_mobility_consumption': 10}  # Tr power: mW, Energy, Jule

Config_QRL = {}
Config_IRL = {}
Config_DRL = {}

pathDist = 'ConfigData/Cells_%d_Size_%d_UEs_%d' % (Config_General.get('NUM_CELLS'), Config_General.get('Size'),
                                                   Config_General.get('NUM_UEs'))
Config_Path = {'PathDist': pathDist}
# pathEnergy = 'ConfigData/Energy_UE_%d_Radius_%d' % (Config_General.get('NUM_UE'), Radius)
# Config_Path = {'PathDist': pathDist, 'pathEnergy': pathEnergy}


"""
#################################
# Configuration File
#################################
"""

Mode = 'Expert'
# Different Modes {"Expert", "IRL", "DRL", "ResultsIRL", "ResultsDRL"}

Config_General = {'NUM_UAV': 1, 'Size': 3, 'NUM_CELLS': 9, 'NUM_UEs': 30}
Config_FLags = {'SAVE_path': True, 'Display_map': True}
Config_Power = {'Transmission_Power': 10}  # 10mW

Config_IRL = {}
Config_DRL = {}

pathDist = 'ConfigData/Cells_%d_Size_%d_UEs_%d' % (Config_General.get('NUM_CELLS'), Config_General.get('Size'),
                                                   Config_General.get('NUM_UEs'))
Config_Path = {'PathDist': pathDist}
# pathEnergy = 'ConfigData/Energy_UE_%d_Radius_%d' % (Config_General.get('NUM_UE'), Radius)
# Config_Path = {'PathDist': pathDist, 'pathEnergy': pathEnergy}


"""
#################################
# Inverse Reinforcement Learning
#################################
"""

#########################################################
# import libraries
import numpy as np
from config import Config_IRL
from config import Config_Path
from config import Config_requirement

#########################################################
# General Parameters
num_epochs = Config_IRL.get('NUM_EPOCHS')
ExpertPath = Config_Path.get('ExpertPath')
num_features = Config_IRL.get('NUM_FEATURES')
epsilon = Config_IRL.get('EPSILON_OPTIMIZATION')
dist_limit = Config_requirement.get('dist_limit')
num_trajectories = Config_IRL.get('NUM_TRAJECTORIES_EXPERT')

#########################################################
# Function definition


def inverse_rl(uav, ues_objects, ax_objects, cell_objects):
    expert_policy_feature_expectation = load_expert_feature_expectation()
    random_policy_feature_expectation = np.asarray([4.75239812, 3.12983145, 0.12987357, 0.98712345, 6.90207523])

    random_initial_t = np.linalg.norm(expert_policy_feature_expectation - random_policy_feature_expectation)
    pass


def load_expert_feature_expectation():
    file_name = '%d_trajectories_%d_length.npz' % (num_trajectories, dist_limit)
    readfile = np.load(ExpertPath + file_name, allow_pickle=True)
    # 'arr_0' = readfile.files[0]
    trajectories = readfile["arr_0"]
    sum_expert_feature_expectation = 0
    for trajectory in trajectories:
        sum_expert_feature_expectation += trajectory[-1]
    expert_feature_expectation = sum_expert_feature_expectation / num_trajectories
    return expert_feature_expectation

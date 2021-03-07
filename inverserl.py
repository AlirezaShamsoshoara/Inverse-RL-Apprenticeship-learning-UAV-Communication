"""
#################################
# Inverse Reinforcement Learning
#################################
"""

#########################################################
# import libraries
import numpy as np
from cvxopt import matrix
from cvxopt import solvers
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
    optimization(expert_policy_feature_expectation, random_policy_feature_expectation)
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


def optimization(policy_expert, policy_agent):
    # https://cvxopt.org/examples/tutorial/qp.html
    # https://cvxopt.org/userguide/coneprog.html#quadratic-programming
    """
    To have more information about this Quadratic programming please read these topics:

    :param policy_expert:
    :param policy_agent:
    :return:
    """
    length = len(policy_expert)
    p = matrix(2.0 * np.eye(length), tc='d')
    q = matrix(np.zeros(length), tc='d')

    policy_subject = [policy_expert]
    h_subject = [1]
    for policy in policy_agent:
        policy_subject.append(policy)
        h_subject.append(1)

    policy_subject_mat = np.array(policy_subject)
    policy_subject_mat[0] = -1 * policy_subject_mat[0]

    g = matrix(policy_subject_mat, tc='d')
    h = matrix(-np.array(h_subject), tc='d')
    solution_weights = solvers.qp(p, q, g, h)

    pass

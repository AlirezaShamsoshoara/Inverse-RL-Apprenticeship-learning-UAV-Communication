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
WeightPath = Config_Path.get('WeightPath')
num_features = Config_IRL.get('NUM_FEATURES')
epsilon = Config_IRL.get('EPSILON_OPTIMIZATION')
dist_limit = Config_requirement.get('dist_limit')
num_trajectories = Config_IRL.get('NUM_TRAJECTORIES_EXPERT')

#########################################################
# Function definition


def inverse_rl(uav, ues_objects, ax_objects, cell_objects):
    weight_list = []
    solution_list = []
    iter_optimization = 0
    weight_file_name_txt = 'weights_features_%d.txt' % num_features
    weight_file = open(WeightPath + weight_file_name_txt, 'w')

    expert_policy_feature_expectation = load_expert_feature_expectation()
    learner_policy_feature_expectation = [[4.75239812, 3.12983145, 0.12987357, 0.98712345, 6.90207523]]

    random_initial_t = np.linalg.norm(expert_policy_feature_expectation -
                                      np.array(learner_policy_feature_expectation[0]))
    weights, weights_norm, solution = optimization(expert_policy_feature_expectation,
                                                   learner_policy_feature_expectation)
    print("Optimization status is: ", solution.get('status'))
    if solution.get('status') == "optimal":
        weight_list.append((weights, weights_norm))
        solution_list.append(solution)
        weight_file.write(str(weight_list[-1]))

    # TODO(1): Run another simulation based on the new weights to update the learner policy (Feature expectation policy)

    # TODO: Update the learner policy (Feature expectation policy) and calculate the hyper distance between the current
    # TODO: (Contd) learner policy (Feature expectation policy) and the expert policy (Feature expectation policy).
    print("Hyper Distance = ",)
    # TODO: If the distance is less than a threshold, then break the optimization and report the optimal weights else
    # TODO: (Contd) go to TODO(1)

    # TODO: Run the last simulation with the optimal weights for the evaluation and result comparison with other methods

    weight_file.close()
    weight_file_name_np = 'weights_iter_%d_features_%d' % (iter_optimization, num_features)
    np.savez(WeightPath + weight_file_name_np, weight_list=weight_list, solution_list=solution_list)


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


def optimization(policy_expert, policies_agent):
    # https://cvxopt.org/examples/tutorial/qp.html
    # https://cvxopt.org/userguide/coneprog.html#quadratic-programming
    """
    To have more information about this Quadratic programming please read these topics:

    :param policy_expert:
    :param policies_agent:
    :return:
    """
    length = len(policy_expert)
    p = matrix(2.0 * np.eye(length), tc='d')
    q = matrix(np.zeros(length), tc='d')

    policy_subject = [policy_expert]
    h_subject = [1]
    for policy in policies_agent:
        policy_subject.append(policy)
        h_subject.append(1)

    policy_subject_mat = np.array(policy_subject)
    policy_subject_mat[0] = -1 * policy_subject_mat[0]

    g = matrix(policy_subject_mat, tc='d')
    h = matrix(-np.array(h_subject), tc='d')
    solution_weights = solvers.qp(p, q, g, h)
    if solution_weights['status'] == "optimal":
        weights = np.squeeze(np.asarray(solution_weights['x']))
        weights_normalized = weights / np.linalg.norm(weights)
        return weights, weights_normalized, solution_weights
    else:
        return None, None, solution_weights

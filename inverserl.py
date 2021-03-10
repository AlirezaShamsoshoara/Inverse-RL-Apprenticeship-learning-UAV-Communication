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
from config import Config_Power
from config import Config_General
from tensorflow.keras import Input
from config import Config_requirement
from config import movement_actions_list
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Dense, Activation, Dropout

#########################################################
# General Parameters
action_list = []
BATCH_SIZE = Config_IRL.get('BATCH_SIZE')
NUM_EPOCHS = Config_IRL.get('NUM_EPOCHS')
INIT_LR = Config_IRL.get('LEARNING_RATE')
ExpertPath = Config_Path.get('ExpertPath')
WeightPath = Config_Path.get('WeightPath')
num_states = Config_General.get('NUM_CELLS')
tx_powers = Config_Power.get('UAV_Tr_power')
num_features = Config_IRL.get('NUM_FEATURES')
epsilon_grd = Config_IRL.get('EPSILON_GREEDY')
dist_limit = Config_requirement.get('dist_limit')
epsilon_opt = Config_IRL.get('EPSILON_OPTIMIZATION')
num_trajectories = Config_IRL.get('NUM_TRAJECTORIES_EXPERT')

num_required_replays = NUM_EPOCHS / 10
for i in range(len(tx_powers) * len(movement_actions_list)):
    action_list.append(i)
action_array = np.array(action_list, dtype=np.int8)

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
    # TODO: To run another simulation we can have simple Q learning model or a deep inverse reinforcement learning one

    model = build_neural_network()
    learner(model, weights_norm)

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


def learner(model, weights):
    episode = 0
    replay = []

    while episode < NUM_EPOCHS:

        if episode >= num_required_replays:
            # do training the model
            pass
        episode += 1

    return model


def build_neural_network():
    input_dim = num_states
    model = Sequential()
    model.add(Input(shape=(input_dim, )))
    # First Layer
    model.add(Dense(units=100, activation='relu', kernel_initializer='lecun_uniform'))

    # Second Layer
    model.add(Dense(units=100, activation='relu', kernel_initializer='lecun_uniform'))

    # Output Layer
    model.add(Dense(units=len(action_list), activation='linear', kernel_initializer='lecun_uniform'))
    opt = Adam(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
    model.compile(optimizer=opt, loss='mse', metrics=["accuracy"])
    return model

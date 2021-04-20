"""
#################################
# PLOT Module for demonstrating the results
#################################
"""

#########################################################
# import libraries
import pickle
import numpy as np
from config import Config_IRL
from config import Config_Path
import matplotlib.pyplot as plt
from config import Config_Flags


#########################################################
# General Parameters
num_features = Config_IRL.get('NUM_FEATURES')
ResultPathPDF = Config_Path.get('ResultPathPDF')
ResultPathFIG = Config_Path.get('ResultPathFIG')
#########################################################
# Function definition


def plot_reward_irl_sgd(trajectories, learner_index):
    # Trajectories:
    #           *: All Trajectories
    #                   1) Current Feature State
    #                   2) Current Specification (interference_on_UAV, SINR, throughput, interference_on_UEs)
    #                   3) Action
    #                   4) Next Feature State
    #                   5) Next Specification (interference_on_UAV, SINR, throughput, interference_on_UEs)
    #                   6) Immediate Reward (reward_location = 5 in the array)
    #                   7) Learner Feature Expectation (Immediate)
    #                   8) Learner Feature Expectation (Final)
    #           *: Final SGD Models for all Q-Action estimator
    #           *: Learner Index
    #
    #           INFO: Len(Trajectories) = NUM_EPOCHS(NUM EPISODES) + 1(SGD Models) + 1(Learner Index)
    #           INFO: Len(Trajectory)   = NUM Distance(max = dist_limit: 8) + 1 (learner_feature_expectation)
    #           INFO: Len(Each Step)    = NUM Elements(7)
    num_epochs = len(trajectories[0:-2])
    accumulative_reward = np.zeros(num_epochs, dtype=float)
    episode = 0
    reward_location = 5  # The reward location in each step of the trajectory array.

    for trajectory in trajectories[0:-2]:
        for step in trajectory[0:-1]:
            accumulative_reward[episode] += step[reward_location]
        episode += 1

    fig_reward = plt.figure(figsize=(8, 8))
    ax_reward = fig_reward.add_subplot(111)
    ax_reward.set_xlabel("EPOCHS", size=12, fontweight='bold')
    ax_reward.set_ylabel("Accumulative Reward", size=12, fontweight='bold')
    ax_reward.plot(np.arange(0, num_epochs) + 1, accumulative_reward, color="blue", linestyle='--', marker='o',
                   markersize='5', label='Accumulative Reward _ EPOCHs)', linewidth=2)
    ax_reward.grid(True)
    ax_reward.legend(prop={'size': 14, 'weight': 'bold'}, loc='best')

    file_fig_obj = ResultPathFIG + 'SGD_accumulative_reward_training_Feature_%d_learner_%d_EPOCHS_%d.fig.pickle' % \
                   (num_features, learner_index, num_epochs)
    file_fig_pdf = ResultPathPDF + 'SGD_accumulative_reward_training_Feature_%d_learner_%d_EPOCHS_%d.pdf' % \
                   (num_features, learner_index, num_epochs)

    if Config_Flags.get("SAVE_PLOT_PDF"):
        fig_reward.savefig(file_fig_pdf, bbox_inches='tight')
    if Config_Flags.get("SAVE_PLOT_FIG"):
        pickle.dump(fig_reward, open(file_fig_obj, 'wb'))


def plot_reward_irl_dqn(trajectories, learner_index):
    # trajectory.append((features_current_state, (interference, sinr, throughput, interference_ues), action,
    #                    features_next_state, (interference_next, sinr_next, throughput_next,
    #                                          interference_ues_next),
    #                    immediate_reward, deepcopy(learner_feature_expectation)))
    # trajectory.append(learner_feature_expectation)
    # trajectories.append(model)
    # trajectories.append(learner_index)
    # Trajectories:
    #           *: All Trajectories
    #                   1) Current Feature State
    #                   2) Current Specification (interference_on_UAV, SINR, throughput, interference_on_UEs)
    #                   3) Action
    #                   4) Next Feature State
    #                   5) Next Specification (interference_on_UAV, SINR, throughput, interference_on_UEs)
    #                   6) Immediate Reward (reward_location = 5 in the array)
    #                   7) Learner Feature Expectation (Immediate)
    #                   8) Learner Feature Expectation (Final)
    #           *: Final SGD Models for all Q-Action estimator
    #           *: Learner Index
    #
    #           INFO: Len(Trajectories) = NUM_EPOCHS(NUM EPISODES) + 1(SGD Models) + 1(Learner Index)
    #           INFO: Len(Trajectory)   = NUM Distance(max = dist_limit: 8) + 1 (learner_feature_expectation)
    #           INFO: Len(Each Step)    = NUM Elements(8)
    num_epochs = len(trajectories[0:-2])
    accumulative_reward = np.zeros(num_epochs, dtype=float)
    episode = 0
    reward_location = 5  # The reward location in each step of the trajectory array.

    for trajectory in trajectories[0:-2]:
        for step in trajectory[0:-1]:
            accumulative_reward[episode] += step[reward_location]
        episode += 1

    fig_reward = plt.figure(figsize=(8, 8))
    ax_reward = fig_reward.add_subplot(111)
    ax_reward.set_xlabel("EPOCHS", size=14, fontweight='bold')
    ax_reward.set_ylabel("Accumulative Reward", size=14, fontweight='bold')
    ax_reward.plot(np.arange(0, num_epochs) + 1, accumulative_reward, color="blue", linestyle='--', marker='o',
                   markersize='5', label='Accumulative Reward - EPOCHs)', linewidth=2)
    ax_reward.grid(True)
    ax_reward.legend(prop={'size': 14, 'weight': 'bold'}, loc='best')

    file_fig_obj = ResultPathFIG + 'DQN_accumulative_reward_training_Feature_%d_learner_%d_EPOCHS_%d.fig.pickle' % \
                   (num_features, learner_index, num_epochs)
    file_fig_pdf = ResultPathPDF + 'DQN_accumulative_reward_training_Feature_%d_learner_%d_EPOCHS_%d.pdf' % \
                   (num_features, learner_index, num_epochs)

    if Config_Flags.get("SAVE_PLOT_PDF"):
        fig_reward.savefig(file_fig_pdf, bbox_inches='tight')
    if Config_Flags.get("SAVE_PLOT_FIG"):
        pickle.dump(fig_reward, open(file_fig_obj, 'wb'))


def plot_training_trajectories(trajectories_sgd_run, trajectories_dqn_run, cell_objects):
    # trajectory.append((features_current_state, (interference, sinr, throughput, interference_ues), action,
    #                    features_next_state, (interference_next, sinr_next, throughput_next,
    #                                          interference_ues_next),
    #                    immediate_reward, deepcopy(learner_feature_expectation)))
    # trajectory.append(learner_feature_expectation)
    # trajectories.append(model)
    # trajectories.append(learner_index)
    # Trajectories:
    #           *: All Trajectories
    #                   1) Current Feature State
    #                   2) Current Specification (interference_on_UAV, SINR, throughput, interference_on_UEs)
    #                   3) Action
    #                   4) Next Feature State
    #                   5) Next Specification (interference_on_UAV, SINR, throughput, interference_on_UEs)
    #                   6) Immediate Reward (reward_location = 5 in the array)
    #                   7) New cell ID
    #
    #           INFO: Len(Trajectories) = NUM_EPOCHS(NUM EPISODES)
    #           INFO: Len(Trajectory)   = NUM Distance(max = dist_limit: 8)
    #           INFO: Len(Each Step)    = NUM Elements(7)
    num_epochs = len(trajectories_sgd_run[0])
    num_runs = len(trajectories_sgd_run)
    throughput_sgd = np.zeros((num_runs, num_epochs), dtype=float)
    throughput_dqn = np.zeros((num_runs, num_epochs), dtype=float)
    interference_ue_sgd = np.zeros((num_runs, num_epochs), dtype=float)
    interference_ue_dqn = np.zeros((num_runs, num_epochs), dtype=float)
    distance_destination_sgd = np.zeros((num_runs, num_epochs), dtype=int)
    distance_destination_dqn = np.zeros((num_runs, num_epochs), dtype=int)
    next_spec_index = 4
    throughput_index = 2
    interference_ue_index = 3
    new_cell_index = 6

    run = 0
    for trajectories_sgd, trajectories_dqn in zip(trajectories_sgd_run, trajectories_dqn_run):
        episode = 0
        for trajectory_sgd, trajectory_dqn in zip(trajectories_sgd, trajectories_dqn):
            distance_destination_sgd[run, episode] = cell_objects[trajectory_sgd[-1][new_cell_index]].get_distance()
            distance_destination_dqn[run, episode] = cell_objects[trajectory_dqn[-1][new_cell_index]].get_distance()
            for step_sgd, step_dqn in zip(trajectory_sgd, trajectory_dqn):
                throughput_sgd[run, episode] += step_sgd[next_spec_index][throughput_index]
                throughput_dqn[run, episode] += step_dqn[next_spec_index][throughput_index]
                interference_ue_sgd[run, episode] += step_sgd[next_spec_index][interference_ue_index]
                interference_ue_dqn[run, episode] += step_dqn[next_spec_index][interference_ue_index]
            episode += 1
        run += 1

    fig_train_throughput = plt.figure(figsize=(8, 8))
    ax_train_throughput = fig_train_throughput.add_subplot(111)
    ax_train_throughput.set_xlabel("EPOCHS", size=14, fontweight='bold')
    ax_train_throughput.set_ylabel("Average throughput UAV up link (Mbps)", size=14, fontweight='bold')
    ax_train_throughput.plot(100*np.arange(0, num_epochs) + 1, np.mean(throughput_sgd, axis=0), color="blue", linestyle='--', marker='o',
                                markersize='5', label='(Q-Learning))', linewidth=2)
    ax_train_throughput.plot(100*np.arange(0, num_epochs) + 1, np.mean(throughput_dqn, axis=0), color="red", linestyle='--', marker='x',
                             markersize='5', label='(DQN))', linewidth=2)
    ax_train_throughput.grid(True)
    ax_train_throughput.legend(prop={'size': 14, 'weight': 'bold'}, loc='best')

    file_fig_obj_throughput = ResultPathFIG + 'Throughput_learning_epochs_%d.fig.pickle' % \
                                num_epochs
    file_fig_pdf_throughput = ResultPathPDF + 'Throughput_learning_epochs_%d.pdf' % \
                   num_epochs

    fig_train_interference = plt.figure(figsize=(8, 8))
    ax_train_interference = fig_train_interference.add_subplot(111)
    ax_train_interference.set_xlabel("EPOCHS", size=14, fontweight='bold')
    ax_train_interference.set_ylabel("Summation of interference on UEs", size=14, fontweight='bold')
    ax_train_interference.plot(100*np.arange(0, num_epochs) + 1, np.mean(interference_ue_sgd, axis=0), color="blue", linestyle='--',
                               marker='o', markersize='5', label='(Q-Learning))', linewidth=2)
    ax_train_interference.plot(100*np.arange(0, num_epochs) + 1, np.mean(interference_ue_dqn, axis=0), color="red", linestyle='--',
                               marker='x', markersize='5', label='(DQN))', linewidth=2)
    ax_train_interference.grid(True)
    ax_train_interference.legend(prop={'size': 14, 'weight': 'bold'}, loc='best')

    file_fig_obj_interference = ResultPathFIG + 'Interference_learning_epochs_%d.fig.pickle' % \
                              num_epochs
    file_fig_pdf_interference = ResultPathPDF + 'Interference_learning_epochs_%d.pdf' % \
                              num_epochs

    fig_train_distance = plt.figure(figsize=(8, 8))
    ax_train_distance = fig_train_distance.add_subplot(111)
    ax_train_distance.set_xlabel("EPOCHS", size=14, fontweight='bold')
    ax_train_distance.set_ylabel("Average distance to the destination", size=14, fontweight='bold')
    ax_train_distance.plot(100*np.arange(0, num_epochs) + 1, np.mean(distance_destination_sgd, axis=0), color="blue", linestyle='--',
                               marker='o', markersize='5', label='(Q-Learning))', linewidth=2)
    ax_train_distance.plot(100*np.arange(0, num_epochs) + 1, np.mean(distance_destination_dqn, axis=0), color="red", linestyle='--',
                               marker='x', markersize='5', label='(DQN))', linewidth=2)
    ax_train_distance.grid(True)
    ax_train_distance.legend(prop={'size': 14, 'weight': 'bold'}, loc='best')

    file_fig_obj_distance = ResultPathFIG + 'Distance_learning_epochs_%d.fig.pickle' % \
                                num_epochs
    file_fig_pdf_distance = ResultPathPDF + 'Distance_learning_epochs_%d.pdf' % \
                                num_epochs

    if Config_Flags.get("SAVE_PLOT_PDF"):
        fig_train_throughput.savefig(file_fig_pdf_throughput, bbox_inches='tight')
        fig_train_interference.savefig(file_fig_pdf_interference, bbox_inches='tight')
        fig_train_distance.savefig(file_fig_pdf_distance, bbox_inches='tight')
    if Config_Flags.get("SAVE_PLOT_FIG"):
        pickle.dump(fig_train_throughput, open(file_fig_obj_throughput, 'wb'))
        pickle.dump(fig_train_interference, open(file_fig_obj_interference, 'wb'))
        pickle.dump(fig_train_distance, open(file_fig_obj_distance, 'wb'))

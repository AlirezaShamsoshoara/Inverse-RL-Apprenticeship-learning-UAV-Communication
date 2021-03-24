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
from config import Config_FLags


#########################################################
# General Parameters
num_features = Config_IRL.get('NUM_FEATURES')
ResultPathPDF = Config_Path.get('ResultPathPDF')
ResultPathFIG = Config_Path.get('ResultPathFIG')
#########################################################
# Function definition


def plot_reward_irl(trajectories, learner_index):
    # Trajectories:
    #           *: All Trajectories
    #                   1) Current Feature State
    #                   2) Current Specification (interference_on_UAV, SINR, throughput, interference_on_UEs)
    #                   3) Action
    #                   4) Next Feature State
    #                   5) Next Specification (interference_on_UAV, SINR, throughput, interference_on_UEs)
    #                   6) Immediate Reward (reward_location = 5 in the array)
    #                   7) Learner Feature Expectation
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

    file_fig_obj = ResultPathFIG + 'accumulative_reward_training_Feature_%d_learner_%d_EPOCHS_%d.fig.pickle' % \
                   (num_features, learner_index, num_epochs)
    file_fig_pdf = ResultPathPDF + 'accumulative_reward_training_Feature_%d_learner_%d_EPOCHS_%d.pdf' % \
                   (num_features, learner_index, num_epochs)

    if Config_FLags.get("SAVE_PLOT_PDF"):
        fig_reward.savefig(file_fig_pdf, bbox_inches='tight')
    if Config_FLags.get("SAVE_PLOT_FIG"):
        pickle.dump(fig_reward, open(file_fig_obj, 'wb'))

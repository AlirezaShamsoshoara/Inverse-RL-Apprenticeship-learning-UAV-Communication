"""
#################################
# Evaluation of models and approaches
#################################
"""

#########################################################
# import libraries
import pickle
import numpy as np
from config import Config_Path
from config import Config_Flags
import matplotlib.pyplot as plt

#########################################################
# General Parameters
ResultPathPDF = Config_Path.get('ResultPathPDF')
ResultPathFIG = Config_Path.get('ResultPathFIG')

#########################################################
# Function definition


def inverse_rl_hyper_distance():
    hyper_distance = [1.4808814477687855, 0.1947743531591942, 0.20178608693770217, 0.37417049598206165,
                      0.37853236096734755,
                      0.33176850191358415, 0.30923747765876247, 0.49682809809649375, 0.501600667872273,
                      0.05795836140829434, 0.13261291392221486, 0.17347945192419448]
    threshold = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    iteration_range = np.arange(0, len(hyper_distance))
    fig_hyper_distance = plt.figure(figsize=(8, 8))
    ax_hyper_distance = fig_hyper_distance.add_subplot(111)
    ax_hyper_distance.set_xlabel("Optimization Iteration", size=14, fontweight='bold')
    ax_hyper_distance.set_ylabel("Distance to expert feature distribution", size=14, fontweight='bold')

    ax_hyper_distance.plot(iteration_range, hyper_distance, color="black", linestyle='-', marker='o',
                           markersize='5', label='Hyper distance', linewidth=2)
    ax_hyper_distance.plot(iteration_range, threshold, color="blue", linestyle='--', label='Threshold',
                           linewidth=2)
    circle = plt.Circle((9, 0.05795836140829434), radius=0.1, color='r', alpha=0.5)
    ax_hyper_distance.add_artist(circle)
    ax_hyper_distance.grid(True)
    ax_hyper_distance.legend(prop={'size': 14, 'weight': 'bold'}, loc='best')
    file_fig_obj = "Hyper_distance.fig.pickle"
    file_fig_pdf = "Hyper_distance.pdf"

    if Config_Flags.get("SAVE_PLOT_PDF"):
        fig_hyper_distance.savefig(ResultPathPDF + file_fig_pdf, bbox_inches='tight')
    if Config_Flags.get("SAVE_PLOT_FIG"):
        pickle.dump(fig_hyper_distance, open(ResultPathFIG + file_fig_obj, 'wb'))

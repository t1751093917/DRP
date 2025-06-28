import pandas as pd
import matplotlib.pyplot as plt

ratio_x = [0.1, 0.15, 0.2, 0.25, 0.3]
ratio_y_100 = [[47.88, 47.79, 47.65, 48.07, 47.72],
               [35.355, 35.53, 36.58, 36.37, 37.36],
               [14.207, 16.277, 17.753, 19.91, 20.853]]
ratio_y_80 = [[50.783, 51.491, 51.763, 51.157, 49.306],
              [36.992, 37.154, 38.188, 37.683, 37.725],
              [13.625, 16.483, 18.896, 20.504, 22.371]]

lambda_x = [0.05, 0.5, 1, 4, 20]
lambda_y =[[47.35, 47.73, 47.01, 48.01, 47.54],
           [35.854, 36.0975, 35.648, 35.92, 35.5025],
           [12.43, 12.73, 15.17, 17.69, 18.07]]

delta_x = [1000, 2500, 5000, 7500, 10000]
for i, num in enumerate(delta_x):
    delta_x[i] /= 50000
delta_y = [[47.86, 47.54, 47.9, 48.05, 47.64],
           [35.54, 35.687, 35.92, 35.51, 35.88],
           [17.66, 17.97, 17.8, 18.43, 18.66]]


x_rate = [ratio_x]
x_ls = [lambda_x, delta_x]
x_ticks_rate = [[1, 2, 3, 4, 5]]
x_ticks = [[1, 2, 3, 4, 5],
           [1, 5, 10, 15, 20],
           [5, 8, 15, 25, 35]]

color_rate = ['blue']
tag_rate = ['ratio']
colors = ['green', 'orange', 'red', 'blue']
# tags = ['α', 'η']
tags = ['ratio', 'ratio']
datasets = ['Symm20%', 'Asym40%', 'Symm80%']
base_acc_100 = [46.73, 35, 11.11]
base_acc_80 = [46.725, 33.638, 10.575]
base_acc = [base_acc_100, base_acc_80]
title_horizon = [0.30, 0.5, 0.7]
title_vertical = [0.95, 0.64, 0.33]


plt.rcParams['font.family'] = 'Times New Roman'
fig, axes = plt.subplots(len(datasets), len(tags), figsize=(6, 5))
# fig, axes = plt.subplots(len(tag_rate), len(datasets), figsize=(6, 1.8))
for j, dataset in enumerate(datasets):
    y_ls = [ratio_y_100[j], ratio_y_80[j]]
    # y_ls = [lambda_y[j], delta_y[j]]
    # plt.figure(figsize=(2, 2))
    for i, tag in enumerate(tags):
        ax = axes[j,i]
        ax.plot(x_ticks[0], y_ls[i], label=tag, color=colors[-1], linewidth=1, marker='o')
        y_mid = (max(y_ls[i]) + min(y_ls[i])) / 2
        # if max(y_ls[i]) - min(y_ls[i]) < 1.5:
        #     ax.set_ylim(y_mid-0.75, y_mid+0.75)
        ax.set_xticks(x_ticks[0])
        # ax.set_xticklabels(x_ls[i])
        ax.axhline(y=base_acc[i][j], color='red', linestyle='--', linewidth=1)  # , xmin=x_ticks_rate[i][0], xmax=x_ticks_rate[i][-1]
        # ax.plot(x_ticks_rate[i], y_rate[i], label=tag, color=color_rate[i], linewidth=1, marker='o')
        # ax.set_xticks(x_ticks_rate[i])
        ax.set_xticklabels(x_rate[0])
        ax.set_xlabel(tag)
        ax.set_ylabel("Performances(%)")
        # if i == 0:
        #     ax.set_title(datasets[j])
        # plt.legend()
        ax.grid(True)
    fig.text(0.5, title_vertical[j]-0.01, datasets[j], ha='center', va='center')  #, fontsize=14, fontweight='bold'
fig.text(0.32, 0.95, "CIFAR100N", ha='center', va='center')
fig.text(0.76, 0.95, "CIFAR80N", ha='center', va='center')

# plt.title("Multiple Line Segments", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(hspace=0.65, wspace=0.4, top=0.92-0.01, bottom=0.12-0.01, left=0.15, right=0.93)  #
plt.savefig('C:\\Users\\JSW\\Desktop\\results\\fig-ablation-ratio2' + '.eps')
# plt.show()



# Noise-Aware Pruning
# ratio_x = [0.1, 0.15, 0.2, 0.25, 0.3]
# ratio_y = [[46.99, 47.39, 47.97, 48.2, 47.91],
#            [45.75, 47.10, 47.10, 47.35, 47.85],
#            [36.32, 36.11, 36.25, 36.39, 36.00],
#            [36.79, 36.36, 37.21, 38.26, 37.01]]
#
# lambda_x = [0.05, 0.5, 1, 4, 20]
# lambda_y =[[47.83, 47.97, 47.83, 48.16, 47.96],
#            [46.95, 46.82, 47.95, 48.44, 48.83],
#            [36.05, 36.10, 35.79, 35.76, 36.31],
#            [36.98, 37.23, 37.11, 37.20, 37.06]]
#
# delta_x = [20, 50, 200, 500, 1000]
# for i, num in enumerate(delta_x):
#     delta_x[i] /= 50000
# delta_y = [[47.73, 48.13, 47.92, 47.99, 47.84],
#            [48.07, 47.86, 47.51, 47.44, 47.23],
#            [35.96, 36.18, 35.86, 35.53, 35.94],
#            [37.62, 37.88, 37.44, 37.43, 37.67]]
#
# k_x = [5, 10, 20, 50, 80]
# k_y = [[48.01, 48.09, 48.10, 47.71, 47.83],
#        [48.00, 47.27, 47.87, 47.88, 47.88],
#        [35.80, 36.09, 35.92, 36.01, 35.92],
#        [37.81, 37.03, 37.36, 37.11, 37.25]]
#
# x_rate = [ratio_x]
# x_ls = [lambda_x, delta_x, k_x]
# x_ticks_rate = [[1, 2, 3, 4, 5]]
# x_ticks = [[1, 2, 3, 4, 5],
#            [1, 5, 10, 15, 20],
#            [5, 8, 15, 25, 35]]




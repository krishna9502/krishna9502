import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle
import numpy as np

PICKLE_FILE = 'output.pickle'

pickle_in = open(PICKLE_FILE, 'rb')
out = pickle.load(pickle_in)

[filt1, filt2, bias1, bias2, theta3, bias3, cost, acc] = out


####### Plotting the cost and accuracy over different background

gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1]) 
ax0 = plt.subplot(gs[0])
line0, = ax0.plot(cost, color='b')
ax1 = plt.subplot(gs[1], sharex = ax0)
line1, = ax1.plot(acc, color='r', linestyle='--')
plt.setp(ax0.get_xticklabels(), visible=False)
ax0.legend((line0, line1), ('Loss', 'Accuracy'), loc='upper right')

######## remove vertical gap between subplots

plt.subplots_adjust(hspace=.0)
plt.show(block=False)

input("Press Enter to continue...")


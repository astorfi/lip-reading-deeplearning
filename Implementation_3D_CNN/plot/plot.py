from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

eer_select = np.array([25, 22, 19.8, 18.5, 17.1,16.4,16.1,15.8,15.6,15.5,15.4,15.3,15.3,15.3,15.3])
eer_nonselect = np.array([21, 16.1, 14.5, 13.2, 12.8, 12.35, 12.1, 11.8, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5])
epoch = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
xnew = np.linspace(0, 10, num=41, endpoint=True)

spl_select = UnivariateSpline(epoch, eer_select)
spl_nonselect = UnivariateSpline(epoch, eer_nonselect)


fig = plt.figure()
plt.plot(epoch,eer_select, label="whole training data",marker='o', ls='-', ms=6,markevery=1)
plt.plot(epoch, eer_nonselect, label="online pair selection",marker='*', ls='-', ms=8,markevery=1)
plt.legend(loc='best', borderaxespad=1.,prop={'size':15})

plt.xlabel('Epoch', fontsize=15)
plt.ylabel('EER', fontsize=15)
plt.grid(True)

plt.show()
fig.savefig('convergence-speed.jpg')
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

FPR = pickle.load( open( "fpr.p", "rb" ) )
TPR = pickle.load( open( "tpr.p", "rb" ) )

fig = plt.figure()

# Calculating EER
EER=[]
num = [0]
for i in num:
    fpr = np.asarray(FPR[i])
    tpr = np.asarray(TPR[i])
    intersect_x = fpr[np.abs(fpr - (1 - tpr)).argmin(0)]
    eer_raw = intersect_x * 100
    EER_NUM = '%.2f' % eer_raw

    EER.append(EER_NUM)

print(len(EER))

plt.plot(FPR[num[0]],TPR[num[0]], label="MFCC-1ch[4]"+',EER=' + str(EER[0]) + '%',marker='o', ls='-', ms=12,markevery=100)
# plt.plot(FPR[num[1]], TPR[num[1]], label="MFCC-3ch"+',EER=' + str(EER[1]) + '%',marker='*', ls='-', ms=12,markevery=100)
# plt.plot(FPR[num[2]], TPR[num[2]], label="MFEC-1ch"+',EER=' + str(EER[2]) + '%',marker='x', ls='-', ms=12,markevery=100)
# plt.plot(FPR[num[3]], TPR[num[3]], label="MFEC-3ch[ours]"+',EER=' + str(EER[3]) + '%',marker='s', ls='-', ms=12,markevery=100)

plt.legend(loc='best', borderaxespad=1.,prop={'size':40})
# plt.legend(bbox_to_anchor=(0.2, 0.85), loc='best', borderaxespad=1.,prop={'size':20})

plt.xlabel('False Acceptance Rate', fontsize=40)
plt.ylabel('True Positive Rate', fontsize=40)
plt.xlim([-.01,0.4])
plt.grid(True)

plt.tick_params(axis='both', which='major', labelsize=30)

plt.show()
fig.savefig('ROC_finetuning.jpg')
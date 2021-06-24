import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import tensorflow as tf 
#from matplotlib.image import imread
from scipy.special import expit, logit

import modules.Node as node

csv = pd.read_csv('./KOSPI.CSV')
print(csv.KOSPI[0:8])


#**********************************************************************************************************
# ARMA(p,q) based ML
#**********************************************************************************************************
# Hyper parameters...
epoch = 3
#----------------------------------------
p = 1 
q = 1
l = m = n = p

lr = 0.5
#----------------------------------------
DNN_p = 2
DNN_l = DNN_m = DNN_n = DNN_p

DNN_lr = 0.5
#----------------------------------------


# ARMA(p,q) based ML 
W0, W1, W2, W3, W4 = node.initWeightMatrix(p, q, l, m, n)

resultBuf = []

cnt=0
while cnt < epoch:
	for i in range(len(csv.KOSPI)-p-1):
		target = np.array(csv.KOSPI[i:i+q])
		x = csv.KOSPI[i+1:i+p+1]

		o1, o2, o3, o4, e, total_input = node.partialFoward(x, target,  W0, W2, W3, W4)
		W1, W2, W3, W4 = node.fullBackward(x, e, total_input, target, W1, W2, W3, W4, o1, o2, o3, o4, lr)

		o1, o2, o3, o4, e, total_input = node.fullForward(x, total_input, target, W1, W2, W3, W4)
		W1, W2, W3, W4 = node.fullBackward(x, e, total_input, target, W1, W2, W3, W4, o1, o2, o3, o4, lr)

		resultBuf.append(target[0] - o4[0])
	
	print("[ARMA-based DNN Epoch %d]" %(cnt))
	cnt += 1

print("[ARMA-based simulation result] : ", (target[0], o4[0]))
"""
print("\n[W1] : \n", W1)
print("\n[W2] : \n", W2)
print("\n[W3] : \n", W3)
print("\n[W4] : \n", W4)
"""

#**********************************************************************************************************
# general DNN-based ML
#**********************************************************************************************************

# general DNN 
W1, W2, W3, W4 = node.DNNinitWeightMatrix(DNN_p, DNN_l, DNN_m, DNN_n)

"""
print("[W1]\n", W1, end="\n\n")
print("[W2]\n", W2, end="\n\n")
print("[W3]\n", W3, end="\n\n")
print("[W4]\n", W4, end="\n\n")
"""

DNNresultBuf = []

cnt=0
while cnt < epoch:
	for i in range(len(csv.KOSPI)-DNN_p-1):

		target = np.array(csv.KOSPI[i])
		x = csv.KOSPI[i+1:i+DNN_p+1]
		x = np.array(x)

		o1, o2, o3, o4 = node.DNNForward(x, target, W1, W2, W3, W4)

		W1, W2, W3, W4 = node.DNNfullBackward(x, target, W1, W2, W3, W4, o1, o2, o3, o4, DNN_lr)

		DNNresultBuf.append(target - o4)

	cnt += 1
	print("[basic DNN Epoch %d]" %(cnt))


print("[DNN-based simulation result] : ", (target, o4[0]))

# graph
plt.plot(resultBuf, color="blue")
plt.plot(DNNresultBuf, color="red")
plt.ylabel("[Fitting Error]")
plt.xlabel("[Iteration]")
plt.title('ARMA based DNN vs. basic DNN')
plt.legend(['ARMA(' + str(p) + ',' + str(q) +') based DNN', 'Basic DNN with learning window ' + str(DNN_p)])

plt.show()



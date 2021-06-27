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
EpochLimit = 4
#----------------------------------------
p = 1 
q = 1
l = m = n = p
WINDOW_SIZE = p+1 # constraints : p > q

lr = 0.5
#----------------------------------------
DNN_p = 3
DNN_l = DNN_m = DNN_n = DNN_p
DNN_WINDOW_SIZE = DNN_p+1

DNN_lr = 0.5
#----------------------------------------


# ARMA(p,q) based ML 
W0, W1, W2, W3, W4 = node.initWeightMatrix(p, q, l, m, n)

resultBuf = []
targetBuf = []

cnt=0
while cnt < EpochLimit:
	for i in range(len(csv.KOSPI) - WINDOW_SIZE -1):
		buf = np.array(csv.KOSPI[i:i + WINDOW_SIZE])
		revBuf = np.flip(buf)

		target = np.array(revBuf[0:q])
		x = revBuf[1:p+1]
		"""
		print("[rev. Buf]", revBuf)
		print("[target]", target)
		print("[x]", x)
		"""

		"""
		target = np.array(csv.KOSPI[i:i+q])
		x = csv.KOSPI[i+1:i+p+1]
		"""

		o1, o2, o3, o4, e, total_input = node.partialFoward(x, target,  W0, W2, W3, W4)
		W1, W2, W3, W4 = node.fullBackward(x, e, total_input, target, W1, W2, W3, W4, o1, o2, o3, o4, lr)

		o1, o2, o3, o4, e, total_input = node.fullForward(x, total_input, target, W1, W2, W3, W4)
		W1, W2, W3, W4 = node.fullBackward(x, e, total_input, target, W1, W2, W3, W4, o1, o2, o3, o4, lr)

		# original series...
		targetBuf.append(o4[0])

		# resultBuf[0] meaningless...use this variable for the forecasting
		resultBuf.append(target[0])

	cnt += 1
	print("[ARMA-based DNN Epoch %d]" %(cnt))

#print("[ARMA-based simulation result] : ", (target[0], o4[0]))
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

cnt=0
while cnt < EpochLimit:
	for i in range(len(csv.KOSPI) - DNN_WINDOW_SIZE -1):
		buf = np.array(csv.KOSPI[i:i + DNN_WINDOW_SIZE])
		revBuf = np.flip(buf)

		x = revBuf[1:DNN_WINDOW_SIZE]
		target = revBuf[0]

		o1, o2, o3, o4 = node.DNNForward(x, target, W1, W2, W3, W4)

		W1, W2, W3, W4 = node.DNNfullBackward(x, target, W1, W2, W3, W4, o1, o2, o3, o4, DNN_lr)

		#print("[Gap]", target, o4[0])

	cnt += 1
	print("[basic DNN Epoch %d]" %(cnt))

#print("[DNN-based simulation result] : ", (target, o4[0]))

"""
print(W1)
print(W2)
print(W3)
print(W4)
"""

DNNForecastBuf = []
for k in range(52):
	#print("[MAIN : (target, x)]", target, x)
	target, x = node.DNNForecast((DNN_WINDOW_SIZE -1), target, x, W1, W2, W3, W4)
	DNNForecastBuf.append(target)

	print("\n[shifted x]\n", x)


print("[DNNForecastBuf]\n", DNNForecastBuf)

# graph
plt.plot(csv.KOSPI, color="grey")
plt.plot(DNNForecastBuf, color="navy")


"""
plt.plot(resultBuf, color="blue")
plt.plot(DNNResultBuf, color="green")
plt.plot(DNNForecastBuf, color="red")
"""

plt.ylabel("[Fitting]")
plt.xlabel("[Iteration]")
plt.title('ARMA based DNN vs. basic DNN')
#plt.legend(['KOSPI', 'ARMA(' + str(p) + ',' + str(q) +') based DNN', 'Basic DNN with learning window ' + str(DNN_p), 'DNN Forecast'])

#plt.show()



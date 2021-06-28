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
EpochLimit = 2
#----------------------------------------
p = 1 
q = 1
l = m = n = p
WINDOW_SIZE = p+1 # constraints : p > q

lr = 10
#----------------------------------------
DNN_p = 3
DNN_l = DNN_m = DNN_n = DNN_p
DNN_WINDOW_SIZE = DNN_p+1

WEEKS_PER_YEAR = 52
DNN_START_FORECASTING = 0
DNN_STOP_FORECASTING = 0

DNN_lr = 3
#----------------------------------------


# ARMA(p,q) based ML 
W0, W1, W2, W3, W4 = node.initWeightMatrix(p, q, l, m, n)

# original raw data
targetBuf = []

# fitting result
resultBuf = []
DNNresultBuf = []

# forecast result
DNNForecastBuf = []

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
		targetBuf.append(target[0])

		# resultBuf[0] meaningless...use this variable for the forecasting
		#resultBuf.append(o4[0])

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

DNN_STOP_FORECASTING  = EpochLimit*len(csv.KOSPI) - DNN_WINDOW_SIZE
DNN_START_FORECASTING = DNN_STOP_FORECASTING - WEEKS_PER_YEAR

cnt=0
obsNo = 0
while cnt < EpochLimit:
	# 1.fitting
	for i in range(len(csv.KOSPI) - DNN_WINDOW_SIZE -1):
		buf = np.array(csv.KOSPI[i:i + DNN_WINDOW_SIZE])
		revBuf = np.flip(buf)

		x = revBuf[1:DNN_WINDOW_SIZE]
		target = revBuf[0]

		o1, o2, o3, o4 = node.DNNForward(x, target, W1, W2, W3, W4)

		W1, W2, W3, W4 = node.DNNfullBackward(x, target, W1, W2, W3, W4, o1, o2, o3, o4, DNN_lr)

		# 2.forecasting
		if ((cnt == (EpochLimit-1)) & (obsNo == DNN_START_FORECASTING)):
			# start forecasting...
			forecast_x = x
			for k in range(WEEKS_PER_YEAR):

				forecasted, forecast_x = node.DNNForecast((DNN_WINDOW_SIZE -1), o4, forecast_x, W1, W2, W3, W4)
				DNNForecastBuf.append(forecasted)
				print("[Forecasted & input]", forecasted, forecast_x)

			break
		else:
			# usual one-step forecasting...
			forecast_x = x

			forecasted, forecast_x = node.DNNForecast((DNN_WINDOW_SIZE -1), o4, forecast_x, W1, W2, W3, W4)
			DNNForecastBuf.append(forecasted)

		obsNo +=1

	# end of Epoch iteration
	cnt += 1
	print("[basic DNN Epoch %d csv.KOSPI %d] " %(cnt, len(csv.KOSPI)))


"""
for k in range(52):
	#print("[MAIN : (target, x)]", target, x)
	target, x = node.DNNForecast((DNN_WINDOW_SIZE -1), target, x, W1, W2, W3, W4)
	DNNForecastBuf.append(target)

	print("\n[shifted x]\n", x)
"""


"""
print("[DNN START FORECASTING]\n", DNN_START_FORECASTING)
print("[DNN STOP FORECASTING]\n", DNN_STOP_FORECASTING)
"""
"""
print("[targetBuf length]\n", len(targetBuf))
print("[resultBuf length]\n", len(resultBuf))
print("[DNNForecastBuf length]\n", len(DNNForecastBuf))
#print(targetBuf)
"""

# graph
fig = plt.figure(figsize=(16,7))
fig.set_facecolor('white')

#plt.plot(DNNForecastBuf, color="green", linestyle='dotted')
plt.plot(DNNForecastBuf, color="green", linestyle='dotted', linewidth='0.9')
plt.plot(targetBuf, color="lightgrey", linewidth='0.5')

plt.axvline(DNN_START_FORECASTING, 0, 0.7, color='grey', linestyle=':', linewidth='1')
plt.axvline(DNN_STOP_FORECASTING, 0, 0.7, color='grey', linestyle=':', linewidth='1')

plt.annotate("[Forecast Interval] ", (DNN_START_FORECASTING*0.92, 3500))
#plt.annotate("[DNN Forecast STOP]", (DNN_STOP_FORECASTING*1.05, 3000))


"""
plt.plot(resultBuf, color="blue")
plt.plot(DNNResultBuf, color="green")
plt.plot(DNNForecastBuf, color="red")
"""

plt.ylabel("[Fitting]")
plt.xlabel("[Iteration]")
plt.title('ARMA based DNN vs. basic DNN')
#plt.legend(['KOSPI', 'ARMA(' + str(p) + ',' + str(q) +') based DNN', 'Basic DNN with learning window ' + str(DNN_p), 'DNN Forecast'])

plt.show()


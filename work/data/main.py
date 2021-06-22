import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import tensorflow as tf 
#from matplotlib.image import imread

import modules.Node as node

csv = pd.read_csv('./KOSPI.CSV')
#print(csv.KOSPI[0:3])

#print(x)

#print(csv.KOSPI, csv.ln_KOSPI)

#print("[RMSE]", node.getRMSE(csv.KOSPI, csv.ln_KOSPI))

#========================================================
# test variables...csv.KOSPI[0] = 100.89333
p = 5 
q = 3
l = m = n = p

lr = 0.5

"""
e = np.array([0.1,0.2])
target = np.array([5])

total_input = np.concatenate((x, e))
"""


j=0
# ML
W0, W1, W2, W3, W4 = node.initWeightMatrix(p, q, l, m, n)

for i in range(len(csv.KOSPI)-p-1):
	#print("target")
	print("*************************************************************************************************")
	j+=1
	target = np.array(csv.KOSPI[i:i+q])
	x = csv.KOSPI[i+1:i+p+1]
	#print("[MAIN target]\n", target)
	#print("[MAIN x]\n",x)

	o1, o2, o3, o4, e, total_input = node.partialFoward(x, target,  W0, W2, W3, W4)
	"""
	print("1.[MAIN e]", e)
	print("1.[MAIN total_input]", total_input)
	print("1.[MAIN o4]", o4)
	print("1.[MAIN target]", target)
	"""
	print("1.[MAIN total_input]", total_input)
	W1, W2, W3, W4 = node.fullBackward(x, e, total_input, target, W1, W2, W3, W4, o1, o2, o3, o4, lr)
	print("2.[MAIN total_input]", total_input)

	o1, o2, o3, o4, e, total_input = node.fullFoward(x, total_input, target, W1, W2, W3, W4)
	print("3.[MAIN total_input]", total_input)
	W1, W2, W3, W4 = node.fullBackward(x, e, total_input, target, W1, W2, W3, W4, o1, o2, o3, o4, lr)

	print("[RMSE]", np.sqrt((target[0] - o4[0])**2))

	if (j==3):
		break
"""
print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
print("[2.TEST in progress...]\n")
print(W1)
print(W1_new)
print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import tensorflow as tf 
#from matplotlib.image import imread

import modules.Node as node

csv = pd.read_csv('./KOSPI.CSV')
#print(csv)

x = csv.KOSPI
#print(x)

'''
for i in range(len(x)):
	print(x[i])
'''
#print(csv.KOSPI, csv.ln_KOSPI)

print(node.getRMSE(csv.KOSPI, csv.ln_KOSPI))

#========================================================
# test variables...
p = 3
q = 2
l = m = n = p
lr = 0.5

y = np.array([10, 20, 30])
e = np.array([0.1,0.2])

total_input = np.concatenate((y, e))


# ML
W0, W1, W2, W3, W4 = node.initWeightMatrix(p, q, l, m, n)

o1, o2, o3, o4 = node.partialFoward(y, W0, W2, W3, W4)

W1_new, W2_new, W3_new, W4_new = node.fullBackward(y, W1, W2, W3, W4, o1, o2, o3, o4, lr)
print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
print("[1.TEST in progress...]\n")
print(W1)
print(W1_new)
print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")

o1, o2, o3, o4 = node.partialFoward(y, W0, W2, W3, W4)

W1_new, W2_new, W3_new, W4_new = node.fullBackward(y, W1, W2, W3, W4, o1, o2, o3, o4, lr)
print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
print("[TEST in progress...]\n")
print(W1_new)
print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")

o1, o2, o3, o4 = node.fullFoward(total_input, W1_new, W2_new, W3_new, W4_new)

W1_new, W2_new, W3_new, W4_new = fullBackward(y, W1, W2, W3, W4, o1, o2, o3, o4, lr)











o1, o2, o3, o4 = node.fullFoward(total_input, W1_new, W2_new, W3_new, W4_new)

W1_new, W2_new, W3_new, W4_new = fullBackward(y, W1, W2, W3, W4, o1, o2, o3, o4, lr)











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
lr = 0.1

y = np.array([10, 20, 30])
e = np.array([0.1,0.2])

# ML
W0, W1, W2, W3, W4 = node.initWeightMatrix(p, q, l, m, n)

o1, o2, o3, o4 = node.partialFoward(y, W0, W2, W3, W4)

print("[TARGET VALUE]" , y)
print("[OUTPUT]" , o4)
print("[OUTPUT1]" , o1)
res = node.getresidual(y, o4)
print(res)




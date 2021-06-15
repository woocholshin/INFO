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



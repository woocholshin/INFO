import numpy as np
import pandas as pd
#import tensorflow as tf 
import matplotlib.pyplot as plt
#from matplotlib.image import imread

import modules.Node as node

csv = pd.read_csv('./KOSPI.CSV')

'''
csv = pd.read_csv('./KOSPI.CSV', 
					dtype={"ts_week":str, 
						   "KOSPI":float, 
					       "ln_KOSPI":float, 
					       "D_ln_KOSPI":float})
'''
print(csv)

#print(csv.ts_week)
#print(csv.KOSPI)

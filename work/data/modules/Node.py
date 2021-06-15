import numpy as np

if __name__ == "__main__":
	print("modules.Node saved & compiled")


# return RMSE
def getRMSE(predicted, target):
	temp = []

	for i in range(len(predicted)):
		temp.append((predicted[i] - target[i])**2)
	
	return np.sqrt(np.sum(temp).mean())


class Node:
	result = False
	
	def __init__(self):
		self.result = True
	
	def process(self):
		# do something
		return self.result

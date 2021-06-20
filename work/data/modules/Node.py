import numpy as np

# return partial
def copyPartialMatrix(w1, p, l):
	print(w1)
	w0 = np.zeros([l,p])
	print(w0)

	for i in range(l):
		for j in range(p):
			w0[j,i] = w1[j,i]
	
	return w0

#init weight matrix
def initWeightMatrix(p, q, l, m, n):

	w1 = np.random.rand(l, p+q)
	w0 = copyPartialMatrix(w1, p, l)

	w2 = np.random.rand(m, l)
	w3 = np.random.rand(n, m)
	w4 = np.random.rand(p, n)

	return w0, w1, w2, w3, w4

def fullFoward(i, w1, w2, w3, w4):
	o1 = np.dot(w1, i)
	o2 = np.dot(w2, o1)
	o3 = np.dot(w3, o2)
	o4 = np.dot(w4, o3)

	return o1, o2, o3, o4

# partial forward without error inpus
def partialFoward(i, w1, w2, w3, w4):
	#o1 = np.dot(w1, i)
	o1 = sigmoid(np.dot(w1, i))
	o2 = sigmoid(np.dot(w2, o1))
	o3 = sigmoid(np.dot(w3, o2))
	o4 = sigmoid(np.dot(w4, o3))

	return o1, o2, o3, o4

# return residual
def getresidual(target, out):
	return np.subtract(target, out)

# return reLU
def reLU(val):
	if val <= 0:
		return 0
	else:
		return val

# return sigmoid
def sigmoid(val):
	return 1/(1 + np.exp(-val))

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

if __name__ == "__main__":
	print("modules.Node saved & compiled")
	temp0, temp1, temp2, temp3, temp4 = initWeightMatrix(3,2,3,4,3)

	print(temp0)
	print(temp1)
	print(temp2)
	print(temp3)
	print(temp4)

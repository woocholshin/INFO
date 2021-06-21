import numpy as np

# return partial
def copyPartialMatrix(w1, p, l):
	#print(w1)
	w0 = np.zeros([l,p])
	#print(w0)

	for i in range(l):
		for j in range(p):
			w0[j,i] = w1[j,i]
	
	return w0

# prepare delta for weight adjustment
def prepareDelta(ifrom, ito, lr):
	#delta = np.zeros([len(ifrom), len(ito)])
	delta = np.zeros([len(ito), len(ifrom)])
	#print(delta)

	for i in range(len(ito)):
		for j in range(len(ifrom)):
			delta[i,j] = -2 * (ito[i] * ifrom[j]) * lr
	
	return delta

#init weight matrix
def initWeightMatrix(p, q, l, m, n):

	w1 = np.random.rand(l, p+q)
	w0 = copyPartialMatrix(w1, p, l)

	w2 = np.random.rand(m, l)
	w3 = np.random.rand(n, m)
	w4 = np.random.rand(p, n)

	return w0, w1, w2, w3, w4

def fullFoward(i, w1, w2, w3, w4):
	print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
	print(i)
	print("\n")
	print(w1)
	print("\n")
	print(w2)
	print("\n")
	print(w3)
	print("\n")
	print(w4)
	print("\n")
	print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
	o1 = reLU(np.dot(w1, i))
	o2 = reLU(np.dot(w2, o1))
	o3 = reLU(np.dot(w3, o2))
	o4 = reLU(np.dot(w4, o3))

	return o1, o2, o3, o4

# partial forward without error inputs
def partialFoward(i, w1, w2, w3, w4):
	#o1 = np.dot(w1, i)
	o1 = reLU(np.dot(w1, i))
	o2 = reLU(np.dot(w2, o1))
	o3 = reLU(np.dot(w3, o2))
	o4 = reLU(np.dot(w4, o3))

	return o1, o2, o3, o4

# return residual
def getresidual(target, out):
	return np.subtract(target, out)

# return reLU
def reLU(val):
	return np.maximum(0, val)

# return sigmoid
def sigmoid(val):
	return 1/(1 + np.exp(-val))

# return RMSE
def getRMSE(predicted, target):
	temp = []

	for i in range(len(predicted)):
		temp.append((predicted[i] - target[i])**2)
	
	return np.sqrt(np.sum(temp).mean())

# full backward
def fullBackward(y, W1, W2, W3, W4, o1, o2, o3, o4, lr):
	# target, calculated
	res = getresidual(y, o4)
	# from(or previous output), to(or next error), learning rate
	W4_new = W4 + prepareDelta(o3, res, lr)
	# new previous output based on new weght matrix
	o3_new = np.transpose(W4_new).dot(o4)

	res = getresidual(o3, o3_new)
	W3_new = prepareDelta(o2, res, lr)
	o2_new = np.transpose(W3_new).dot(o3)

	res = getresidual(o2, o2_new)
	W2_new = prepareDelta(o1, res, lr)
	o1_new = np.transpose(W2_new).dot(o2)

	res = getresidual(o1, o1_new)
	W1_new = prepareDelta(y, res, lr)

	return W1_new, W2_new, W3_new, W4_new



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

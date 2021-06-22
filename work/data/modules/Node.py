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
	#print(ifrom)

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
	w4 = np.random.rand(q, n)

	return w0, w1, w2, w3, w4

def fullFoward(x, total_input, target, w1, w2, w3, w4):
	o1 = reLU(np.dot(w1, total_input))
	o2 = reLU(np.dot(w2, o1))
	o3 = reLU(np.dot(w3, o2))
	o4 = reLU(np.dot(w4, o3))

	e = np.array(getRMSE(o4, target))
	print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
	print(o4)
	print("[fullFoward]", e)
	print(target)
	print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")

	return o1, o2, o3, o4, getRMSE(o4, target), np.concatenate((x,e))


# return RMSE
def getRMSE(predicted, target):
	"""
	print("*************************************************")
	print("[getRMSE target]")
	#print(predicted, target)
	"""
	temp = []

	for i in range(len(predicted)):
		temp.append(np.sqrt((predicted[i] - target[i])**2))
		#print("[getRMSE]", i, temp)
	
	return temp


# partial forward without error inputs
def partialFoward(x, target, w1, w2, w3, w4):
	#o1 = np.dot(w1, i)
	o1 = reLU(np.dot(w1, x))
	o2 = reLU(np.dot(w2, o1))
	o3 = reLU(np.dot(w3, o2))
	o4 = reLU(np.dot(w4, o3))

	print("[partialForward]")

	e = np.array(getRMSE(o4, target))
	"""
	print(o4)
	print(target)
	print(e)
	"""
	"""
	print(x)
	print(target)
	print(o4)
	print(e)
	print("[partialForward End]")
	"""

	return o1, o2, o3, o4, e, np.concatenate((x,e))


# return residual
def getresidual(target, out):
	return np.subtract(target, out)


# return reLU
def reLU(val):
	#return np.maximum(0, val)
	return 1/(1 + np.exp(-val))


# return sigmoid
def sigmoid(val):
	return 1/(1 + np.exp(-val))


# full backward
def fullBackward(x, e, y, target, W1, W2, W3, W4, o1, o2, o3, o4, lr):
	# target, calculated
	res = getresidual(target, o4)
	# from(or previous output), to(or next error), learning rate
	W4_new = W4 - prepareDelta(o3, res, lr)
	# new previous output based on new weght matrix
	o3_new = np.transpose(W4_new).dot(o4)

	res = getresidual(o3, o3_new)
	W3_new = W3 - prepareDelta(o2, res, lr)
	o2_new = np.transpose(W3_new).dot(o3)
	"""
	print(W3_new)
	print(o2_new)
	"""

	res = getresidual(o2, o2_new)
	W2_new = W2 - prepareDelta(o1, res, lr)
	o1_new = np.transpose(W2_new).dot(o2)

	res = getresidual(o1, o1_new)
	"""
	print("[x]", x)
	print("[e]", e)
	print("[y]",y)
	print("[res]",res)
	print("[W1]",W1)
	print(prepareDelta(y, res, lr))
	"""
	W1_new = W1 - prepareDelta(y, res, lr)

	return W1_new, W2_new, W3_new, W4_new



class Node:
	result = False
	
	def __init__(self):
		self.result = True
	
	def process(self):
		# do something
		return self.result

if __name__ == "__main__":
	print("-------------------------------")
	print("modules.Node saved & compiled")
	print("-------------------------------")

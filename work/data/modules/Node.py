import numpy as np
from scipy.special import expit, logit

# hyper-parameter for weight start
INIT_WEIGHT = 0 
MU = 0
SIGMA = 50
THRESHOLD = 0.01
#AMPLIFIER = 0.00000001 # sigmoid input algle parameter (ax)
AMPLIFIER = 0.00005 # tanh input algle parameter (ax)

# return partial
def copyPartialMatrix(w1, p, l):
	#print("[w1]", w1)
	w0 = np.zeros([l,p])

	for i in range(l):
		for j in range(p):
			w0[j,i] = w1[j,i]
	
	#print(w0)
	return w0


# prepare delta for weight adjustment
# (previous output, weight matrix, next residuals, next output, learning rate)
def prepareDelta(ifrom, prevW, ito, iout, lr, sigmoidFlag = True):
	delta = np.zeros([len(ito), len(ifrom)])
	"""
	prevErrors = np.dot(prevW.T, ito)

	tempSigma = np.dot(prevW, ifrom)
	sigmoidValue = sigmoid(tempSigma) * (1-sigmoid(tempSigma))
	test = lr * np.dot(ito, sigmoidValue) 
	"""

	#print("\n[prepareDelta]")
	"""
	print("[prev Weight]", prevW)
	print("[prev output]", ifrom)
	print("[prevErrors]", prevErrors)
	print("[tempSigma]", tempSigma)
	print("[sigmoidValue]", sigmoidValue)
	"""

    # sigmoid(default)
	if sigmoidFlag:
		# 'cause, sigmoid(sigmaWjkOj) == Ok  (in j-->k)
		# Wjknew = Wjkold - lr * (-2*Ek*Ok*(1-Ok)*Oj)
		for i in range(len(ito)):
			for j in range(len(ifrom)):
				#delta[i,j] = -1 * lr * ito[i] * iout[i]*(1-iout[i]) * ifrom[j] 
				# overflow may occur here...
				temp = -1 * lr * ito[i] * iout[i]*(1.0 - iout[i]) * ifrom[j]
				#print("[DELTA]", temp)

				# take only the direction from sigmoid directives...
				if temp < 0:
					temp = -lr
				else:
					temp = lr

				delta[i,j] = temp 

	# reLU
	else:
		for i in range(len(ito)):
			for j in range(len(ifrom)):
				delta[i,j] = -1 * (ito[i] * ifrom[j]) * lr
	
	#print("[W]", prevW)
	#print("[delta]", delta)
	return delta


# prepare delta for weight adjustment
# (previous output, weight matrix, next residuals, next output, learning rate)
def DNNprepareDelta(ifrom, prevW, ito, iout, lr, sigmoidFlag = True):
	delta = np.zeros([len(ito), len(ifrom)])

	"""
	print("============= DNNprepareDelta ================================")
	print("[from]\n", ifrom[0], end="\n\n")
	print("[W]\n", prevW, end="\n\n")
	print("[residual]\n", ito, end="\n\n")
	print("[next output]\n", iout, end="\n\n")
	"""

    # sigmoid(default)
	if sigmoidFlag:
		for i in range(len(ito)):
			for j in range(len(ifrom)):
				#temp = -1 * lr * ito[i] * iout[i]*(1.0 - iout[i]) * ifrom[j]             # sigmoid
				#temp = -1 * lr * ito[i] * (1.0 + iout[i])*(1.0 - iout[i]) * ifrom[j]     # tanh

				inv = inverseSoftplus(iout[i])
				temp = -2 * lr * ito[i] * (expit(inv) / (expit(inv)+1)) * ifrom[j]      # softplus

				# takes only the direction from sigmoid diff....
				"""
				if temp < 0:
					temp = -lr
				else:
					temp = lr

				delta[i,j] = temp 
				"""
				temp *= -lr
				
				delta[i,j] = temp 

	# reLU
	else:
		for i in range(len(ito)):
			for j in range(len(ifrom)):
				delta[i,j] = -1 * (ito[i] * ifrom[j]) * lr
	
	return delta


#init weight matrix
def initWeightMatrix(p, q, l, m, n):
	w1 = np.random.randn(l, p+q) + INIT_WEIGHT + np.random.normal(INIT_WEIGHT, SIGMA)
	w0 = copyPartialMatrix(w1, p, l)

	w2 = np.random.randn(m, l) + INIT_WEIGHT + np.random.normal(INIT_WEIGHT, SIGMA)
	w3 = np.random.randn(n, m) + INIT_WEIGHT + np.random.normal(INIT_WEIGHT, SIGMA)
	w4 = np.random.randn(q, n) + INIT_WEIGHT + np.random.normal(INIT_WEIGHT, SIGMA)

	return w0, w1, w2, w3, w4


#init weight matrix
def DNNinitWeightMatrix(p, l, m, n):
	w1 = np.round(np.random.randn(l, p), 1) + INIT_WEIGHT + np.round(np.random.normal(INIT_WEIGHT, SIGMA), 1)
	w2 = np.round(np.random.randn(m, l), 1) + INIT_WEIGHT + np.round(np.random.normal(INIT_WEIGHT, SIGMA), 1)
	w3 = np.round(np.random.randn(n, m), 1) + INIT_WEIGHT + np.round(np.random.normal(INIT_WEIGHT, SIGMA), 1)

	# single node for final output layer
	w4 = np.round(np.random.randn(1, n), 1) + INIT_WEIGHT + np.round(np.random.normal(INIT_WEIGHT, SIGMA), 1)

	print("-------------------------")
	print("[init W1]\n",w1)
	print("[init W2]\n",w2)
	print("[init W3]\n",w3)
	print("[init W4]\n",w4)
	print("-------------------------")

	return w1, w2, w3, w4


# general DNN Forward propagation...one target variable(Xt), multiple input(Xt-1, Xt-2...)
def DNNForward(x, target, w1, w2, w3, w4):
	"""
	o1 = activate(np.dot(w1, x))
	o2 = activate(np.dot(w2, o1))
	o3 = activate(np.dot(w3, o2))
	o4 = np.round(np.dot(w4, o3), 3)
	"""

	"""
	o1 = activateTanh(np.dot(w1, x), False)
	o2 = activateTanh(np.dot(w2, o1), False)
	o3 = activateTanh(np.dot(w3, o2), False)
	o4 = np.round(np.dot(w4, o3), 3)
	"""

	o1 = activateSoftplus(np.dot(w1, x), False)
	o2 = activateSoftplus(np.dot(w2, o1), False)
	o3 = activateSoftplus(np.dot(w3, o2), False)
	o4 = np.round(np.dot(w4, o3), 3)

	return o1, o2, o3, o4

# forecast...
def DNNForecast(size, x, W1, W2, W3, W4):
	"""
	print("\n[size]", size)
	print("\n[output]", output)
	print("\n[x]", x)
	print("\n[shifted buf]\n", tempBuf)
	print("\n[x]", x)
	print("\n[W1]\n", W1)
	"""

	tempBuf = []

	# forecasting
	"""
	o1 = activate(np.dot(W1, x))
	o2 = activate(np.dot(W2, o1))
	o3 = activate(np.dot(W3, o2))
	o4 = np.round(np.dot(W4, o3), 3)
	"""
	
	"""
	o1 = activateTanh(np.dot(W1, x), False)
	o2 = activateTanh(np.dot(W2, o1), False)
	o3 = activateTanh(np.dot(W3, o2), False)
	o4 = np.round(np.dot(W4, o3), 3)
	"""

	o1 = activateSoftplus(np.dot(W1, x), False)
	o2 = activateSoftplus(np.dot(W2, o1), False)
	o3 = activateSoftplus(np.dot(W3, o2), False)
	o4 = np.round(np.dot(W4, o3), 3)

	o4 = o4.item(0)

	"""
	print("o1 : ", o1)
	print("o2 : ", o2)
	print("o3 : ", o3)
	print("o4 : ", o4)
	"""

	# shifting...
	tempBuf.append(o4)
	for i in range(size-1):
		tempBuf.append(x[i])

	"""
	# update result
	finalBuf = []
	finalBuf.append(o4)

	for j in range(size-1):
		finalBuf.append(tempBuf[j])

	"""
	#print("[tempBuf]", tempBuf)
	#print("[o4]", o4)
	return o4, tempBuf


# DNN backward
def DNNfullBackward(x, target, W1, W2, W3, W4, o1, o2, o3, o4, lr):
	res = getResidual(target, o4)
	W4_new = W4 + DNNprepareDelta(o3, W4, res, o4, lr)
	o3_new = np.transpose(W4_new).dot(o4)

	res = getResidual(o3, o3_new)
	W3_new = W3 + DNNprepareDelta(o2, W3, res, o3, lr)
	o2_new = np.transpose(W3_new).dot(o3)

	res = getResidual(o2, o2_new)
	W2_new = W2 + DNNprepareDelta(o1, W2, res, o2, lr)
	o1_new = np.transpose(W2_new).dot(o2)

	res = getResidual(o1, o1_new)
	W1_new = W1 + DNNprepareDelta(x, W1, res, o1, lr)

	return W1_new, W2_new, W3_new, W4_new


# forward for ARMA(p, q)
def fullForward(x, total_input, target, w1, w2, w3, w4):
	o1 = activate(np.dot(w1, total_input))
	o2 = activate(np.dot(w2, o1))
	o3 = activate(np.dot(w3, o2))
	#o4 = activate(np.dot(w4, o3))
	o4 = np.dot(w4, o3)

	e = np.array(getDiff(o4, target))
	"""
	print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
	print("[fullForward e]", e)
	print("o4",o4)
	print("target", target)
	print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
	"""

	return o1, o2, o3, o4, getDiff(o4, target), np.concatenate((x,e))


# return difference
def getDiff(predicted, target):
	temp = []

	for i in range(len(predicted)):
		temp.append(predicted[i] - target[i])
	
	return temp


# partial foward without error inputs
def partialFoward(x, target, w1, w2, w3, w4):
	#o1 = np.dot(w1, i)
	o1 = activate(np.dot(w1, x))
	o2 = activate(np.dot(w2, o1))
	o3 = activate(np.dot(w3, o2))
	#o4 = activate(np.dot(w4, o3))
	o4 = np.dot(w4, o3)

	e = np.array(getDiff(o4, target))
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
	print("--------------------------------------------------------")
	print("[partialFoward e]", e)
	print("x",x)
	print("o4",o4)
	print("target", target)
	print("--------------------------------------------------------")
	"""

	return o1, o2, o3, o4, e, np.concatenate((x,e))


# return residual
def getResidual(target, out):
	return np.subtract(target, out)

# reLU or sigmoid(default)
def activate(val, sigmoidFlag = True):
	if sigmoidFlag:
		return np.round(1/(1 + expit(-val * AMPLIFIER)), 4)
	else:
		return np.maximum(0, val)

# tanh activate
def activateTanh(val, diff=False):
	if diff:  # return diff.
		#return np.round((1 + np.tanh(val*AMPLIFIER))*(1 - np.tanh(val*AMPLIFIER)), 4)
		return (1 + np.tanh(val*AMPLIFIER))*(1 - np.tanh(val*AMPLIFIER))
	else:
		#return (np.exp(val) - np.exp(-val))/(np.exp(val) + np.exp(-val))
		#return np.round(np.tanh(val*AMPLIFIER), 4) 
		return np.tanh(val*AMPLIFIER) 


# inverse of softplus function
def inverseSoftplus(val):
	temp = np.maximum((expit(val) - 1), THRESHOLD)
	
	return np.log(temp)


# softplus - reLU like but smoother
def activateSoftplus(val, diff=False):
	if diff:
		return sigmoid(val)
	else:
		return np.log(expit(val) + 1)
	

# sigmoid
def sigmoid(val):
	return 1/(1 + expit(-val))


# full backward
def fullBackward(x, e, y, target, W1, W2, W3, W4, o1, o2, o3, o4, lr):
	# target, calculated
	#print("==================================================")
	res = getResidual(target, o4)
	# from(or previous output), weight, to(or next error), out(next), learning rate
	W4_new = W4 + prepareDelta(o3, W4, res, o4, lr)
	# new previous output based on new weght matrix
	o3_new = np.transpose(W4_new).dot(o4)
	#print("\n[fullBackward W4_new]", W4_new)

	res = getResidual(o3, o3_new)
	W3_new = W3 + prepareDelta(o2, W3, res, o3, lr)
	o2_new = np.transpose(W3_new).dot(o3)
	#print("\n[fullBackward W3_new]", W3_new)
	"""
	print(W3_new)
	print(o2_new)
	"""

	res = getResidual(o2, o2_new)
	W2_new = W2 + prepareDelta(o1, W2, res, o2, lr)
	o1_new = np.transpose(W2_new).dot(o2)
	#print("\n[fullBackward W2_new]", W2_new)

	res = getResidual(o1, o1_new)
	"""
	print("[x]", x)
	print("[e]", e)
	print("[y]",y)
	print("[res]",res)
	print("[W1]",W1)
	print(prepareDelta(y, res, lr))
	"""
	W1_new = W1 + prepareDelta(y, W1, res, o1, lr)
	#print("\n[fullBackward W1_new]", W1_new)

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

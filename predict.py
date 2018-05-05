import numpy as np


#### Auxiliary_functions

################ Returns indexes of maximum value of the array

def nanargmax(a):
	idx = np.argmax(a, axis=None)
	multi_idx = np.unravel_index(idx, a.shape)
	if np.isnan(a[multi_idx]):
		nan_count = np.sum(np.isnan(a))
		idx = np.argpartition(a, -nan_count-1, axis=None)[-nan_count-1]
		multi_idx = np.unravel_index(idx, a.shape)
	return multi_idx

################ forwards propogates through pool layer

def maxpool(X, f, s):
	(l, w, w) = X.shape
	pool = np.zeros((l, int((w-f)/s+1),int((w-f)/s+1)))
	for jj in range(0,l):
		i=0
		while(i<w):
			j=0
			while(j<w):
				pool[jj,int(i/2),int(j/2)] = np.max(X[jj,i:i+f,j:j+f])
				j+=s
			i+=s
	return pool

################ computes the cost at the output layer

def softmax_cost(out,y):
	eout = np.exp(out, dtype=np.float)
	probs = eout/sum(eout)
	
	p = sum(y*probs)
	cost = -np.log(p)	
	return cost,probs	

################# predict 

def predict(image, filt1, filt2, bias1, bias2, theta3, bias3):
	
	## l - channel
	## w - size of square image
	## l1 - No. of filters in Conv1
	## l2 - No. of filters in Conv2
	## w1 - size of image after conv1
	## w2 - size of image after conv2

	(l,w,w)=image.shape

	
	(l1,f,f) = filt2[0].shape
	l2 = len(filt2)
	w1 = w-f+1
	w2 = w1-f+1
	conv1 = np.zeros((l1,w1,w1))
	conv2 = np.zeros((l2,w2,w2))
	for jj in range(0,l1):
		for x in range(0,w1):
			for y in range(0,w1):
				conv1[jj,x,y] = np.sum(image[:,x:x+f,y:y+f]*filt1[jj])+bias1[jj]
	
		
	conv1[conv1<=0] = 0 #relu activation
	## Calculating second Convolution layer
	for jj in range(0,l2):
		for x in range(0,w2):
			for y in range(0,w2):
				conv2[jj,x,y] = np.sum(conv1[:,x:x+f,y:y+f]*filt2[jj])+bias2[jj]
	
	

	conv2[conv2<=0] = 0 # relu activation
	
	pooled_layer = maxpool(conv2, 2, 2)	
	fc1 = pooled_layer.reshape((int((w2/2)*(w2/2)*l2),1))
	out = theta3.dot(fc1) + bias3	#10*1
	eout = np.exp(out, dtype=np.float)
	probs = eout/sum(eout)
	
	return np.argmax(probs), np.max(probs)

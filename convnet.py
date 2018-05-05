import numpy as np
import cv2
####### Auxiliary Functions for convnet

##### Returns indexes of maximum value of the array

def nanargmax(a):
	idx = np.argmax(a, axis=None)
	multi_idx = np.unravel_index(idx, a.shape)
	if np.isnan(a[multi_idx]):
		nan_count = np.sum(np.isnan(a))
		idx = np.argpartition(a, -nan_count-1, axis=None)[-nan_count-1]
		multi_idx = np.unravel_index(idx, a.shape)
	return multi_idx

##### forwards propogates through pool layer

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

##### computes the entropy based cost at the output layer


def softmax_cost(out,y):
	eout = np.exp(out, dtype=np.float)
	probs = eout/sum(eout)
	
	p = sum(y*probs)
	cost = -np.log(p)	
	return cost,probs	


####### Returns gradient for all the paramaters in each iteration

def  ConvNet(args):
	#####################################################################################################################
	#######################################  Feed forward to get all the layers  ########################################
	#####################################################################################################################

	## Calculating first Convolution layer
		
	## l - channel
	## w - size of square image
	## l1 - No. of filters in Conv1
	## l2 - No. of filters in Conv2
	## w1 - size of image after conv1
	## w2 - size of image after conv2

	image = args[0] 
	label=args[1]  
	filt1=args[2] 
	filt2=args[3]
	
	bias1 = args[4]
	bias2 = args[5]
	theta3= args[6]
	bias3 =	args[7]
	(l, w, w) = image.shape
	
	l1 = len(filt1)
	l2 = len(filt2)
	
	( _, f, f) = filt1[0].shape
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
	
	
	## Pooled layer with 2*2 size and stride 2,2
	pooled_layer = maxpool(conv2, 2, 2)	
	
	fc1 = pooled_layer.reshape((int((w2/2)*(w2/2)*l2),1))
	
	out = theta3.dot(fc1) + bias3	#10*1
	
	######################################################################################################################
	########################################  Using softmax function to get cost  ########################################
	######################################################################################################################

	cost, probs = softmax_cost(out, label)
	if np.argmax(out)==np.argmax(label):
		acc=1
	else:
		acc=0

	#######################################################################################################################
	##########################  Backpropagation to get gradient	using chain rule of differentiation  ######################
	#######################################################################################################################
	dout = probs - label			##	dL/dout
	
	dtheta3 = dout.dot(fc1.T) 		##	dL/dtheta3
	
	
	dbias3 = dout 				##	dbias3	
	

	dfc1 = theta3.T.dot(dout)		##	dL/dfc1

	dpool = dfc1.T.reshape((l2, int(w2/2), int(w2/2)))

	dconv2 = np.zeros((l2, w2, w2))
	
	for jj in range(0,l2):
		i=0
		while(i<w2):
			j=0
			while(j<w2):
				(a,b) = nanargmax(conv2[jj,i:i+2,j:j+2]) ## Getting indexes of maximum value in the array
				dconv2[jj,i+a,j+b] = dpool[jj,int(i/2),int(j/2)]
				j+=2
			i+=2
	
	dconv2[conv2<=0]=0

	dconv1 = np.zeros((l1, w1, w1))
	dfilt2 = {}
	dbias2 = {}
	for xx in range(0,l2):
		dfilt2[xx] = np.zeros((l1,f,f))
		dbias2[xx] = 0

	dfilt1 = {}
	dbias1 = {}
	for xx in range(0,l1):
		dfilt1[xx] = np.zeros((l,f,f))
		dbias1[xx] = 0

	for jj in range(0,l2):
		for x in range(0,w2):
			for y in range(0,w2):
				dfilt2[jj]+=dconv2[jj,x,y]*conv1[:,x:x+f,y:y+f]
				dconv1[:,x:x+f,y:y+f]+=dconv2[jj,x,y]*filt2[jj]
		dbias2[jj] = np.sum(dconv2[jj])
	dconv1[conv1<=0]=0
	for jj in range(0,l1):
		for x in range(0,w1):
			for y in range(0,w1):
				dfilt1[jj]+=dconv1[jj,x,y]*image[:,x:x+f,y:y+f]

		dbias1[jj] = np.sum(dconv1[jj])

	
	return [dfilt1, dfilt2, dbias1, dbias2, dtheta3, dbias3, cost, acc]





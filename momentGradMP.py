import numpy as np
import multiprocessing as mp
from convnet import ConvNet
import time

def momentGradMP(batch, LEARNING_RATE, w, l, MU, filt1, filt2, bias1, bias2, theta3, bias3, cost, acc):
	#Momentum Gradient Update
	# MU=0.5
	batch_size = len(batch)	
	n_cores = 4
	X = batch[:,0:-1]
	X = X.reshape(int(batch_size/n_cores) ,n_cores, l, w, w)
	y = batch[:,-1]
	y = y.reshape(int(batch_size/n_cores), n_cores)

        


        ###### computing per batch update
	
	pool = mp.Pool(processes=n_cores)

	labels = np.zeros((int(batch_size/n_cores),n_cores,theta3.shape[0],1))	

	results = []
	
	
	for i in range(0,int(batch_size/n_cores)): 
		for j in range(0,n_cores):
			label = np.zeros((theta3.shape[0],1))
			label[int(y[i][j]),0] = 1
			labels[i][j] = label

	for i in range(0,int(batch_size/n_cores)):		
		results_ = pool.map(ConvNet, [[X[i][j], labels[i][j], filt1, filt2, bias1, bias2, theta3, bias3] for j in range(0,n_cores)]) 
		results.append(results_)	
	pool.close()	
	
	
	return results

	

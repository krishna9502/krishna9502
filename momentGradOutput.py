import numpy as np
import time


### updates the weights after the each batch processing 

def momentGradOutput(results,filt1, filt2, bias1, bias2, theta3, bias3, cost, acc,MU,LEARNING_RATE):
	
	
	##### initializing all differentials in filters for the batch
	n_correct=0
	cost_ = 0
	
	dfilt2 = {}
	dfilt1 = {}
	dbias2 = {}
	dbias1 = {}
	v1 = {}
	v2 = {}
	bv1 = {}
	bv2 = {}
	
	
	for k in range(0,len(filt2)):
		dfilt2[k] = np.zeros(filt2[0].shape)
		dbias2[k] = 0
		v2[k] = np.zeros(filt2[0].shape)
		bv2[k] = 0

	for k in range(0,len(filt1)):
		dfilt1[k] = np.zeros(filt1[0].shape)
		dbias1[k] = 0
		v1[k] = np.zeros(filt1[0].shape)
		bv1[k] = 0


	dtheta3 = np.zeros(theta3.shape)
	dbias3 = np.zeros(bias3.shape)
	v3 = np.zeros(theta3.shape)
	bv3 = np.zeros(bias3.shape)

		
	
	batch_size = len(results)
	n_cores = 4
	for i in range(0,int(batch_size/n_cores)):
		for j in range(0,n_cores):
			for k in range(0,len(filt2)):
				dfilt2[k]+=results[i][j][1][k]
				dbias2[k]+=results[i][j][3][k]
			for k in range(0,len(filt1)):
				dfilt1[k]+=results[i][j][0][k]
				dbias1[k]+=results[i][j][2][k]
		
			dtheta3+=results[i][j][4]
			dbias3+=results[i][j][5]

			cost_+=results[i][j][6]
			n_correct+=results[i][j][7]

        #### updating per batch
		
	for j in range(0,len(filt1)):
		v1[j] = MU*v1[j] -LEARNING_RATE*dfilt1[j]/batch_size
		filt1[j] += v1[j]
		# filt1[j] -= LEARNING_RATE*dfilt1[j]/batch_size
		bv1[j] = MU*bv1[j] -LEARNING_RATE*dbias1[j]/batch_size
		bias1[j] += bv1[j]
	for j in range(0,len(filt2)):
		v2[j] = MU*v2[j] -LEARNING_RATE*dfilt2[j]/batch_size
		filt2[j] += v2[j]
		# filt2[j] += -LEARNING_RATE*dfilt2[j]/batch_size
		bv2[j] = MU*bv2[j] -LEARNING_RATE*dbias2[j]/batch_size
		bias2[j] += bv2[j]
	v3 = MU*v3 - LEARNING_RATE*dtheta3/batch_size
	theta3 += v3
	# theta3 += -LEARNING_RATE*dtheta3/batch_size
	bv3 = MU*bv3 -LEARNING_RATE*dbias3/batch_size
	bias3 += bv3

	cost_ = cost_/batch_size
	cost.append(cost_)
	accuracy = float(n_correct)/batch_size
	acc.append(accuracy)
	

	return [filt1, filt2, bias1, bias2, theta3, bias3, cost, acc]




import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle
import time
import random

from remtime import *
from extract import *
from initialize import *
from predict import *


from momentGradMP import *
from momentGradOutput import *

PICKLE_FILE = 'output.pickle'



#####################################################################################################################################
################################################## ------ START HERE --------  ######################################################
##################################### ---------- CONVOLUTIONAL NEURAL NETWORK ---------------  ######################################
################ ----ARCHITECTURE PROPOSED : [INPUT - CONV1 - RELU - CONV2 - RELU- MAXPOOL - FC1 - OUT]---- #########################
#####################################################################################################################################


#######  Data extracting

m =10000
X = extract_data('t10k-images-idx3-ubyte.gz', m, IMG_WIDTH)
y_dash = extract_labels('t10k-labels-idx1-ubyte.gz', m).reshape(m,1)
X-= int(np.mean(X))
X/= int(np.std(X))
test_data = np.hstack((X,y_dash))


m =50000
X = extract_data('train-images-idx3-ubyte.gz', m, IMG_WIDTH)
y_dash = extract_labels('train-labels-idx1-ubyte.gz', m).reshape(m,1)
print (np.mean(X), np.std(X))
X-= int(np.mean(X))
X/= int(np.std(X))
train_data = np.hstack((X,y_dash))

np.random.shuffle(train_data)



NUM_IMAGES = train_data.shape[0]


print("Learning Rate:"+str(LEARNING_RATE)+", Batch Size:"+str(BATCH_SIZE))

###### Training starts here

for epoch in range(0,NUM_EPOCHS):
	np.random.shuffle(train_data)
	batches = [train_data[k:k + BATCH_SIZE] for k in range(0, NUM_IMAGES, BATCH_SIZE)]
	x=0
	for batch in batches:
		stime = time.time()

		results = momentGradMP(batch, LEARNING_RATE, IMG_WIDTH, IMG_DEPTH, MU, filt1, filt2, bias1, bias2, theta3, bias3, cost, acc)

		out = momentGradOutput(results,filt1, filt2, bias1, bias2, theta3, bias3, cost, acc,MU,LEARNING_RATE)

		[filt1, filt2, bias1, bias2, theta3, bias3, cost, acc] = out

		epoch_acc = round(np.sum(acc[int(epoch*NUM_IMAGES/BATCH_SIZE):])/(x+1),2)
		
		per = float(x+1)/len(batches)*100
		print("Epoch:"+str(round(per,2))+"% Of "+str(epoch+1)+"/"+str(NUM_EPOCHS)+", Cost:"+str(cost[-1])+", B.Acc:"+str(acc[-1]*100)+", E.Acc:"+str(epoch_acc))
		
		ftime = time.time()
		deltime = ftime-stime
		
		remtime = (len(batches)-x-1)*deltime+deltime*len(batches)*(NUM_EPOCHS-epoch-1)
		printTime(remtime)
		x+=1


################################################################################# saving the trained model parameters

with open(PICKLE_FILE, 'wb') as file:
	pickle.dump(out, file)

################################################################################# Opening the saved model parameter

pickle_in = open(PICKLE_FILE, 'rb')
out = pickle.load(pickle_in)

[filt1, filt2, bias1, bias2, theta3, bias3, cost, acc] = out

################################################################################# Computing Test accuracy


X = test_data[:,0:-1]
X = X.reshape(len(test_data), IMG_DEPTH, IMG_WIDTH, IMG_WIDTH)
y = test_data[:,-1]
corr = 0
print("Computing accuracy over test set:")
for i in range(0,len(test_data)):
	image = X[i]
	digit, prob = predict(image, filt1, filt2, bias1, bias2, theta3, bias3)
	print (digit, y[i])
	if digit==y[i]:
		corr+=1
	if (i+1)%int(0.01*len(test_data))==0:
		print(str(float(i+1)/len(test_data)*100)+"% Completed")
test_acc = float(corr)/len(test_data)*100
print("Test Set Accuracy:"+str(test_acc))



from PIL import Image
import numpy as np
import os
import glob
import re
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

def read_img(location):
	path = os.path.abspath('.AE_simpson.py') #absolute path of program
	path = re.sub('[a-zA-Z\s._]+$', '', path) #remove unintended file
	x_train = [] #set array of training data
	y_train = [] #set array of training label
	x_test = [] #set array of testing data
	y_test = [] #set array of testing label
	label_name = [] #name of label in folder will be stored
	dirs = os.listdir(path+'images/') #list of all directory in images folder
	label = 0 #label is used as y data for each x. label will be incremented until num_label/class
	for i in dirs: #loop all directory
		n = 0 
		count = 0 #counting data in folder
		label_name.append(i) #save folder name in var label_name
		for pic in glob.glob(path+'images/'+i+'/*.jpg'): #loop all data
			im = Image.open(pic) #open data
			im = np.array(im) #store im as numpy array
			if(im.shape[0]==180 and im.shape[1]==120): #take only image with shape 120 x 180
				#print(im.shape)
				r = im[:,:,0]
				g = im[:,:,1]
				b = im[:,:,2]
				if(n<5): # 5 data in beginning set as test data
					x_test.append([r,g,b]) #save in x_test
					y_test.append([label]) #save in y_test
				else: #remaining data set as training data
					x_train.append([r,g,b]) #save in x_train
					y_train.append([label]) #save in y_train
				n = n + 1 #increment n
				count = count + 1 #increment count
		#print(count)
		label = label + 1 #increment label
	#print(label_name)
	return np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)

img_rows = 120 #num of image height
img_cols = 180 #num of image width
num_class = 20 #num of classes/labels
x_train,y_train,x_test,y_test = read_img('images/') #call read_img

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3) #reshape x_train into: (num of data, 120,180,20)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3) #reshape x_test into (num of data, 120, 180,3)

input_shape = (img_rows, img_cols, 3) #input size: 120x180x3

x_train = x_train.astype('float32') #normalization of x_train
x_test = x_test.astype('float32') #normalization of x_test
x_train /= 255 #make x_train between  0 - 1
x_test /= 255 #make x_test between 0 - 1
y_train = keras.utils.to_categorical(y_train, num_class) #change y_train into categorical like [0,1,0...,0]
y_test = keras.utils.to_categorical(y_test, num_class) # change y_test into categorical
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_test[0:10])
print(y_train)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_class, activation='softmax'))
model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=10, nb_epoch=10, verbose=1, validation_data=(x_test, y_test))
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
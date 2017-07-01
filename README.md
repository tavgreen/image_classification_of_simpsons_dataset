# Simpson Image Dataset Classification using Convolutional Neural Network (CNN) in Keras and Tensorflow #

## Description ##
This project is about how to classify image character of Simpson dataset using one of most sophisticated deep learning algorithm: Convolutional Neural Network (CNN) Model in Keras and Tensorflow library. Dataset resource can be found in [Kaggle Simpsons Character](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset). This project is developed by using [Python3.6](https://www.python.org/downloads/release/python-360/), [Tensorflow](http://tensorflow.org) or [Keras](http://keras.io). Based on dataset, there are 20 classes which consists of 400 - 2000 images per class. Each image dataset has different width and height size. In this project, i only use image that has height 120 pixels and width 180 pixels (uniform).     

## Methods ##
has not completed yet

## Program ##
- Import Library
```python
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
```

- Read Image
```python
path = os.path.abspath('.AE_simpson.py') #absolute path of program
path = re.sub('[a-zA-Z\s._]+$', '', path) #remove unintended file
dirs = os.listdir(path+'images/') #list of all directory in images folder
for i in dirs: #loop all directory
	label_name.append(i) #save folder name in var label_name
	for pic in glob.glob(path+'images/'+i+'/*.jpg'): #loop all data
		im = Image.open(pic) #open data
		im = np.array(im) #store im as numpy array
		if(im.shape[0]==180 and im.shape[1]==120): #take only image with shape 120 x 180
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
	label = label + 1 #increment label
```

- Data Normalization
```python
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
```

- Model Definition (Keras)
```python
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
```

- Running Model
```python
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=10, nb_epoch=10, verbose=1, validation_data=(x_test, y_test))
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
```
## Result ##
- 10 epochs: 63 %

## Future Works ##
This projects can be extended by implementing recent convolutional neural network algorithm like:
- [Alex Krizhevsky et al](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). (2012). ImageNet Classification with Deep Convolutional Neural Networks. 
- [Simonyan K et al](https://arxiv.org/pdf/1409.1556v6.pdf). (2015). VGG Net.
- [He et al](https://arxiv.org/pdf/1512.03385v1.pdf). (2015). Deep Residual Learning for Image Recognition

## References ##
- Simpson Image Character Dataset. [Kaggle Image Character Dataset](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset).
- Keras Blogg. [@fchollet](https://github.com/fchollet/keras/tree/master/examples).
- Keras Documentation. [Keras IO](http://keras.io).
- Tensor Flow Documentation. [Tensorflow](http://tensorflow.org).
- Adnan Ardhian. [@adnanardhian](https://github.com/adnanardhian).

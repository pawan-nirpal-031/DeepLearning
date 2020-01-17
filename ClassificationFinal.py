#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2



datadir = "C:/Users/jayspc/Documents/MachineLearning/DataSets/classification/train"
categories = ['alai_darwaza','alai_minar','iron_pillar','jamali_kamali_tomb','qutub_minar']

for cat in categories:
    path = os.path.join(datadir,cat)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap='gray')
        plt.show()
        #break
    #break


# In[5]:


IMG_SIZE = 200
new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
plt.imshow(new_array,cmap='gray')
plt.show()


# In[7]:


train_data = []
def create_train_data():
    for cat in categories:
        path = os.path.join(datadir,cat)
        class_num = categories.index(cat)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                train_data.append([new_array,class_num])
            except Exception as e:
                pass
    
create_train_data()


# In[8]:


print(len(train_data))


# In[21]:


import random as rd
rd.shuffle(train_data)

x = []# feature set
y = []# label set

for features,label in train_data:
    x.append(features)
    y.append(label)
   

x = np.array(x).reshape(-1,IMG_SIZE,IMG_SIZE,1)# we cannot pass a list into neural network it has to be a numpy array hence conversion
#1 is because it is a grayscale image in convnet try 3 for color image


# In[27]:


import pickle as pk #export or save the training 
'''
The pickle module is used for implementing binary protocols for serializing and de-serializing a Python object structure.

Pickling: It is a process where a Python object hierarchy is converted into a byte stream.
Unpickling: It is the inverse of Pickling process where a byte stream is converted into an object hierarchy.
Module Interface :

dumps() – This function is called to serialize an object hierarchy.
loads() – This function is called to de-serialize a data stream.
For more control over serialization and de-serialization, Pickler or an Unpickler objects are created respectively.

'''
pickle_out = open('x.pickle','wb')
pk.dump(x,pickle_out)
pickle_out.close()


pickle_out = open('y.pickle','wb')
pk.dump(y,pickle_out)
pickle_out.close()

pickle_in = open("x.pickle","rb")
x = pk.load(pickle_in)

'''
Convolutional layers are the major building blocks used in convolutional neural networks.

A convolution is the simple application of a filter to an input that results in an activation. Repeated application of the same filter to an input results in a map of activations called a feature map, indicating the locations and strength of a detected feature in an input, such as an image.

The innovation of convolutional neural networks is the ability to automatically learn a large number of filters in parallel specific to a training dataset under the constraints of a specific predictive modeling problem, such as image classification. The result is highly specific features that can be detected anywhere on input images.

In this tutorial, you will discover how convolutions work in the convolutional neural network.

After completing this tutorial, you will know:

Convolutional neural networks apply a filter to an input to create a feature map that summarizes the presence of detected features in the input.
Filters can be handcrafted, such as line detectors, but the innovation of convolutional neural networks is to learn the filters during training in the context of a specific prediction problem.
How to calculate the feature map for one- and two-dimensional convolutional layers in a convolutional neural network.

Strides
Stride is the number of pixels shifts over the input matrix. When the stride is 1 then we move the filters to 1 pixel at a time. When the stride is 2 then we move the filters to 2 pixels at a time and so on. The below figure shows convolution would work with a stride of 2.


Padding
Sometimes filter does not fit perfectly fit the input image. We have two options:
Pad the picture with zeros (zero-padding) so that it fits
Drop the part of the image where the filter did not fit. This is called valid padding which keeps only valid part of the image.
Non Linearity (ReLU)
ReLU stands for Rectified Linear Unit for a non-linear operation. The output is ƒ(x) = max(0,x).
Why ReLU is important : ReLU’s purpose is to introduce non-linearity in our ConvNet. Since, the real world data would want our
ConvNet to learn would be non-negative linear values.

Pooling Layer
Pooling layers section would reduce the number of parameters when the images are too large. Spatial pooling also called subsampling or downsampling which reduces the dimensionality of each map but retains important information. Spatial pooling can be of different types:
Max Pooling
Average Pooling
Sum Pooling
Max pooling takes the largest element from the rectified feature map. Taking the largest element could also take the average pooling. Sum of all elements in the feature map call as sum pooling.

Provide input image into convolution layer
Choose parameters, apply filters with strides, padding if requires. Perform convolution on the image and apply ReLU activation to the matrix.
Perform pooling to reduce dimensionality size
Add as many convolutional layers until satisfied
Flatten the output and feed into a fully connected layer (FC Layer)
Output the class using an activation function (Logistic Regression with cost functions) and classifies images.
In the next post, I would like to talk about some popular CNN architectures such as AlexNet, VGGNet, GoogLeNet, and ResNet.

'''


# In[91]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
import pickle as pk
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
import time
import datetime
name = "Monument classification {} ".format(int(time.time()))






tboard_log_dir = os.path.join("logs",name)
tnsrb = TensorBoard(log_dir=f".\logs\MODEL", histogram_freq=1,
                                  write_grads=True)
#tnsrb = TensorBoard(log_dir = r"logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") )
#log_dir=r"logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
x = pk.load(open('x.pickle','rb'))
y = pk.load(open('y.pickle','rb'))







#from tf.keras.backend import K
#K.clear_session()


from tensorflow.keras import backend as K
K.clear_session()

x = x/255.0


model = Sequential()
model.add(  Conv2D(64, (3,3),  input_shape = x.shape[1:] ) )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(   Conv2D( 64, (3,3),input_shape=x.shape[1:]  )    )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

#now we have  2*64 convolutiol neural network

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(5, activation='softmax'))
model.add(Activation("sigmoid"))
model.compile(loss = "sparse_categorical_crossentropy",optimizer = "adam", metrics = ['accuracy'])


batch_size = 70
def generator(features, labels, batch_size):
 # Create empty arrays to contain batch of features and labels#
 batch_features = np.zeros((batch_size, 64, 64, 3))
 batch_labels = np.zeros((batch_size,1))
 while True:
   for i in range(batch_size):
     # choose random index in features
     index= rd.choice(len(features),1)
     batch_features[i] = some_processing(features[index])
     batch_labels[i] = labels[index]
   yield batch_features, batch_labels




checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
'''
train_generator  = generator(features, label, batch_size)
history = model.fit_generator(generator=train_generator,
                                steps_per_epoch=10//train_generator.batch_size,
                                epochs=epochs,
                                validation_data = validation_generator,
                                validation_steps = validation_generator.n//validation_generator.batch_size,
                                callbacks=callbacks_list
                                )



'''

#history = model.fit_generator(,steps_per_epoch = 10)
model.fit(x,y,batch_size=32,epochs = 1,validation_split = 0.1,callbacks =[tnsrb] )
epochs= 1




model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    

for i in train_data:
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap='gray')
        plt.show()
        break
        


# In[ ]:





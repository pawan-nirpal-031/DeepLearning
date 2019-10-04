# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist#28*28 size dataset of handwritten digits 0-9

(x_train,y_train),(x_test,y_test) = mnist.load_data()#loading data into a tuple and x_train is tuple of images and y_train are thier respactive labels


i = int(input("Enter the index of image "))

print("image data in pixel values 28*28 numpy matrix each pixel has a decimal value from 0 to 255: ")
print(x_train[i])
print("Your machine knows it is a  : ",y_train[i])
print("image data : ")
plt.imshow(x_train[i], cmap=plt.cm.binary)

x_train = tf.keras.utils.normalize(x_train,axis = 1)#data preprocessing technique 
x_test = tf.keras.utils.normalize(x_test,axis = 1)

print("after scaling the training data it looks like this for  1 st image  : ")
print(x_train[0])
print("image data : ")
plt.imshow(x_train[0], cmap=plt.cm.binary)
print("again your machine knows it is a  : ",y_train[0])

#here we define the arhchitechture of our neaural network aka model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation = tf.nn.softmax))


#compliation of model

model.compile(optimizer='adam',loss ='sparse_categorical_crossentropy',metrics =['accuracy'])
model.fit(x_train,y_train,epochs = 7)

#testing our model on testing data set  to get a genrealized algorithm to avoid overfitting 

val_loss,val_acc = model.evaluate(x_test,y_test)
print("Test performance of our model")
print("val loss : ",val_loss)
print("val acc : ",val_acc)

#let's make predictions now 

pred = model.predict([x_test])
print('Our prediction says it ia a: ')
#print(pred)

print(np.argmax(pred[0]))
print('let us have a look ')

plt.imshow(x_test[0])
plt.show()












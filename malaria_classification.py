#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import random
import cv2

dir_para = "/media/engineer/01D2FFBA0368DE90/cell_images/"

parasitized = []
for i in glob.glob(os.path.join(dir_para,'Parasitized/*.png')):
    img = plt.imread(i)
    parasitized.append(img)
    
uninfected = []
for i in glob.glob(os.path.join(dir_para,'Uninfected/*.png')):
    img_ = plt.imread(i)
    uninfected.append(img_)
    
label_parasitized = np.zeros([len(parasitized),1])
label_uninfected = np.ones([len(uninfected), 1])

images = uninfected + parasitized
labels = np.concatenate((label_uninfected,label_parasitized))

print(np.shape(images), np.shape(labels))

temp_data = list(zip(images, labels))
random.shuffle(temp_data)
images_shuffled, labels_shuffled = zip(*temp_data)

train_split = 90
val_split = 5
test_split = 5

i = train_split*np.shape(images_shuffled)[0]//100
i_val = val_split*np.shape(images_shuffled)[0]//100
i_test = test_split*np.shape(images_shuffled)[0]//100

def image_preprocessing(img):
    img = cv2.resize(img,(64,64))
    #hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return img

training_images = (images_shuffled[:i])
label_train = labels_shuffled[:i]
validation_images = (images_shuffled[i:i+i_val])
label_validation = labels_shuffled[i:i+i_val]
test_images = (images_shuffled[i+i_val:i+i_val+i_test])
label_test = labels_shuffled[i+i_val:i+i_val+i_test]


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import concatenate
from keras.callbacks import ModelCheckpoint 

train_images = []
for i in range(len(training_images)):
    train_images.append(cv2.resize(training_images[i],(64,64)))
    
    
valid_images = []
for i in range(len(validation_images)):
    valid_images.append(cv2.resize(validation_images[i],(64,64))) 
    
testing_images = []
for i in range(len(test_images)):
    testing_images.append(cv2.resize(test_images[i],(64,64))) 
    
X = np.array(train_images)
X = X.reshape(X.shape[0],64,64,3)

X_val = np.array(valid_images)
X_val = X_val.reshape(X_val.shape[0],64,64,3)

X_test = np.array(testing_images)
X_test = X_test.reshape(X_test.shape[0],64,64,3)

X = np.array(X)
label_train = np.array(label_train)

X_val = np.array(X_val)
label_validation = np.array(label_validation)


classifier = Sequential()
classifier.add(Conv2D(32,  (3, 3), padding='same', input_shape = (64, 64, 3)))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3, 3), padding='same'))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


classifier.add(Conv2D(128, (3, 3), padding='same'))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


classifier.add(Conv2D(256, (3, 3), padding = 'same'))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


classifier.add(Flatten())

classifier.add(Dense(units = 512, activation = 'relu'))#4096
classifier.add(Dense(units = 512, activation = 'relu'))#4096
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.summary()

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(zoom_range=0.2, 
                             shear_range=0.1,
                             horizontal_flip=True,
                            rotation_range=10.0)

datagen.fit(X)
datagen.fit(X_val)
batch_size = 16
epochs = 10

filepath="weights.malaria1.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


history = classifier.fit_generator(datagen.flow(X, label_train, batch_size=batch_size),
                    epochs=epochs,validation_data=(X_val, label_validation), callbacks=callbacks_list)

X_test = np.array(X_test)
classifier.load_weights("weights.malaria1.hdf5")
Y_pred = classifier.predict_classes(X_test)
# =============================================================================
# Y_pred = model.predict(X_test)
# =============================================================================
sum_= 0 
for i in range(len(X_test)):
    if((Y_pred[i]) == label_test[i]):
        sum_+=1

print(sum_/len(X_test))  #accuracy on test set

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


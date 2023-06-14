from sklearn import svm

import time
import os
import random
import cv2
import numpy as np
import imgaug.augmenters as iaa

start = time.time()

image_folder = "./images/data/data"
minimum_data = 11800
x_train = []
y_train = []

while len(x_train) <= minimum_data:
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            dataset_class = random.randint(0,1)
            cv_img = cv2.imread(os.path.join(root, file), 0)
            cv_img = cv2.resize(cv_img, (600,600)) # original (512,512)
            # cv_img = cv_img.reshape(-1)
            x_train.append(cv_img)
            y_train.append(dataset_class)

            if len(x_train) > minimum_data:
                break
        if len(x_train) > minimum_data:
            break

x_train = np.array(x_train)
y_train = np.array(y_train)

seq = iaa.Sequential([
    iaa.Affine(rotate=(-15, 15)),
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 3.0)),
    iaa.ScaleY((0.5, 1.5))
])

x_train_aug = seq(images=x_train)
y_train_aug = np.random.randint(2, size=(len(x_train_aug)))

x_train = x_train.reshape(len(x_train),-1)
x_train_aug = x_train_aug.reshape(len(x_train_aug),-1)

total_x_train = np.concatenate((x_train, x_train_aug), axis=0)
total_y_train = np.concatenate((y_train, y_train_aug), axis=0)
print(total_x_train.shape)
print(total_y_train.shape)

np_size = total_x_train.itemsize*total_x_train.size/1000000000
print(f"The memory size of numpy array arr is: {np_size} GB")

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(x_train, y_train)

print("Time taken :", (time.time() - start)/60, 'minutes')
import tensorflow as tf
from tensorflow.keras import layers
import imgaug.augmenters as iaa
import os
import cv2
import random
import numpy as np

image_folder = "./images/data/data"
minimum_data = 7700 #7600 #7800 # 11800 (8GB) 5900 (4GB) 2950 (2GB) 2700(1.9GB) 2400 (1.7GB) 2500(1.8GB)
x_train = []
y_train = []

# while len(x_train) <= minimum_data:
for root, dirs, files in os.walk(image_folder):
    for file in files:
        dataset_class = random.randint(0,1)
        cv_img = cv2.imread(os.path.join(root, file), 0)
        cv_img = cv2.resize(cv_img, (600,600)) # original (512,512)
        # cv_img = cv_img.reshape(-1)
        x_train.append(cv_img)
        y_train.append(dataset_class)

        #     if len(x_train) > minimum_data:
        #         break
        # if len(x_train) > minimum_data:
        #     break

x_train = np.array(x_train)
y_train = np.array(y_train)

total_x_train = x_train
total_y_train = y_train

# seq = iaa.Sequential([
#     iaa.Affine(rotate=(-15, 15)),
#     iaa.Fliplr(0.5),
#     iaa.GaussianBlur(sigma=(0, 3.0)),
#     iaa.ScaleY((0.5, 1.5))
# ])

# x_train_aug = seq(images=x_train)
# y_train_aug = np.random.randint(2, size=(len(x_train_aug)))

# x_train = x_train.reshape(len(x_train),-1)
# x_train_aug = x_train_aug.reshape(len(x_train_aug),-1)

# total_x_train = np.concatenate((x_train, x_train_aug), axis=0)
# total_y_train = np.concatenate((y_train, y_train_aug), axis=0)
# print(total_x_train.shape)
# print(total_y_train.shape)
total_x_train = total_x_train.astype(np.uint8)
total_y_train = total_y_train.astype(np.uint8)
y_np_size = total_y_train.itemsize*total_y_train.size/1000000000
print(f"The memory size of x numpy array arr is: {x_np_size} GB {total_x_train.dtype}")
print(f"The memory size of y numpy array arr is: {y_np_size} GB {total_y_train.dtype}")

# Normalize the data
total_x_train = total_x_train.astype("float32") / 255.0

# Define the model
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(600, 600, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(2, activation="softmax")
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Print the model summary
model.summary()

# Train the model
print("Starting model training...")
model.fit(total_x_train, total_y_train, batch_size=64, epochs=1)
print("Model training completed.")

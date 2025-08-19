#Importing Libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

#load dataset
(x_train,y_train),(x_test,y_test) = keras.datasets.cifar10.load_data()

#normalize image pixels
x_train,x_test = x_train/255.0,x_test/255.0

#define class tables
class_names = ["airplane", "automobile", "bird", "cat", "deer","dog", "frog", "horse", "ship", "truck"]

#display the images
# plt.figure(figsize=(10,5))
# for i in range(10):
#     plt.subplot(2,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(x_train[i])
#     plt.xlabel(class_names[y_train[i][0]])
# plt.show()

#Define the CNN Model
model = Sequential([
    layers.Conv2D(32,(3,3),activation = 'relu',input_shape = (32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation = 'relu'),
    layers.Flatten(),
    layers.Dense(64,activation = 'relu'),
    layers.Dense(10,activation = 'softmax'),
])

#compile model
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

#train the model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test,y_test))

#Evaluate model performance
test_loss, test_acc = model.evaluate(x_test,y_test, verbose=2)
print(f"Accuracy : {test_acc * 100:.2f}%")

#save model
model.save("imageClassifier.keras")
print("Model Saved")
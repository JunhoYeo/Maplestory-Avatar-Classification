# -*- coding: utf-8 -*-
import tensorflow as tf 
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import PIL
import AvatarDataset as ad

(train_images, train_labels), (test_images, test_labels) = ad.load_data()

train_images = train_images / 510.0
test_images = test_images / 510.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(96, 96)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(21, activation=tf.nn.softmax)
])

print model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

predictions = model.predict(test_images)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
        color = 'green' 
    else:
        color = 'red'
    plt.xlabel("{} ({})".format(
        str(predicted_label), 
        str(true_label)),
        color=color
    )
plt.show()

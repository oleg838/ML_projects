

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
# %matplotlib inline
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import image
from keras.utils import np_utils
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import utils

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boots']

#предварительная обработка данных
plt.figure()
plt.imshow(x_train[10])
plt.colorbar()
plt.grid(False)

#нормализация данных
x_train = x_train/255
x_test = x_test/255

#Создание модели нейронной сети
model = keras.Sequential([
                          keras.layers.Flatten(input_shape=(28,28)),
                          keras.layers.Dense(128, activation="relu"),
                          keras.layers.Dense(10, activation="softmax")
])

#Компиляция модели
model.compile(optimizer=tf.keras.optimizers.SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, epochs=20)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy', test_acc)

predictions = model.predict(x_train)

test_item = 10

predictions[test_item]

np.argmax(predictions[test_item])

y_train[test_item]

plt.figure()
plt.imshow(x_train[test_item])
plt.colorbar()
plt.grid(False)
class_names[np.argmax(predictions[test_item])]

pip install python-telegram-bot

import telegram
from telegram import Update
from telegram.ext import MessageHandler
from telegram.ext import Updater
from telegram.ext import CallbackContext
from telegram.ext import Filters

def message_handler(updater, context):

  img = updater.message.photo[-1].get_file().download()
  img_path=img
  img = image.load_img(img_path, target_size=(28,28), grayscale=True)
  plt.imshow(img, cmap='gray')
  plt.show()
  x = image.img_to_array(img)
  x = 255-x
  x /= 255
  x = np.expand_dims(x, axis=0)
  predictions = model.predict(x)
  predictions[0]
  a = class_names[np.argmax(predictions[0])]
  print(a)
  updater.message.reply_text(a)


    
def main():
  updater = Updater(
      token="token_place",
      use_context=True
  )
  updater.dispatcher.add_handler(MessageHandler(filters=Filters.all, callback=message_handler))
  updater.start_polling()
  updater.idle()
if __name__ == '__main__':
  main()
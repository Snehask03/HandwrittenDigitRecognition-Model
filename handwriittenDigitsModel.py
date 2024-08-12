import os 
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#train the model
model.fit(x_train,y_train, epochs=3)
model.save('Handwritten.model.keras')

model = tf.keras.models.load_model('Handwritten.model.keras')
loss, accuracy = model.evaluate(x_test, y_test)

print(f"loss: {loss}")
print(f"accuracy:{accuracy}")

image_number = 1
#if the image is in jpg format mention jpg instead of png
while os.path.isfile(f"Path to your input image{image_number}.png"):
    try:
        img = cv2.imread(f"path to your input image{image_number}.png",cv2.IMREAD_GRAYSCALE)
        print(f"Original Image Shape:{img.shape}")
        img = cv2.resize(img,(28,28))
        img = np.invert(img)
        img = img.reshape(1,28,28)
        img = tf.keras.utils.normalize(img,axis = 1)
        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)
        print(f"The digit is {predicted_digit}")
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.title(f"Predicted Digit: {predicted_digit}")
        plt.show()
    except:
        print("Error")
    finally:
        image_number+=1
    


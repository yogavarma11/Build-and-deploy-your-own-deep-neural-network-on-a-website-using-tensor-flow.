# Build-and-deploy-your-own-deep-neural-network-on-a-website-using-tensor-flow.
## AIM:
To Build-and-deploy-your-own-deep-neural-network-on-a-website-using-tensor-flow.
## ALGORITHM:
Step 1 — Build Your Neural Network in TensorFlow
  ~~~
   1. Create a deep learning model using TensorFlow/Keras.

   2. Train it with your dataset and save the final model.
~~~
Step 2 — Convert the Model for Web Use
Convert the trained model into TensorFlow.js format using tensorflowjs_converter so that it can run directly inside the browser.

Step 3 — Create a Web Interface (HTML/CSS)
Build a simple webpage with an input box, file upload, or drawing canvas where users can give input to your model.

Step 4 — Load the Model in the Browser
  1. In JavaScript (tfjs), load the converted model using:

  2. const model = await tf.loadGraphModel("model/model.json");
Step 5 — Run Inference in the Browser
Take user input from the webpage,Convert it into a tensor,Pass it to the model to get predictions.

Step 6 — Deploy the Website Online
Upload all files (HTML, JS, CSS, and model folder) to a hosting platform such as GitHub Pages, Netlify, or Vercelso anyone can use your neural network online.



## PROGRAM:
Name: YOGAVARMA B

Register Number:2305002029
~~~
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import os

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0
x_train = x_train[..., None]  # (N,28,28,1)
x_test  = x_test[..., None]

y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

def build_model():
    model = models.Sequential([
        layers.Input(shape=(28,28,1)),
        layers.Conv2D(16, 3, activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

model = build_model()
model.summary()

model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("Test acc:", acc)

import os
out_dir = "saved_model"
os.makedirs(out_dir, exist_ok=True)
model_path = os.path.join(out_dir, "my_model.keras")
model.save(model_path)  # SavedModel dir
print("Saved to", model_path)

~~~


### OUTPUT:

 MODEL SEQUENTIAL
<img width="810" height="434" alt="image" src="https://github.com/user-attachments/assets/f51f06d0-db35-4dff-8edb-677689c1f7bc" />
 EPOCH
<img width="815" height="120" alt="image" src="https://github.com/user-attachments/assets/540ce96d-b6c4-40a1-9f77-63afe0599975" />
 TEST ACCURACY
<img width="811" height="165" alt="image" src="https://github.com/user-attachments/assets/16576a33-fc0c-4918-b879-0f2779630a0b" />
### RESULT:
Thus, the Build-and-deploy-your-own-deep-neural-network-on-a-website-using-tensor-flow is successfully executed.

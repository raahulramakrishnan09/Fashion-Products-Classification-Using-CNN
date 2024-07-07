<h1 align="center" id="title">Fashion Clothing Classification</h1>

<p id="description">Developed a convolutional neural network to classify the images of fashion products applied convolutional filters to extract features from the input images reduced the spatial dimension by using MaxPooling and processed the features extracted by the convolutional layers and made final classifications. Fashion-MNIST is a dataset of a training set of 60000 examples and a test set of 10000 examples. Each example is a 28x28 grayscale image associated with a label from 10 classes</p>

<h2>🛠️ Installation Steps:</h2>

<p>1. Importing Libraries:</p>

```
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras
import layersdatasetsmodels
```

<p>2. Loading a Dataset</p>

```
(x_train y_train) (x_test y_test) = datasets.fashion_mnist.load_data()
x_train x_test = x_train / 255.0 x_test / 255.0
```

<p>3. Defining a Model</p>

```
model=models.Sequential()
model.add(layers.Conv2D(32(33)activation='relu'input_shape=(28281)))
model.add(layers.MaxPooling2D(22))
model.add(layers.Conv2D(28(33)activation='relu'))
model.add(layers.MaxPooling2D(22))
model.add(layers.Conv2D(28(33)activation='relu'))
```

```
model.add(layers.Flatten())
model.add(layers.Dense(512activation='relu'))
model.add(layers.Dense(256activation='relu'))
model.add(layers.Dense(64activation='relu'))
model.add(layers.Dense(10))
```

<p>5. Compile the Model</p>

```
model.compile(optimizer='adam'
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics=['accuracy'])
```

<p>6. Train the Model</p>

```
history=model.fit(x_trainy_trainvalidation_split=0.1epochs=15)
```

<p>7. Plotting Accuracy and Loss Curves</p>

```
import matplotlib.pyplot as plt
# Plot training and validation accuracy values plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy']) plt.xlabel('Epoch') plt.ylabel('Accuracy')
plt.title('Model Accuracy') plt.legend(['Training Data' 'Validation Data'] loc='lower right') plt.show()
plt.xlabel('Epoch') plt.ylabel('Loss') plt.title('Model Loss') plt.legend(['Training Data','Validation Data'],loc='upper right')
plt.show()
```

<p>8. Making Predictions</p>

```
y_pred = model.predict(x_test)
```

<p>9. Post-processing Predictions</p>

```
import numpy as np y_pred = np.argmax(y_pred axis=1) y_pred
```

<p>10. Evaluating Model Performance</p>

```
from sklearn.metrics import accuracy_score accuracy_score(y_test y_pred)
```

  
  
<h2>💻 Built with</h2>

Technologies used in the project:

*   Python
*   Tensorflow
*   Keras
*   Matplot
*   numpy
*   scikit-learn
*   Jupyter

MNIST Udemy Project 

## Import Relevant Libraries


```python
import numpy as np
import tensorflow as tf
```


```python
#pip install tensorflow_datasets
```


```python
import tensorflow_datasets as tfds
```

## Data


```python
mnist_dataset,mnist_info = tfds.load(name = 'mnist', with_info = True, as_supervised = True)
```


```python
mnist_train,mnist_test = mnist_dataset['train'],mnist_dataset['test']
```

### Make a validation dataset


```python
num_validation_samples = 0.1*mnist_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples,tf.int64)
```

#### Above we split the train dataset into validation dataset and train dataset by 10%
#### Also the same task is done on the test dataset in order to scale all the datasets


```python
num_test_samples = 0.1*mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples,tf.int64)
```

#### Now we define a function called Scale which will scale both the Train and Validation Datasets


```python
def scale(image,label):
    image = tf.cast(image,tf.float32)
    image/=255.
    return image,label
```


```python
scaled_train_and_validation_data = mnist_train.map(scale)
```


```python
test_data = mnist_test.map(scale)
```

### Now we are shuffling the dataset. Shuffling the dataset keeps the same info but changes the order


```python
BUFFER_SIZE = 10000
shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)
```


```python
validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
train_data = shuffled_train_and_validation_data.skip(num_validation_samples)

```


```python
BATCH_SIZE = 100

train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(num_validation_samples)
test_data = test_data.batch(num_test_samples)

```


```python
validation_inputs,validation_targets = next(iter(validation_data))
```

## Model

### Outline the model


```python
input_size = 784
output_size = 10
# Use same hidden layer size for both hidden layers. Not a necessity.
hidden_layer_size = 200
    
# define how the model will look like
model = tf.keras.Sequential([
    
    # the first layer (the input layer)
    # each observation is 28x28x1 pixels, therefore it is a tensor of rank 3
    # since we don't know CNNs yet, we don't know how to feed such input into our net, so we must flatten the images
    # there is a convenient method 'Flatten' that simply takes our 28x28x1 tensor and orders it into a (None,) 
    # or (28x28x1,) = (784,) vector
    # this allows us to actually create a feed forward neural network
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)), # input layer
    
    # tf.keras.layers.Dense is basically implementing: output = activation(dot(input, weight) + bias)
    # it takes several arguments, but the most important ones for us are the hidden_layer_size and the activation function
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 1st hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2nd hidden layer

    # the final layer is no different, we just make sure to activate it with softmax
    tf.keras.layers.Dense(output_size, activation='softmax') # output layer
])
```

### 1. Data
### 2. Model
### 3. Objective Function
### 4. Optimization Algorithm

### Data and Model have been done

### Choose the Optimizer and Loss Function


```python
model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
```

### Training


```python
NUM_EPOCHS = 5
early_stopping = tf.keras.callbacks.EarlyStopping(patience = 2)
# we fit the model, specifying the
# training data
# the total number of epochs
# and the validation data we just created ourselves in the format: (inputs,targets)
#model.fit(train_data, epochs=NUM_EPOCHS, validation_data=(validation_inputs, validation_targets), verbose =2)
model.fit(train_data,
    epochs=NUM_EPOCHS,
    callbacks = [early_stopping],
    validation_data=(validation_inputs, validation_targets),
    validation_steps = 20
    )
```

    Epoch 1/5
    540/540 [==============================] - 18s 33ms/step - loss: 0.2769 - accuracy: 0.9191 - val_loss: 0.1316 - val_accuracy: 0.9593
    Epoch 2/5
    540/540 [==============================] - 18s 33ms/step - loss: 0.1040 - accuracy: 0.9688 - val_loss: 0.1108 - val_accuracy: 0.9643
    Epoch 3/5
    540/540 [==============================] - 17s 32ms/step - loss: 0.0689 - accuracy: 0.9787 - val_loss: 0.0681 - val_accuracy: 0.9805
    Epoch 4/5
    540/540 [==============================] - 19s 35ms/step - loss: 0.0496 - accuracy: 0.9842 - val_loss: 0.0633 - val_accuracy: 0.9797
    Epoch 5/5
    540/540 [==============================] - 20s 36ms/step - loss: 0.0388 - accuracy: 0.9876 - val_loss: 0.0451 - val_accuracy: 0.9858





    <tensorflow.python.keras.callbacks.History at 0x10bf3b990>



## Test the model


```python
test_loss,test_accuracy = model.evaluate(test_data)
print('Test loss : {0:2f},Test Accuracy : {1:2f}%'.format(test_loss,test_accuracy*100))
```

         10/Unknown - 2s 208ms/step - loss: 0.0697 - accuracy: 0.9784Test loss : 0.069735,Test Accuracy : 97.839999%



```python

```


```python

```

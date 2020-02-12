## Import Libraries


```python
import numpy as np
import tensorflow as tf
```

## Data


```python
# here 'npz' is a temporary storage variable
npz  = np.load('Audiobooks_data_train.npz')
train_inputs,train_targets = npz['inputs'].astype(np.float),npz['targets'].astype(np.int)
```


```python
npz  = np.load('Audiobooks_data_validation.npz')
validation_inputs,validation_targets = npz['inputs'].astype(np.float),npz['targets'].astype(np.int)
```


```python
npz  = np.load('Audiobooks_data_test.npz')
test_inputs,test_targets = npz['inputs'].astype(np.float),npz['targets'].astype(np.int)
```

## Model

### Outline the model


```python
input_size = 10
hidden_layer_size = 100
output_size = 2

model = tf.keras.Sequential([
                    tf.keras.layers.Dense(hidden_layer_size,activation = 'relu'),
                    tf.keras.layers.Dense(hidden_layer_size,activation = 'relu'),
                    #tf.keras.layers.Dense(hidden_layer_size,activation = 'relu'),
                    #tf.keras.layers.Dense(hidden_layer_size,activation = 'relu'),
                    #tf.keras.layers.Dense(hidden_layer_size,activation = 'relu'),
                    #tf.keras.layers.Dense(hidden_layer_size,activation = 'relu'),
                    tf.keras.layers.Dense(output_size,activation = 'softmax')
                            ])

# we know that our output is a classifier so we use softmax in our output layer
```


```python
model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
```


```python
batch_size = 1000
max_epoch = 100

early_stopping = tf.keras.callbacks.EarlyStopping(patience = 2)
# By default, this object will monitor the validation loss and stop the training the first time the 
# validation loss starts to increase.

# indicating the batch size in model.fit() will automatically batch the data
model.fit(train_inputs,
         train_targets,
         batch_size = batch_size,
         epochs = max_epoch,
          callbacks = [early_stopping],
          validation_data = (validation_inputs,validation_targets),
          verbose = 2)
          
```

    Train on 2684 samples, validate on 1342 samples
    Epoch 1/100
    2684/2684 - 0s - loss: 0.7081 - accuracy: 0.5194 - val_loss: 0.6537 - val_accuracy: 0.6066
    Epoch 2/100
    2684/2684 - 0s - loss: 0.6324 - accuracy: 0.6461 - val_loss: 0.5906 - val_accuracy: 0.7481
    Epoch 3/100
    2684/2684 - 0s - loss: 0.5741 - accuracy: 0.7813 - val_loss: 0.5393 - val_accuracy: 0.8279
    Epoch 4/100
    2684/2684 - 0s - loss: 0.5261 - accuracy: 0.8271 - val_loss: 0.4943 - val_accuracy: 0.8435
    Epoch 5/100
    2684/2684 - 0s - loss: 0.4839 - accuracy: 0.8443 - val_loss: 0.4542 - val_accuracy: 0.8636
    Epoch 6/100
    2684/2684 - 0s - loss: 0.4471 - accuracy: 0.8551 - val_loss: 0.4186 - val_accuracy: 0.8756
    Epoch 7/100
    2684/2684 - 0s - loss: 0.4151 - accuracy: 0.8607 - val_loss: 0.3882 - val_accuracy: 0.8852
    Epoch 8/100
    2684/2684 - 0s - loss: 0.3891 - accuracy: 0.8659 - val_loss: 0.3627 - val_accuracy: 0.8867
    Epoch 9/100
    2684/2684 - 0s - loss: 0.3678 - accuracy: 0.8685 - val_loss: 0.3413 - val_accuracy: 0.8890
    Epoch 10/100
    2684/2684 - 0s - loss: 0.3517 - accuracy: 0.8718 - val_loss: 0.3245 - val_accuracy: 0.8897
    Epoch 11/100
    2684/2684 - 0s - loss: 0.3390 - accuracy: 0.8737 - val_loss: 0.3119 - val_accuracy: 0.8920
    Epoch 12/100
    2684/2684 - 0s - loss: 0.3301 - accuracy: 0.8752 - val_loss: 0.3027 - val_accuracy: 0.8949
    Epoch 13/100
    2684/2684 - 0s - loss: 0.3236 - accuracy: 0.8774 - val_loss: 0.2958 - val_accuracy: 0.8994
    Epoch 14/100
    2684/2684 - 0s - loss: 0.3183 - accuracy: 0.8785 - val_loss: 0.2909 - val_accuracy: 0.9031
    Epoch 15/100
    2684/2684 - 0s - loss: 0.3141 - accuracy: 0.8789 - val_loss: 0.2869 - val_accuracy: 0.9039
    Epoch 16/100
    2684/2684 - 0s - loss: 0.3092 - accuracy: 0.8815 - val_loss: 0.2824 - val_accuracy: 0.9046
    Epoch 17/100
    2684/2684 - 0s - loss: 0.3042 - accuracy: 0.8811 - val_loss: 0.2784 - val_accuracy: 0.9054
    Epoch 18/100
    2684/2684 - 0s - loss: 0.3000 - accuracy: 0.8830 - val_loss: 0.2749 - val_accuracy: 0.9098
    Epoch 19/100
    2684/2684 - 0s - loss: 0.2962 - accuracy: 0.8838 - val_loss: 0.2728 - val_accuracy: 0.9098
    Epoch 20/100
    2684/2684 - 0s - loss: 0.2926 - accuracy: 0.8852 - val_loss: 0.2714 - val_accuracy: 0.9083
    Epoch 21/100
    2684/2684 - 0s - loss: 0.2897 - accuracy: 0.8890 - val_loss: 0.2696 - val_accuracy: 0.9106
    Epoch 22/100
    2684/2684 - 0s - loss: 0.2869 - accuracy: 0.8893 - val_loss: 0.2675 - val_accuracy: 0.9106
    Epoch 23/100
    2684/2684 - 0s - loss: 0.2845 - accuracy: 0.8908 - val_loss: 0.2651 - val_accuracy: 0.9121
    Epoch 24/100
    2684/2684 - 0s - loss: 0.2819 - accuracy: 0.8908 - val_loss: 0.2625 - val_accuracy: 0.9121
    Epoch 25/100
    2684/2684 - 0s - loss: 0.2794 - accuracy: 0.8901 - val_loss: 0.2600 - val_accuracy: 0.9121
    Epoch 26/100
    2684/2684 - 0s - loss: 0.2772 - accuracy: 0.8901 - val_loss: 0.2577 - val_accuracy: 0.9121
    Epoch 27/100
    2684/2684 - 0s - loss: 0.2753 - accuracy: 0.8920 - val_loss: 0.2561 - val_accuracy: 0.9143
    Epoch 28/100
    2684/2684 - 0s - loss: 0.2733 - accuracy: 0.8931 - val_loss: 0.2548 - val_accuracy: 0.9151
    Epoch 29/100
    2684/2684 - 0s - loss: 0.2713 - accuracy: 0.8938 - val_loss: 0.2532 - val_accuracy: 0.9136
    Epoch 30/100
    2684/2684 - 0s - loss: 0.2695 - accuracy: 0.8953 - val_loss: 0.2519 - val_accuracy: 0.9143
    Epoch 31/100
    2684/2684 - 0s - loss: 0.2678 - accuracy: 0.8964 - val_loss: 0.2504 - val_accuracy: 0.9151
    Epoch 32/100
    2684/2684 - 0s - loss: 0.2662 - accuracy: 0.8968 - val_loss: 0.2492 - val_accuracy: 0.9151
    Epoch 33/100
    2684/2684 - 0s - loss: 0.2646 - accuracy: 0.8972 - val_loss: 0.2483 - val_accuracy: 0.9143
    Epoch 34/100
    2684/2684 - 0s - loss: 0.2633 - accuracy: 0.8987 - val_loss: 0.2474 - val_accuracy: 0.9151
    Epoch 35/100
    2684/2684 - 0s - loss: 0.2619 - accuracy: 0.8990 - val_loss: 0.2452 - val_accuracy: 0.9151
    Epoch 36/100
    2684/2684 - 0s - loss: 0.2609 - accuracy: 0.8979 - val_loss: 0.2447 - val_accuracy: 0.9151
    Epoch 37/100
    2684/2684 - 0s - loss: 0.2594 - accuracy: 0.8979 - val_loss: 0.2434 - val_accuracy: 0.9151
    Epoch 38/100
    2684/2684 - 0s - loss: 0.2580 - accuracy: 0.8979 - val_loss: 0.2432 - val_accuracy: 0.9151
    Epoch 39/100
    2684/2684 - 0s - loss: 0.2569 - accuracy: 0.9001 - val_loss: 0.2432 - val_accuracy: 0.9158
    Epoch 40/100
    2684/2684 - 0s - loss: 0.2559 - accuracy: 0.9001 - val_loss: 0.2422 - val_accuracy: 0.9158
    Epoch 41/100
    2684/2684 - 0s - loss: 0.2547 - accuracy: 0.8994 - val_loss: 0.2408 - val_accuracy: 0.9151
    Epoch 42/100
    2684/2684 - 0s - loss: 0.2538 - accuracy: 0.9001 - val_loss: 0.2400 - val_accuracy: 0.9151
    Epoch 43/100
    2684/2684 - 0s - loss: 0.2529 - accuracy: 0.8994 - val_loss: 0.2398 - val_accuracy: 0.9151
    Epoch 44/100
    2684/2684 - 0s - loss: 0.2519 - accuracy: 0.9001 - val_loss: 0.2399 - val_accuracy: 0.9165
    Epoch 45/100
    2684/2684 - 0s - loss: 0.2512 - accuracy: 0.9016 - val_loss: 0.2396 - val_accuracy: 0.9165
    Epoch 46/100
    2684/2684 - 0s - loss: 0.2502 - accuracy: 0.9013 - val_loss: 0.2373 - val_accuracy: 0.9151
    Epoch 47/100
    2684/2684 - 0s - loss: 0.2493 - accuracy: 0.8998 - val_loss: 0.2371 - val_accuracy: 0.9165
    Epoch 48/100
    2684/2684 - 0s - loss: 0.2487 - accuracy: 0.9016 - val_loss: 0.2385 - val_accuracy: 0.9173
    Epoch 49/100
    2684/2684 - 0s - loss: 0.2479 - accuracy: 0.9016 - val_loss: 0.2375 - val_accuracy: 0.9165





    <tensorflow.python.keras.callbacks.History at 0x13e429490>




```python
# model.fit() function has a function called "callbacks" which are called at certain points during model training
# Earlystopping checks if the validation loss at the current epoch with the previous epoch and stops when overfitted

```

## Test the model


```python
test_loss,test_accuracy = model.evaluate(test_inputs,test_targets)
print('\n Test loss : {0:.2f}. Test Accuracy : {1:.2f}%'.format(test_loss,test_accuracy*100))
```

    448/448 [==============================] - 0s 283us/sample - loss: 0.2642 - accuracy: 0.9040
    
     Test loss : 0.26. Test Accuracy : 90.40%



```python

```


```python

```


```python

```


```python

```


```python

```

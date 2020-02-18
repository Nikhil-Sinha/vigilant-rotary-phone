```python
# baseline model for the dogs vs cats dataset
import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
```

    Using TensorFlow backend.



```python
from matplotlib import pyplot
from matplotlib.image import imread
# define location of dataset
folder = '/Users/apple/Downloads/dogs-vs-cats/train/'
# plot first few images
for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # define filename
    filename = folder + 'dog.' + str(i) + '.jpg'
    # load image pixels
    image = imread(filename)
    # plot raw pixel data
    pyplot.imshow(image)
# show the figure
pyplot.show()
```


![png](output_1_0.png)



```python
for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # define filename
    filename = folder + 'cat.' + str(i) + '.jpg'
    # load image pixels
    image = imread(filename)
    # plot raw pixel data
    pyplot.imshow(image)
# show the figure
pyplot.show()
```


![png](output_2_0.png)



```python
import os
dataset_home = 'dataset_dogs_vs_cats/'
subdirs = ['train/','test/']
for subdir in subdirs:
    #create label subdirectories
    labeldirs = ['dogs/','cats/']
    for labeldir in labeldirs:
        newdir = dataset_home + subdir + labeldir
        os.makedirs(newdir,exist_ok = True)
```


```python
import random
from os import listdir
import shutil
random.seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.25
# copy training dataset images into subdirectories
src_directory = '/Users/apple/Downloads/dogs-vs-cats/train'
for file in listdir(src_directory):
    src = src_directory + '/' + file
    dst_dir = 'train/'
    if random.random() < val_ratio:
        dst_dir = 'test/'
    if file.startswith('cat'):
        dst = dataset_home + dst_dir + 'cats/' + file
        shutil.copyfile(src,dst)
    elif file.startswith('dog'):
        dst = dataset_home + dst_dir + 'dogs/' + file
        shutil.copyfile(src,dst)
```


```python
# block 1
#model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
#model.add(MaxPooling2D((2, 2)))
# block 2
#model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
#model.add(MaxPooling2D((2, 2)))
# block 3
#model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
#model.add(MaxPooling2D((2, 2)))
```


```python
#import tensorflow as tf

#def define_model():
    #model = Sequential()
    #model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    #model.add(MaxPooling2D((2, 2)))
    #model.add(Flatten())
    #model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    #model.add(Dense(1, activation='sigmoid'))
    # compile model
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    #return model
```


```python
# define model
#model = define_model()
```


```python
# create data generator
#datagen = ImageDataGenerator(rescale=1.0/255.0)
```


```python
# prepare iterators
# 1. We can use the flow_from_directory() function on the data generator and create one iterator for each of the train/ 
# and test/ directories. 
# 2 .We must specify that the problem is a binary classification problem via the “class_mode” argument, and to load the 
# images with the size of 200×200 pixels via the “target_size” argument. We will fix the batch size at 64.
#train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
    #class_mode='binary', batch_size=64, target_size=(200, 200))
#test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
    #class_mode='binary', batch_size=64, target_size=(200, 200))
```

 1. We can then fit the model using the train iterator (train_it). 
 2. We use (test_it) as a validation dataset during training.
 3. The number of steps for the train and test iterators must be specified.
 4. This is the number of batches that will comprise one epoch. This can be specified via the length of each iterator, and will be the total number of images in the train and test directories divided by the batch size (64).
 5. The model will be fit for 20 epochs, a small number to check if the model can learn the problem


```python
# fit the model
#history = model.fit_generator(train_it, steps_per_epoch = len(train_it),
                             #validation_data = test_it, validation_steps = len(test_it), epochs = 5, verbose = 0)
```


```python
# Evaluate model 
#_,acc = model.evaluate_generator(test_it, steps = len(test_it), verbose = 0)
#print('> %.3f' % (acc*100.0))
```


```python
# baseline model for the dogs vs cats dataset
import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
early_stopping = tf.keras.callbacks.EarlyStopping(patience = 2)
 
# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = adam(lr=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model
 
# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()
 
# run the test harness for evaluating a model
def run_test_harness():
    # define model
    model = define_model()
    # create data generator
    datagen = ImageDataGenerator(featurewise_center=True)
    # specify imagenet mean values for centering
    datagen.mean = [123.68, 116.779, 103.939]
    # prepare iterator
    train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
        class_mode='binary', batch_size=64, target_size=(200, 200))
    test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
        class_mode='binary', batch_size=64, target_size=(200,200))
    # fit model
    history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
        validation_data=test_it, validation_steps=len(test_it), epochs=10,callbacks = [early_stopping], verbose=1)
    # evaluate model
    _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=1)
    print('> %.3f' % (acc * 100.0))
    #learning curves
    summarize_diagnostics(history)
 
# entry point, run the test harness
run_test_harness()

```

    Found 18697 images belonging to 2 classes.
    Found 6303 images belonging to 2 classes.
    Epoch 1/10
    293/293 [==============================] - 1959s 7s/step - loss: 30.3685 - accuracy: 0.6142 - val_loss: 0.7319 - val_accuracy: 0.6472
    Epoch 2/10
    293/293 [==============================] - 2171s 7s/step - loss: 0.5561 - accuracy: 0.7181 - val_loss: 0.6096 - val_accuracy: 0.6595
    Epoch 3/10
    293/293 [==============================] - 2144s 7s/step - loss: 0.4041 - accuracy: 0.8185 - val_loss: 0.5016 - val_accuracy: 0.6621
    Epoch 4/10
    293/293 [==============================] - 2017s 7s/step - loss: 0.2383 - accuracy: 0.9033 - val_loss: 0.4284 - val_accuracy: 0.6573
    Epoch 5/10
    293/293 [==============================] - 2416s 8s/step - loss: 0.1221 - accuracy: 0.9580 - val_loss: 0.9640 - val_accuracy: 0.6752
    Epoch 6/10
    293/293 [==============================] - 2121s 7s/step - loss: 0.0694 - accuracy: 0.9795 - val_loss: 1.8018 - val_accuracy: 0.6717
    99/99 [==============================] - 248s 3s/step
    > 67.174



```python
# making the final dataset
from os import makedirs
from os import listdir
from shutil import copyfile
# create directories
dataset_home = 'finalize_dogs_vs_cats/'
# create label subdirectories
labeldirs = ['dogs/', 'cats/']
for labldir in labeldirs:
    newdir = dataset_home + labldir
    makedirs(newdir, exist_ok=True)
# copy training dataset images into subdirectories
src_directory = '/Users/apple/Downloads/dogs-vs-cats/train/'
for file in listdir(src_directory):
    src = src_directory + file
    if file.startswith('cat'):
        dst = dataset_home + 'cats/'  + file
        copyfile(src, dst)
    elif file.startswith('dog'):
        dst = dataset_home + 'dogs/'  + file
        copyfile(src, dst)
```


```python
datagen = ImageDataGenerator(featurewise_center=True)
train_it = datagen.flow_from_directory('finalize_dogs_vs_cats/',
    class_mode='binary', batch_size=64, target_size=(200, 200))
```

    Found 25000 images belonging to 2 classes.



```python
model = define_model()
# fit model
model.fit_generator(train_it, steps_per_epoch=len(train_it), epochs=10,callbacks = [early_stopping], verbose=1)
```

    Epoch 1/10


    /Users/apple/opt/anaconda3/envs/py3-TF2.0/lib/python3.7/site-packages/keras_preprocessing/image/image_data_generator.py:716: UserWarning: This ImageDataGenerator specifies `featurewise_center`, but it hasn't been fit on any training data. Fit it first by calling `.fit(numpy_data)`.
      warnings.warn('This ImageDataGenerator specifies '


    391/391 [==============================] - 2024s 5s/step - loss: 98.3670 - accuracy: 0.5777
    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy
    Epoch 2/10
    391/391 [==============================] - 2013s 5s/step - loss: 0.6182 - accuracy: 0.6374
    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy
    Epoch 3/10
    391/391 [==============================] - 2029s 5s/step - loss: 0.5715 - accuracy: 0.6822
    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy
    Epoch 4/10
    391/391 [==============================] - 2016s 5s/step - loss: 0.5313 - accuracy: 0.7201
    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy
    Epoch 5/10
    391/391 [==============================] - 2275s 6s/step - loss: 0.4759 - accuracy: 0.7574
    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy
    Epoch 6/10
    391/391 [==============================] - 2195s 6s/step - loss: 0.4416 - accuracy: 0.7827
    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy
    Epoch 7/10
    391/391 [==============================] - 2201s 6s/step - loss: 0.3785 - accuracy: 0.8154
    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy
    Epoch 8/10
    391/391 [==============================] - 2035s 5s/step - loss: 0.3300 - accuracy: 0.8511
    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy
    Epoch 9/10
    391/391 [==============================] - 2045s 5s/step - loss: 0.2802 - accuracy: 0.8749
    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy
    Epoch 10/10
    391/391 [==============================] - 2143s 5s/step - loss: 0.2337 - accuracy: 0.8987
    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy





    <keras.callbacks.callbacks.History at 0x1445eb910>




```python
# save model
model.save('final_model.h5')
```


```python
# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(200, 200))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 200, 200, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img

# load an image and predict the class
def run_example():
    # load the image
    img = load_image('/Users/apple/Desktop/sample_image.jpg')
    # load model
    model = load_model('final_model.h5')
    # predict the class
    result = model.predict(img)
    print(result[0])
    

# entry point, run the example
run_example()


```

    [0.9902644]



```python

```

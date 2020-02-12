## Load the data from the .csv


```python
import numpy as np
from sklearn import preprocessing
```


```python
raw_csv_data = np.loadtxt('Audiobooks_data.csv',delimiter = ',')
# this contains all the raw data

# now to store all the data apart from the ids and targets which are in the 1st and last column.
unscaled_inputs_all = raw_csv_data[:,1:-1]

# storing the targets into another separate variable
targets_all = raw_csv_data[:,-1]
```

## Balance the dataset


```python
# Count the number of 1s and 0s
# Then we will keep as manay 0s as 1s (Delete the rest)
# If we sum all the targets then we get the number of targets as the targets are either 0s or 1s

num_one_targets = int(np.sum(targets_all))
zero_targets_counter = 0
indices_to_remove = []

for i in range(np.sum(targets_all.shape[0])):
    if targets_all[i]==0:
        zero_targets_counter +=1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)

unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all,indices_to_remove,axis = 0)
targets_equal_priors = np.delete(targets_all,indices_to_remove,axis = 0)

                                 
```

## Standardize the inputs


```python
scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)
```

## Shuffle the data


```python
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]
```

## Split the dataset into Test, Validation and Train dataset


```python
samples_count = shuffled_inputs.shape[0]

# Count the samples in each subset, assuming we want 80-10-10 distribution of training, validation, and test.
# Naturally, the numbers are integers.
train_samples_count = int(0.6 * samples_count)
validation_samples_count = int(0.3 * samples_count)

# The 'test' dataset contains all remaining data.
test_samples_count = samples_count - train_samples_count - validation_samples_count

# Create variables that record the inputs and targets for training
# In our shuffled dataset, they are the first "train_samples_count" observations
train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

# Create variables that record the inputs and targets for validation.
# They are the next "validation_samples_count" observations, folllowing the "train_samples_count" we already assigned
validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

# Create variables that record the inputs and targets for test.
# They are everything that is remaining.
test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]

# We balanced our dataset to be 50-50 (for targets 0 and 1), but the training, validation, and test were 
# taken from a shuffled dataset. Check if they are balanced, too. Note that each time you rerun this code, 
# you will get different values, as each time they are shuffled randomly.
# Normally you preprocess ONCE, so you need not rerun this code once it is done.
# If you rerun this whole sheet, the npzs will be overwritten with your newly preprocessed data.

# Print the number of targets that are 1s, the total number of samples, and the proportion for training, validation, and test.
print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)
```

    1350.0 2684 0.5029806259314457
    674.0 1342 0.5022354694485842
    213.0 448 0.47544642857142855


## Save the three datasets in *.npz


```python
# Save the three datasets in *.npz.
# In the next lesson, you will see that it is extremely valuable to name them in such a coherent way!

np.savez('Audiobooks_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Audiobooks_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('Audiobooks_data_test', inputs=test_inputs, targets=test_targets)
```


```python

```


```python

```


```python

```

```python
import numpy as np
from PIL import  Image
import os, cv2
# Method to train custom classifier to recognize face
def train_classifer(data_path):
    # Read all the images in custom data-set
    data_path = '/Users/apple/Desktop/Test/dataset'
    onlyfiles = [os.path.join(data_path,f) for f in os.listdir(data_path) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith('.png')]
    Training_Data,ids = [],[]

    for i, files in enumerate(onlyfiles):
        image_path = onlyfiles[i]
        #print(image_path)
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        imagesNp = np.array(images,'uint8')
        Training_Data.append(imagesNp)
        #print(Training_Data)
        ids.append(i)


    ids = np.asarray(ids, dtype=np.int32)
    print(ids)

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Training_Data), np.asarray(ids))
    model.write("classifier.yml")
    print("model trained")

data_path = "/Users/apple/Desktop/Test"
train_classifer(data_path)
```

    [ 0  1  2  3  4  5  6  7  8  9 10 11 12]
    model trained



```python

```


```python

```

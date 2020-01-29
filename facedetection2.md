```python
import cv2
import numpy as np
import os
#from os import listdir
#from os.path import isfile, join

data_path = '/Users/apple/Desktop/Test/dataset'
onlyfiles = [os.path.join(data_path,f) for f in os.listdir(data_path) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith('.png')]
Training_Data,ids = [],[]

for i, files in enumerate(onlyfiles):
    image_path = onlyfiles[i]
    #print(image_path)
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    imagesNp = np.array(images,'uint8')
    Training_Data.append(imagesNp)
    ids.append(i)


ids = np.asarray(ids, dtype=np.int32)
print(ids)

model = cv2.face.LBPHFaceRecognizer_create()
model.read('classifier.yml')
model.train(np.asarray(Training_Data), np.asarray(ids))

print("Model Training Complete!!!!!")

```

    [ 0  1  2  3  4  5  6  7  8  9 10 11 12]
    Model Training Complete!!!!!



```python
import cv2
import numpy as np
import os
def face_detector(img, faceCascade,model):
    faceCascade = cv2.CascadeClassifier('/Users/apple/opt/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (100,100))

    return img

model = cv2.face.LBPHFaceRecognizer_create()
model.read('classifier.yml')
#model.train(np.asarray(Training_Data), np.asarray(ids))
cap = '/Users/apple/Desktop/psgold.jpeg'
frame = cv2.imread(cap)
faceCascade = cv2.CascadeClassifier('/Users/apple/opt/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
face = face_detector(frame,faceCascade,model)
face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
result = model.predict(face)
print(result[1])

if result[1] < 500:
        confidence = int(100*(1-(result[1])/300))
        display_string = str(confidence)+'% Confidence it is Neymar'
        print(display_string)
        cv2.putText(frame,display_string,(20,40), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        #cv2.imshow('face',frame)
        
        #if confidence > 75:
            #cv2.putText(frame, "Neymar", (100,120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            #cv2.imshow('Face Cropper', frame)

        #else:
            #cv2.putText(frame, "Unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255,0), 2)
            #cv2.imshow('Face Cropper', frame)
    
cv2.waitKey(1000)
cv2.destroyAllWindows()
```

    52.69505900863386
    82% Confidence it is Neymar



```python

```


```python

```

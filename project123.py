import cv2
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps
import os, ssl, time

X=np.load('image.npz')['arr_0']
y=pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses=len(classes)

X_trained, X_test, y_trained, y_test=train_test_split(X, y, train_size=7500, random_state=9, test_size=2500)
X_train_scale=X_trained/255.0
X_test_scale=X_test/255.0
classifier=LogisticRegression(solver='saga', multi_class='multinomial')
clf=classifier.fit(X_train_scale, y_trained)
y_predict=clf.predict(X_test_scale)
accuracy=accuracy_score(y_test, y_predict)
print(accuracy)

cap=cv2.VideoCapture(0 + cv2.CAP_DSHOW)
while(True):
    try:
        ret, frame=cap.read()
        grey=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width=grey.shape
        upper_left=(int(width/2-56), int(height/2-56))
        bottom_right=(int(width/2+56), int(height/2+56))
        cv2.rectangle(grey, upper_left, bottom_right, (0, 255, 0), 2)
        roi=grey[upper_left[1]: bottom_right[1], upper_left[0]:bottom_right[0]]
        im_PIL=Image.fromarray(roi)
        image_bw=im_PIL.convert('L')
        image_bw_resize=image_bw.resize((28, 28), Image.ANTIALIAS)
        image_bw_inverted=PIL.ImageOps.invert(image_bw_resize)
        pixel_filter=20
        min_pixel=np.percentile(image_bw_inverted, pixel_filter)
        image_bw_scaled=np.clip(image_bw_inverted-min_pixel, 0, 255)
        max_pixel=np.max(image_bw_inverted)
        image_bw_scaled=np.asarray(image_bw_scaled)/max_pixel
        test_sample=np.array(image_bw_scaled).reshape(1, 784)
        test_predict=clf.predict(test_sample)
        print("predicted class is : ", test_predict)
        cv2.imhow('frame', grey)
        if cv2.waitKey(1)& 0xff==ord('q'):
            break
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()
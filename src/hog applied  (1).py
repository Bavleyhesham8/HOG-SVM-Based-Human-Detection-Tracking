#!/usr/bin/env python
# coding: utf-8

# In[8]:


import cv2
import numpy as np
import random
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import TruncatedSVD

def create_shape(shape, img_size=64, rotation=True, add_noise=True):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    if shape == "square":
        cv2.rectangle(img, (16, 16), (48, 48), 255, -1)
    elif shape == "circle":
        cv2.circle(img, (32, 32), 16, 255, -1)
    elif shape == "triangle":
        pts = np.array([[32, 16], [16, 48], [48, 48]], np.int32)
        cv2.fillPoly(img, [pts], 255)
    elif shape == "line":
        cv2.line(img, (16, 32), (48, 32), 255, 5)
    elif shape == "star":
        pts = np.array([[32, 16], [36, 28], [48, 28], [38, 36],
                       [42, 48], [32, 40], [22, 48], [26, 36],
                       [16, 28], [28, 28]], np.int32)
        cv2.fillPoly(img, [pts], 255)

    if rotation:
        angle = random.randint(0, 360)
        M = cv2.getRotationMatrix2D((img_size/2, img_size/2), angle, 1.0)
        img = cv2.warpAffine(img, M, (img_size, img_size), borderValue=0)

    if add_noise:
        noise = np.random.randint(0, 40, (img_size, img_size), dtype=np.uint8)
        img = cv2.add(img, noise)

    return img
    
shapes = ["square", "circle", "triangle", "line", "star"]
X, y = [], []

print("[INFO] Generating dataset...")
for label, shape in enumerate(shapes):
    for i in range(300):  # 300 images per shape
        img = create_shape(shape)
        features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("[INFO] Reducing dimensions with SVD...")
svd = TruncatedSVD(n_components=50)  
X_svd = svd.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_svd, y, test_size=0.3, random_state=42)

print("[INFO] Training SVM...")
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=shapes))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

test_img = create_shape("star")
test_features, _ = hog(test_img, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
test_features_svd = svd.transform([test_features])
prediction = clf.predict(test_features_svd)[0]
print("Predicted Shape:", shapes[prediction])

cv2.imshow("Test Image", test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[1]:


import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import color
import numpy as np
image = cv2.imread(r"C:\Users\ashou\.cache\kagglehub\datasets\jcoral02\inriaperson\versions\1\Test\JPEGImages\crop_000017.png")  
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
features, hog_image = hog(gray,
                          orientations=18,
                          pixels_per_cell=(2, 2),
                          cells_per_block=(4, 4),
                          visualize=True,
                          block_norm='L2-Hys')
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Gradient Magnitude")
plt.imshow(gradient_magnitude, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("HOG Visualization")
plt.imshow(hog_image, cmap='gray')
plt.axis('off')


# In[ ]:


import cv2
image = cv2.imread(r"C:\Users\ashou\.cache\kagglehub\datasets\jcoral02\inriaperson\versions\1\Test\JPEGImages\crop001512.png") 
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
rects, weights = hog.detectMultiScale(image,
                                      winStride=(8, 8),
                                      padding=(8, 8),
                                      scale=1.05)
for (x, y, w, h) in rects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Pedestrian Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[1]:


import cv2
import time
import numpy as np
from imutils.object_detection import non_max_suppression  

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(r"C:\Users\ashou\Downloads\People Walking Free Stock Footage, Royalty-Free No Copyright Content.mp4")

while True:
    start_time = time.time()  
    
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))

    rects, weights = hog.detectMultiScale(frame,
                                          winStride=(8, 8),
                                          padding=(8, 8),
                                          scale=1.05)
    rects_np = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects_np, probs=None, overlapThresh=0.65)

    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
    cv2.putText(frame, f"People: {len(pick)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("People Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:





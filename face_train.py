import os
import face_recognition
import cv2
import numpy as np
import pickle

path = 'dataset'
images = []
classNames = []
image_list = os.listdir(path) #os.listdir(path) this loads all image filenames from dataset/

for img_name in image_list:
    img = cv2.imread(f'{path}/{img_name}') #cv2.imread() reads each image using OpenCV
    images.append(img)
    classNames.append(os.path.splitext(img_name)[0])  # "Aneesh1.jpg" → "Aneesh1"

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Required by face_recognition
        encodes = face_recognition.face_encodings(img) #face_recognition.face_encodings() converts the image into a 128D face vector
        if len(encodes) > 0:
            encodeList.append(encodes[0])
    return encodeList

print("[INFO] Encoding images...")
encodeListKnown = findEncodings(images)
print(f"[INFO] Encoding complete! Total faces encoded: {len(encodeListKnown)}")

data = {"encodings": encodeListKnown, "names": classNames}
with open("known_encodings.pkl", "wb") as f:
    pickle.dump(data, f) # saves the vectors and names to a file that i will load later

print("[INFO] Encodings saved to 'known_encodings.pkl'")

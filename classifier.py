from edgetpu.classification.engine import ClassificationEngine
from PIL import Image
import cv2
import re
import os
# import tensorflow as tf
import numpy as np
# the TFLite converted to be used with edgetpu
modelPath = 'model.tflite'

# The path to labels.txt that was downloaded with your model
labelPath = 'labels.txt'

# This function parses the labels.txt and puts it in a python dictionary
def loadLabels(labelPath):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(labelPath, 'r', encoding='utf-8') as labelFile:
        lines = (p.match(line).groups() for line in labelFile.readlines())
        return {int(num): text.strip() for num, text in lines}

# This function takes in a PIL Image from any source or path you choose
def normalize(arr):
    arr = arr.astype('float')
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255./(maxval-minval))
    return arr


def classifyImage(image_path, engine):
    # Load and format your image for use with TM2 model
    # image is reformated to a square to match training
    # print(image_path)
    # image = Image.open(image_path)
    # image.resize((224, 224))
    # image = tf.keras.preprocessing.image.img_to_array(image_path)
    # image = tf.expand_dims(image, axis=0)
    # Classify and ouptut inference
    image = np.array(image_path)
    image = Image.fromarray(normalize(image).astype('uint8'), 'RGB')
    classifications = engine.ClassifyWithImage(image)
    return classifications

def main():
    # Load your model onto your Coral Edgetpu
    engine = ClassificationEngine(modelPath)
    labels = loadLabels(labelPath)

    cap = cv2.VideoCapture(-1)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Format the image into a PIL Image so its compatable with Edge TPU
        cv2_im = frame
        pil_im = Image.fromarray(cv2_im)

        # Resize and flip image so its a square and matches training
        pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)
        pil_im = pil_im.transpose(Image.FLIP_LEFT_RIGHT)
        # Classify and display image
        results = classifyImage(pil_im, engine)
        cv2.imshow('frame', cv2_im)
        print(results)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

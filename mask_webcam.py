import cv2
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import re
import time
from gpiozero import LED
from time import sleep

led = LED(17)

np.set_printoptions(suppress=True)

def pred_text(prediction):
    perc = np.max(prediction[0])
    perc = str(np.round(perc*100))
    return f'{encoder[np.argmax(prediction[0])]} : {perc} %'

def main():
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)

    font = cv2.FONT_HERSHEY_SIMPLEX
    upperLeftCornerOfText = (10, 50)
    bottomLeftCornerOfText = (10, 450)
    fontScale = 1
    fontColor = (0,100,0)
    lineType = 2
    prev_frame_time = 0
    new_frame_time = 0
    cap = cv2.VideoCapture(-1)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame
        pil_im = Image.fromarray(cv2_im)

        # Resize and flip image so its a square and matches training
        image = ImageOps.fit(pil_im, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        # Classify and display image
        prediction = model.predict(data)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))
        if prediction[0][1] > 0.7:
            led.on()
        else:
            led.off()
        cv2.putText(cv2_im, pred_text(prediction),
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

        cv2.putText(cv2_im, f'fps = {fps}', 
            upperLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.imshow('Mask Detector', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

model = tensorflow.keras.models.load_model('keras_model.h5')
pattern = re.compile(r'\s*(\d+)(.+)')
with open('labels.txt', 'r', encoding='utf-8') as labelFile:
    lines = (pattern.match(line).groups() for line in labelFile.readlines())
    encoder =  {int(num): text.strip() for num, text in lines}

main()
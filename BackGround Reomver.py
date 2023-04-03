# python version 3.10
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

# Read the webcam
cap = cv2.VideoCapture(0)
# set the width and height
cap.set(3, 640)
cap.set(4, 480)

# Create SelfiSegmentation object
background_remover = SelfiSegmentation(model=1)
# Add FPS
fps_reader = cvzone.FPS()

img_path = []

all_img = os.listdir("Resized Images")
# print(all_img)

for i in all_img:
    img = cv2.imread(f"Resized Images/{i}")
    img_path.append(img)

image_index = 0

while True:
    success, img = cap.read()
    if success:
        # remove the background
        # rm_img = background_remover.removeBG(img=img, imgBg=(255, 0, 255), threshold=0.5)
        rm_img = background_remover.removeBG(img=img, imgBg=img_path[image_index], threshold=0.3)
        # stacked image
        stacked_img = cvzone.stackImages([img, rm_img], cols=2, scale=1)
        fps_reader.update(img=stacked_img, pos=(20, 50), color=(255, 0, 0), scale=1, thickness=2)
        cv2.imshow('Video', stacked_img)
        key = cv2.waitKey(1)
        if key == ord('a'):
            if image_index > 0:
                image_index -= 1
        elif key == ord('d'):
            if image_index < len(img_path) - 1:
                image_index += 1
        elif key & 0xFF == ord('q'):
            break
    else:
        exit()
cap.release()
cv2.destroyAllWindows()

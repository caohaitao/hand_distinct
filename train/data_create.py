__author__ = 'ck_ch'
import cv2
import numpy as np

index=0

def image_create(id):
    width = 28
    height = 28
    #image = np.zeros((height,width,3),dtype=np.uint8)
    image = np.full((height,width,3),fill_value=255,dtype=np.uint8)

    file_path = format("data\\%d_%d.jpg"%(index,id))

    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(image, str(id), (6, 20), font, 0.8, (0, 0, 255), 1, False)

    #cv2.imshow("img",image)
    #cv2.waitKey(0)
    cv2.imwrite(file_path,image)

for i in range(10):
    image_create(i)
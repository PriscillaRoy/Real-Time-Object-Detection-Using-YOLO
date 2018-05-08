from darkflow.net.build import TFNet
import cv2
import matplotlib.pyplot as plt

options = {"model": "cfg/tiny-yolo-voc-10c.cfg", "load": 6375, "threshold": 0.01}

tfnet = TFNet(options)

imgcv = cv2.imread("51409302.jpg")
result = tfnet.return_predict(imgcv)
print(result)
print(len(result))
for i in range(0,len(result)):
    tl = (result[i]['topleft']['x'], result[i]['topleft']['y'])
    br = (result[i]['bottomright']['x'], result[i]['bottomright']['y'])
    label = result[i]['label']
    # add the box and label and display it
    img = cv2.rectangle(imgcv, tl, br, (0, 255, 0), 2)
    img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
cv2.imwrite('Result.jpg',img)
cv2.imshow('img',img)
cv2.waitKey(0)

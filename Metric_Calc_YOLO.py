from darkflow.net.build import TFNet
from xml.dom import minidom
import numpy as np
import cv2
import xml.etree.ElementTree as ET


def bb_intersection_over_union( boxA, boxB):
    # I have taken following code snippet is taken from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return (iou)


#The configuration and models
options = {"model": "cfg/tiny-yolo-voc-10c.cfg", "load": 6375, "threshold": 0.1}
xml_path = "all_logos.xml"
img_path = "all_logos.jpg"
tfnet = TFNet(options)

imgcv = cv2.imread("all_logos.jpg")
result = tfnet.return_predict(imgcv)
print(result)
#Objects detected by the model
object_list = [['#' for x in range(5)] for y in range(len(result))]
for i in range(0,len(result)):
    tl = (result[i]['topleft']['x'], result[i]['topleft']['y'])
    br = (result[i]['bottomright']['x'], result[i]['bottomright']['y'])
    label = result[i]['label']
    # add the box and label and display it
    img = cv2.rectangle(imgcv, tl, br, (0, 255, 0), 2)
    img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    object_list[i] = [label, tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]]

#Storing the result to a folder
cv2.imwrite('Result.jpg',img)

# To get the bounding boxes from ground truth file
doc = minidom.parse(xml_path)
# Getting points list from xml file
objects = doc.getElementsByTagName("object")
tree = ET.parse(xml_path)
root = tree.getroot()
ground_truth_list = [['#' for x in range(5)] for y in range(len(objects))]
i = 0
for objects in root.findall('object'):

    for sub_object in objects:
        if sub_object.tag == 'bndbox':
            for child in sub_object:

                if child.tag == 'xmin':
                    xmin = int(child.text)
                if child.tag == 'ymin':
                    ymin = int(child.text)
                if child.tag == 'xmax':
                    xmax = int(child.text)
                if child.tag == 'ymax':
                    ymax = int(child.text)
        if sub_object.tag == 'name':
            name = sub_object.text
    ground_truth_list[i] = [name, xmin, ymin, xmax - xmin, ymax - ymin]
    i = i + 1

#Metrics
tp = 0
fp = 0
tn = 0
fn = 0
found = 0

for object in object_list:
    boxA = np.array([object[1], object[2], object[3], object[4]])
    i = 0
    found = 0
    if(len(ground_truth_list)>0):
        for box in ground_truth_list:
            i = i + 1
            label1 = str(object[0])
            label2 = str(box[0])
            #Check if the labels are equal
            if(label1.lower() == label2.lower()):
                boxB = np.array([box[1], box[2], box[3], box[4]])
                #Calculating the Intersection over Union
                iou = bb_intersection_over_union(boxB, boxA)
                print("IOU BEFORE>>", iou)
                if (iou > 0.5):
                    print("IOU", iou)
                    #If the detected image matches with that of the ground truth - True Positive
                    ground_truth_list.pop(i)
                    tp = tp + 1
                    found = 1
                    break
    else:
        break
    #If the particular detected object is not found in the ground truth data, then it is a false positive
    if(found == 0):
        fp = fp + 1
#All the remaining items in the ground truth file are the not detected images - True Negatives
tn = len(ground_truth_list)

#print('TP >>', tp, "FN>>", fn,"TN >>", tn, "FP", fp)
Accuracy = (tp + fn)/(tp + fp + tn + fn)
print("Accuracy : " , Accuracy)
Precision = (tp )/(tp + fp )
print("Precision : " , Precision)
Recall = (tp )/(tp + fn )
print("Recall : " , Recall)
F_Statistic = 2*(Precision * Recall)/(Precision + Recall)
print("F Statistic :",F_Statistic)





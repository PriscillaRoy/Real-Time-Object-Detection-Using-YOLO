import cv2

from darkflow.net.build import TFNet

import numpy as np

import time

options = {"model": "cfg/tiny-yolo-voc-20c.cfg", "load": 45875, "threshold": 0.41}



tfnet = TFNet(options)



capture = cv2.VideoCapture('tt.mp4')

colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))




while (capture.isOpened()):

    stime = time.time()

    ret, frame = capture.read()

    if ret:

        results = tfnet.return_predict(frame)

        for color, result in zip(colors, results):

            tl = (result['topleft']['x'], result['topleft']['y'])

            br = (result['bottomright']['x'], result['bottomright']['y'])

            label = result['label']

            frame = cv2.rectangle(frame, tl, br, color, 7)

            frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        out.write(frame)

        #cv2.imshow('frame', frame)

        print('FPS {:.1f}'.format(1 / (time.time() - stime)))

        if cv2.waitKey(1) & 0xFF == ord('q'):

            break

    else:

        capture.release()
        out.release()
        cv2.destroyAllWindows()

        break
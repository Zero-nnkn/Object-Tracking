import sys
sys.path.append('deep_sort/deep/reid')

import cv2
import numpy as np
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort


NUM_FRAME = 20
cap = cv2.VideoCapture('TestA/testA_5.mp4')
cfg = get_config('deep_sort/configs/deep_sort.yaml')
ds = DeepSort(cfg.DEEPSORT.MODEL_TYPE,
                        'cpu',
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, 
                        n_init=cfg.DEEPSORT.N_INIT, 
                        nn_budget=cfg.DEEPSORT.NN_BUDGET)

if (cap.isOpened()== False):
    print("Error opening video stream or file")


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged

roi = np.array([]).reshape(-1, 4)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:

        img = frame #cv2.imread('BG.png')
        height, width = img.shape[0], img.shape[1]

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        thresh_inv = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,39,1)

        edges = auto_canny(thresh_inv)

        ctrs, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_ctrs = sorted(ctrs, key=len, reverse=True)[:3]
        img_area = img.shape[0]*img.shape[1]


        # for y_start in WINDOW_SIZE[1]:
        #     for x_start in WINDOW_SIZE[0]: 
        max_area = 0
        s = None
        for i, ctr in enumerate(sorted_ctrs):
            x, y, w, h = cv2.boundingRect(ctr)
            roi_area = w*h
            roi_ratio = roi_area/img_area

            if w*h > max_area and w < 3*h and w > 1.2*h and h > height/4.5 and w > width/4.5:
                s = [x, y, w, h]
                max_area = w*h

        if s is not None:
            s = [s[0] + s[2] // 2, s[1] + s[3]//2, s[2], s[3]]
            xywhs = np.array(s).reshape(-1, 4)
            cfs = np.array([0.9]).reshape(-1, 1)
            clss = np.array([1]).reshape(-1, 1)
            outputs = ds.update(xywhs, cfs, clss, img)
            s = outputs[:, 1:5].reshape(-1, 4)

            if len(s) > 0:
                roi = np.append(roi, s, axis=0)
                roi = roi[-NUM_FRAME:]

        if len(roi) >= NUM_FRAME:
            s = np.mean(roi, axis = 0, dtype=int)
            cv2.rectangle(img, (s[0], s[1]),(s[2], s[3]),(90,0,255),3)


        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.imshow("Video", img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()
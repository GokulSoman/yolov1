# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import cv2
import numpy as np

from utils import non_max_suppression

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    import pdb; pdb.set_trace()

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


# # Image
# image = np.zeros((480,640,3))
# # print(image.shape)
# h, w,_ = image.shape

# bboxes = np.ones((25,5))

# # random number for x1 and x2, y1, y2
# bboxes[:,[0,2]] = np.random.randint(0,h,(25,2))
# bboxes[:,[1,3]] = np.random.randint(0,w,(25,2))

# # making proper boxes
# indices = bboxes[:,[0,2]].argsort(axis=1)
# bboxes[:,0] , bboxes[:,2] = np.min(bboxes[:,[0,2]], axis=1), np.max(bboxes[:,[0,2]], axis=1)
# bboxes[:,1] , bboxes[:,3] = np.min(bboxes[:,[1,3]], axis=1), np.max(bboxes[:,[1,3]], axis=1)

# # make prob. values

# bboxes[:,4] = np.random.rand(len(bboxes))

# # import pdb; pdb.set_trace()


# check_im = image.copy()
# for box in bboxes:
#     rect = box.astype(int)
#     cv2.rectangle(check_im, (rect[0], rect[1]), (rect[2], rect[3]), (125,125,125), thickness=2)
# cv2.imshow("Before NMS", check_im)
# cv2.waitKey()

# # import pdb; pdb.set_trace()

# import pdb; pdb.set_trace()
# print("Before Nms", bboxes.shape)
# bboxes = np.concatenate((np.ones((len(bboxes),1)), bboxes), axis=1)
# bboxes = non_max_suppression(bboxes, 0.3, 0.4)
# # bboxes = bboxes[keep]

# print("After NMS : ", bboxes.shape)
# for box in bboxes:
#     rect = box[-4:].astype(int)
#     cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (125,125,125), thickness=2)


# cv2.imshow("After NMS", image)
# cv2.waitKey()

##---------------------------FOr custom nms

# Image
image = np.zeros((480,640,3))
# print(image.shape)
h, w,_ = image.shape

bboxes = np.ones((25,6))

# random number for x1 and x2, y1, y2
bboxes[:,[-4,-2]] = np.random.randint(0,h,(25,2))
bboxes[:,[-3,-1]] = np.random.randint(0,w,(25,2))

# making proper boxes
indices = bboxes[:,[-4,-2]].argsort(axis=1)
bboxes[:,-4] , bboxes[:,-2] = np.min(bboxes[:,[-4,-2]], axis=1), np.max(bboxes[:,[-4,-2]], axis=1)
bboxes[:,-3] , bboxes[:,-1] = np.min(bboxes[:,[-3,-1]], axis=1), np.max(bboxes[:,[-3,-1]], axis=1)

# make prob. values

bboxes[:,1] = np.random.rand(len(bboxes))

# import pdb; pdb.set_trace()


check_im = image.copy()
for box in bboxes:
    rect = box[-4:].astype(int)
    cv2.rectangle(check_im, (rect[0], rect[1]), (rect[2], rect[3]), (125,125,125), thickness=2)
cv2.imshow("Before NMS", check_im)
cv2.waitKey()

# import pdb; pdb.set_trace()

import pdb; pdb.set_trace()
print("Before Nms", bboxes.shape)
# bboxes = np.concatenate((np.ones((len(bboxes),1)), bboxes), axis=1)
bboxes = non_max_suppression(bboxes, 0.3, 0.4)
# bboxes = bboxes[keep]

print("After NMS : ", bboxes.shape)
for box in bboxes:
    rect = box[-4:].astype(int)
    cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (125,125,125), thickness=2)


cv2.imshow("After NMS", image)
cv2.waitKey()



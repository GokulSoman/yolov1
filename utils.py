import torch
import numpy as np

def intersection_over_union(boxes_1, boxes_2, box_format="midpoint"):
    """
    Both boxes are of shape (...,4)
    if midpoint, the input values should be (x_mid, y_mid, width, height)
    """
    if "midpoint":
        box1_x1 = boxes_1[...,0:1] - boxes_1[...,2:3] / 2
        box1_y1 = boxes_1[...,1:2] - boxes_1[...,3:4] / 2
        box1_x2 = boxes_1[...,0:1] + boxes_1[...,2:3] / 2
        box1_y2 = boxes_1[...,1:2] + boxes_1[...,3:4] / 2
        box2_x1 = boxes_2[...,0:1] - boxes_2[...,2:3] / 2
        box2_y1 = boxes_2[...,1:2] - boxes_2[...,3:4] / 2
        box2_x2 = boxes_2[...,0:1] + boxes_2[...,2:3] / 2
        box2_y2 = boxes_2[...,1:2] + boxes_2[...,3:4] / 2
    
    # Intersection box coordinates

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # clamps the value to 0 if the value goes negative
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    union = box1_area + box2_area - intersection + 1e-6

    intersection_over_union = intersection / union

    return intersection_over_union


def non_max_suppression(bboxes, iou_threshold, min_score, box_format="corners"):

    """
    bboxes -> list of bounding boxes having values
        [class_index, prob_score, x1,y1, x2, y2]
        Will be in shape (N,6) (if performed for batches) 
    iou_threshold -> iou to perform nms
    min_score -> minimum probability score to consider box for nms
    box_format -> whether the boxes are in format corners, midpoint, etc.

    #NMS concept
    1. filter the boxes per class
    2. for a class, first keep the highest confidence box
    3. for the same class, remove the boxes having iou > threshold
    4. From the boxes left, again keep the highest confidence box
    5. Repeat till no boxes are left to be considered
    """

    # Reduce the boxes based on minimum score

    bboxes = np.array(bboxes) if type(bboxes) is list else bboxes
    bboxes = bboxes[bboxes[:,1] > min_score]
    
    # Perform NMS

    # Filter by class
    indices = np.unique(bboxes[:,0])

    out_boxes = np.empty(0)
    for index in indices:
        filtered_bboxes = bboxes[bboxes[:,0] == index,1:]
        scores = filtered_bboxes[:,0]
        order = scores.argsort()[::-1]

        keep = []
        
        x1 = filtered_bboxes[:,1]
        y1 = filtered_bboxes[:,2]
        x2 = filtered_bboxes[:,3]
        y2 = filtered_bboxes[:,4]

        areas = (x2 - x1) * (y2 - y1)

        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(filtered_bboxes[i,1], filtered_bboxes[order[1:],1])
            yy1 = np.maximum(filtered_bboxes[i,2], filtered_bboxes[order[1:],2])
            xx2 = np.minimum(filtered_bboxes[i,3], filtered_bboxes[order[1:],3])
            yy2 = np.minimum(filtered_bboxes[i,4], filtered_bboxes[order[1:],4])

            w = np.maximum(0, (xx2 - xx1))
            h = np.maximum(0, (yy2 - yy1))

            inter = w*h

            # import pdb; pdb.set_trace()
            ovr = inter / ( areas[i] + areas[order[1:]] - inter )

            order = order[1:][ovr <= iou_threshold]

        if len(keep) >= 1:
            class_indices = np.ones((len(keep), 1)) * float(index)
            # import pdb; pdb.set_trace()

            out_boxes_per_index = np.concatenate((class_indices, filtered_bboxes[keep]), axis=1)

            # concatenate if out_boxes are present, else
            out_boxes = np.concatenate((out_boxes, out_boxes_per_index), axis=0) if len(out_boxes)>0 else out_boxes_per_index

    return out_boxes

# convert from label matrix to bounding box format 
def predictions_to_bboxes(predictions, S = 7, C = 20, B=2, obj_exists=0.8, class_threshold=0.75):
    # predictions in shape (N,S,S,C+B*5)
    # convert to list of bounding boxes with format [class_index, prob_score, x1,y1,x2,y2]
    #batch_size = predictions.shape[0]
    predictions = predictions.detach().view(S, S, C + B*5)

    # convert from cell location to image location
    locs_x = torch.arange(S).repeat(S,1).view(S, S)
    locs_y = locs_x.transpose(1,2)

    predictions[..., 21] = (predictions[..., 21] + locs_x) / S
    predictions[..., 26] = (predictions[..., 26] + locs_x) / S

    predictions[..., 22] = (predictions[..., 22] + locs_y) / S
    predictions[..., 27] = (predictions[..., 27] + locs_y) / S

    best_box = predictions[..., [20,25]].argmax(dim=-1)
    
    # make preds into an array of boxes
    box1_args = [i for i in range(25)]
    box2_args = [i for i in range(20)] + [25 + i for i in range(5)]
    #TODO: Choose boxes? or take all preds
    predictions = torch.cat((predictions[best_box == 0][..., box1_args], 
                             predictions[best_box==1][..., box2_args])) # (S*S, 25)


    # filter top predictiosn based on object exists
    predictions = predictions[predictions[...,20] > obj_exists] # shape k,25

    # filter based prob of obj provided box exists
    class_probs_best = predictions[..., :C].max(dim=-1) # shape (k)
    prob_class = predictions[...,20] * class_probs_best # k

    predictions = predictions[prob_class > class_threshold].view(-1, C + 5) # y,25

    #convert predicitons to bbox
    # class_id, xmin, ymin, w, h, score

    bboxes = torch.zeros(predictions.shape[0], 6) # y,6

    bboxes[...,0] = predictions[..., :C].argmax(dim=-1)
    bboxes[...,-1] = prob_class[prob_class > class_threshold]
    bboxes[...,1:-1] = predictions[..., 21:25] # taking the best box, TODO: make suitable for more than 2 boxes

    
    #bboxes = []
    #for i in range(batch_size):
    #    bboxes.append(non_max_suppression(predictions[i], iou_threshold, threshold))
    
    return bboxes



if __name__ == "__main__":
    a = torch.tensor(2.5 * np.ones((3,4)))
    a[:,2:] = 5
    b = torch.tensor(6 * np.ones((3,4)))
    b[:,2:] = 4
    print("coordinates as x_mid, y_mid, w,h")
    print(f"a_coordinates check --> {a[0,0]}")
    print(f"b_coordinates check --> {b[0,0]}")
    c = intersection_over_union(a,b)
    # answer should be 1/ (25 + 16 - 1)
    print(f"Answer should be {torch.tensor(1/ (25 + 16 - 1))}")
    print(f"Answer: {c[0,0]}")
    print(f"Answer is matching: {torch.isclose(torch.tensor(1/ (25 + 16 - 1), dtype=c.dtype),  c[0,0])}")

    # Testing NMS
    nms_bboxes = np.ones((5,6)) * 5
    nms_bboxes[:, 1] = (0.33,0.31,0.04, 0.32,0.32)
    nms_bboxes[1,2] = 4
    nms_bboxes[3,2] = 0
    nms_bboxes[:,-2] = nms_bboxes[:,-4] + 3
    nms_bboxes[:,-1] = nms_bboxes[:,-3] + 2
    nms_bboxes[-1] = nms_bboxes[3]
    nms_bboxes[-1,0] = 4


 
    print(f"NMS boxes shape: {nms_bboxes.shape}")
    print(f"NMS boxes input: {nms_bboxes}")
    ans = non_max_suppression(nms_bboxes, .3 ,.3)
    print(f"Answer: {ans.shape}")
    print(f"Answer content: {ans}")

    

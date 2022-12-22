import torch

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
     
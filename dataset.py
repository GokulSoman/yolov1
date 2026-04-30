import pandas as pd
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import os.path as osp
from random import randint
from torchvision.transforms import transforms, v2, ToPILImage, ToTensor
from torchvision.io import read_image, decode_image
import matplotlib.pyplot as plt

test = 0

import torchvision.transforms.functional as F


cmap = plt.get_cmap("hsv")  # or "hsv", "viridis"

def class_to_color(cls_id, num_classes):
    color = cmap(cls_id / num_classes)
    return tuple(int(c * 255) for c in color[:3])

def to_linear(c):
    return c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4

def light_black(color):
    text_color = (255,255,255)
    # normalize
    r,g,b = [x/255.0 for x in color]
    # linearize to srgb
    r,g,b = [to_linear(x) for x in [r,g,b]]
    # find luminance
    luminance = (0.2126 * r + 0.7152  *g + 0.0722 * b)
    if luminance > 0.5:
        # make text dark
        text_color = (0,0,0)
    return text_color
        


def draw_pil_boxes(img, boxes, labels, class_dict):
    w,h = img.size
    draw = ImageDraw.Draw(img)

    # Optional: load a better font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    for box, cls in zip(boxes, labels):
        x1, y1, x2, y2 = box
        label = f"{cls}"

        color = class_to_color(class_dict[cls], num_classes=20)

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Get text size
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        t_x1, t_y1, t_x2, t_y2 = x1, y1 - text_h, x1+ text_w, y1

        # clip extremes
        if y1 - text_h < 0:
            # move box downwards
            print("move box downwards")
            t_y1 = 0
            t_y2 = text_h
        if x1 + text_h > (w-1):
            # move left
            print("move left")
            t_x2 = w - 1
            t_x1 = (w-1) - text_w
            
        text_dims = [t_x1, t_y1, t_x2, t_y2]

        
        # Draw background rectangle (for readability)
        draw.rectangle(
            text_dims,
            fill=color
        )

        # Draw text
        draw.text((t_x1, t_y1), label, fill=light_black(color), font=font)

    return img

class Letterbox:
    def __init__(self, size, fill=0):
        self.target_h, self.target_w = size
        self.fill = fill  # padding color (0 = black)

    def __call__(self, img, gt_bboxes, scale=None, return_params=False):

        # messy fix
        if "PIL" in str(type(img)):
            # original size
            w, h = img.size

        else:
            # torchvision image
            h,w = img.shape[-2:]

        if not isinstance(gt_bboxes, torch.Tensor):
            gt_bboxes = torch.tensor(gt_bboxes)

        assert len(gt_bboxes.shape) == 2, f"Target should be of shape (N,5), where N is num of bboxes, got {gt_bboxes.shape}"
        
        # # bounding boxes (not img)
        # boxes = img # shape (N,5), 5 vals are class, xmid, ymid, w,h (normalized)

        # assert scale is not None, "Scale has to be provided for bbox transforms"
        # assert type(scale) == float, "Scale should be provided as a float"

        # boxes[:,1:] *= scale

        if scale is None: 
            # compute scale (preserve aspect ratio)
            scale = min(self.target_w / w, self.target_h / h)
        
        new_w, new_h = int(w * scale), int(h * scale)

        # resize
        img = F.resize(img, (new_h, new_w))
        # import pdb; pdb.set_trace()
        gt_bboxes[:,1:] *= torch.tensor([new_w, new_h, new_w, new_h]).view(1,-1)

        # compute padding
        pad_w = self.target_w - new_w
        pad_h = self.target_h - new_h

        assert pad_w >= 0, f"Negative padding not allowed: {pad_w=}"
        assert pad_h >= 0, f"Negative padding not allowed: {pad_h=}"

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        # apply padding
        img = F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)

        #import pdb; pdb.set_trace()
        # pad bboxes
        gt_bboxes[:,1] += pad_left
        gt_bboxes[:,2] += pad_top

        # renormalize
        gt_bboxes[:,[1,3]] /= self.target_w
        gt_bboxes[:,[2,4]] /= self.target_h

        return img, gt_bboxes
    
class CustomToTensor:
    def __call__(self, img, gt_bboxes):
        if isinstance(img, Image.Image):
            img = ToTensor()(img)
        if not isinstance(gt_bboxes, torch.Tensor):
            gt_bboxes = torch.tensor(gt_bboxes)
        return img, gt_bboxes

class PascalVOC(torch.utils.data.Dataset):
    def __init__(self, csv_file, image_dir, label_dir, classes_file, grids=7,  box_per_cell=2, classes=20, pil_read=True, inp_size=448, debug=False) -> None:
        self.annotations = pd.read_csv(csv_file, header=None)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.S = grids
        self.B = box_per_cell
        self.C = classes
        self.inp_size = inp_size
        self.letterbox_t = Letterbox((self.inp_size,self.inp_size))
        self.transforms = [
            self.letterbox_t,
            CustomToTensor(),
        ]
        self.pil_read = pil_read
        # if not self.pil_read:
        #     self.transforms.pop()
        self.transforms = v2.Compose(self.transforms)
        # self.bbox_transforms = self.compose_bbox_transforms_(self.transforms)
        self.debug = debug
        self.classes_file = classes_file
        with open(self.classes_file,'r') as cf:
            self.classes = cf.readlines()
        self.classes = [x.strip() for x in self.classes]
        self.id2class = {i:x for i,x in enumerate(self.classes)}
        self.class2id = {x:i for i,x in self.id2class.items()}
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = osp.join(self.label_dir,self.annotations.iloc[index,1])
        bboxes = []
        with open(label_path, 'r') as annotations_file:
            for annotation in annotations_file.readlines():
                class_index, x_mid_r, y_mid_r, width_r, height_r = [float(value) if float(value)!= int(float(value)) else int(value) for value in annotation.split()]
                bboxes.append([class_index, x_mid_r, y_mid_r, width_r, height_r])

        if self.pil_read:
            image = Image.open(osp.join(self.image_dir, self.annotations.iloc[index,0]))

        else:
            #torchvision read -> should be faster
            image = decode_image(osp.join(self.image_dir, self.annotations.iloc[index,0]))/255.0
            # img_h, img_w = image.shape[-2:]

        #import pdb; pdb.set_trace()
        
        image, bboxes = self.transforms(image, bboxes)

        if self.debug:
            return image, bboxes

        # create label matrix
    
        # The label is provided as a matrix, with each cell consisting of the 20 class scores, presence of an object, and the bounding box values
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 ))


        # (boxes, 2) where columns are x,y indices
        grid_indices = (bboxes[:,1:3] * self.S).int()
        # if center ever equals image extreme edge (1)
        grid_indices[grid_indices==self.S] -= 1



        xy_cell = bboxes[:,1:3] * self.S - grid_indices

        # NP Indexing
        
        #TODO: check 4 steps, esp bboxes
        # col, row -> row, col
        label_matrix[grid_indices[:,1], grid_indices[:,0], 20] = 1
        # label_matrix[grid_indices[:,0], grid_indices[:,1],20] = 1
        
        label_matrix[grid_indices[:,1], grid_indices[:,0],21:23] = xy_cell
        label_matrix[grid_indices[:,1], grid_indices[:, 0 ],23:] = bboxes[:,-2:]

        # class labels are 0-19, set element to 1 if index == label
        label_matrix[grid_indices[:,1], grid_indices[:,0], bboxes[:,0].int()] = 1

        return image, label_matrix

    # def test_annotations(self, index = None):
    #     index = randint(0, len(self.annotations)) if index is None else index
    #     image = Image.open(osp.join(self.image_dir, self.annotations.iloc[index,0])) .convert("RGB")
    #     image_width, image_height = image.size
    #     label_path = osp.join(self.label_dir, self.annotations.iloc[index, 1])
    #     # #import pdb; pdb.set_trace()
    #     with open(label_path, 'r') as annotations_file:
    #         for annotation in annotations_file.readlines():
    #             # var_r shows that it is a ratio w.r.t image.
    #             class_index, x_mid_r, y_mid_r, width_r, height_r = [float(value) if float(value) != int(float(value)) else int(value) for value in annotation.split()]
    #             # Shape to PIL given in form [(x0, y0),(x1,y1)]
    #             x_mid, y_mid = x_mid_r * image_width, y_mid_r * image_height
    #             width, height = width_r * image_width, height_r * image_height
    #             x0,y0 = int(x_mid - (width / 2)), int(y_mid - (height/2))
    #             x1,y1 = int(x_mid + (width / 2)), int(y_mid + (height/2))

    #             # make pixels fit image
    #             x0, y0 = max(0,x0), max(0,y0)
    #             x1, y1 = min(image_width, x1), min(image_height, y1)

    #             box = [(x0,y0), (x1,y1)]
    #             draw = ImageDraw.Draw(image)
    #             draw.rectangle(box, outline="yellow")
    #     # #import pdb; pdb.set_trace()
    #     image.show()

    def test_sample(self, index=None):
        assert self.debug is True, "test sample works only in debug mode"
        if index is None:
            index = randint(0, len(self) - 1)
        image, boxes = self[index]

        assert len(image[image > 1]) == 0, "Values > 1 in normalized image is not allowed"
        assert len(image[image < 0]) == 0, "Negative values in normalized image is not allowed"
        # remove normalization
        # image = (image * 255).int()
        image = ToPILImage()(image) # back to 0-255
        w,h = image.size

        assert w==h==self.inp_size, "Values are different from model input"
        boxes[:,1:] *= w

        # #import pdb; pdb.set_trace()

        # boxes[:,3:] += boxes[:,1:2] #x,y,w,h -> xmin, ymin, xmax, ymax
        #  move labels from grid to image
        # out should be b,5 , where n is num_boxes
        # 5 values are x_mid, y_mid, w, h
        # boxes = torch.zeros( (len(label[label[..., 20] == 1]), 5))
        # box_id = 0
        # for i in range(self.S):
        #     for j in range(self.S):
        #         if label[i,j,20] == 0:
        #             # no box here
        #             continue
        #         # class
        #         boxes[box_id][0] = label[i,j,:20].flatten().argmax()
        #         boxes[box_id][1:] = label[i,j,21:]
        #         boxes[box_id][1] = (boxes[box_id][1] + j) / self.S
        #         boxes[box_id][2] = (boxes[box_id][2] + i) / self.S

        #         box_id += 1
        
        # # convert to xmin, ymin, x_max, y_max
        # #import pdb; pdb.set_trace()
        # w_half, h_half = boxes[:,3], boxes[:,4]
        # boxes[:,1] -= w_half
        # boxes[:,2] -= h_half
        # boxes[:,3] += w_half
        # boxes[:,4] += h_half

        # # scale to image width and height

        # boxes[:,[1,3]] *= w
        # boxes[:,[2,4]] *= h

        # # add padding
        # pad_left, pad_top = padding
        # boxes[:[1,3]] += pad_left
        # boxes[:,[2,4]] += pad_top

        classes = boxes[:,0].int().tolist()
        labels = [self.id2class[x] for x in classes]
        w_half, h_half = boxes[:,3]/2, boxes[:,4]/2
        
        boxes[:,3] = boxes[:,1] + w_half # xmax = xmid + w/2
        boxes[:,4] = boxes[:,2] + h_half # ymax = ymid + h/2
        boxes[:,1] -= w_half # xmin = xmid - w/2
        boxes[:,2] -= h_half #ymin = ymid - h/2

        boxes = boxes[:,1:].int().clip(min=0,max=w-1).tolist() # 0-447
        
        # pil_boxes = [((xmin, ymin),(xmax,ymax)) for (xmin,ymin,xmax,ymax) in boxes]
        # draw = ImageDraw.Draw(image)
        
        # for id,box in zip(classes,boxes):
        #     color = class_to_color(id, 20) # for pascal_voc
        #     draw.rectangle(box, outline=color, width=2)
        # print(image.mode, image.size)

        image = draw_pil_boxes(image, boxes, labels, self.class2id)
        # image.show()
        plt.figure(f"{index}") 
        plt.imshow(image)
        plt.axis("off")
        plt.show()
        return image, boxes
if __name__ == "__main__":
    dataset_dir = "/home/gokul/datasets/pascal_voc"
    csv_file = osp.join(dataset_dir, "train.csv")
    image_dir = osp.join(dataset_dir, "images")
    label_dir = osp.join(dataset_dir, "labels")
    classes_file = osp.join(dataset_dir, "classes.txt")
    dataset = PascalVOC(csv_file, image_dir, label_dir,classes_file, debug=True)
    # dataset.test_annotations(index=None)

    # #import pdb; pdb.set_trace()
    a,b = dataset.test_sample()
    #import pdb; pdb.set_trace()

                




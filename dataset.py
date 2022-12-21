import pandas as pd
import numpy as np
import torch
from PIL import Image, ImageDraw
import os.path as osp
from random import randint

test = 0

class PascalVOC(torch.utils.data.Dataset):
    def __init__(self, csv_file, image_dir, label_dir, grids=7,  box_per_cell=2, classes=20, transforms=None) -> None:
        self.annotations = pd.read_csv(csv_file, header=None)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.S = grids
        self.B = box_per_cell
        self.C = classes
        self.transforms = transforms
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = osp.join(self.label_dir,self.annotations.iloc[index,1])
        boxes = []
        with open(label_path, 'r') as annotations_file:
            for annotation in annotations_file.readlines():
                class_index, x_mid_r, y_mid_r, width_r, height_r = [float(value) if float(value)!= int(float(value)) else int(value) for value in annotation.split()]
                boxes.append([class_index, x_mid_r, y_mid_r, width_r, height_r])
                if test == 1:
                    s_values = np.linspace(0,1,self.S, endpoint=False)
                    test_indices = []
                    test_i, test_j = 0, 0
                    for n in range(len(s_values)):
                        test_j = n if x_mid_r > s_values[n] else test_j
                        test_i = n if y_mid_r > s_values[n] else test_i
                    test_indices.append((test_i, test_j))
                    print(test_indices)
        image = Image.open(osp.join(self.image_dir, self.annotations.iloc[index,0]))
        # image_width, image_height = image.size #Not required
        orig_values = np.array(boxes)

        if self.transforms:
            orig_values = torch.tensor(orig_values)
            image, orig_values = self.transforms(image, orig_values)
            orig_values.detach().cpu().numpy()

        # boxes = orig_values.copy()
        # # import pdb; pdb.set_trace()
        
        # # saving classes as separate array, removing from box values
        # class_indices = boxes[:,0].astype(int)
        # boxes = boxes[:,1:]

        # # scaling values w.r.t image size
        # boxes *= np.array([image_width, image_height, image_width, image_height])
        # # boxes[:,[0,2]] *= np.array([image_width, image_height, image_width, image_height])

        # # boxes[:,[1,3]] *= image_height

        # # changing values from x_mid,y_mid, w,h to x0, y0, x1, y1

        # boxes[:,:2] -= boxes[:,2:]/2
        # boxes[:,2:] += boxes[:,:2]

        # # changing all values to int
        # boxes.astype(int)


        # if test == 1 :
        # image2 = Image.open(osp.join(self.image_dir, self.annotations.iloc[index,0])).convert("RGB")
        # image_draw = ImageDraw.Draw(image2)
        # boxes = boxes.tolist()
        # for box in boxes:
        #     image_draw.rectangle(box, outline="yellow")
        # image2.show()


        
    
        # The label is provided as a matrix, with each cell consisting of the 20 class scores, presence of an object, and the bounding box values
        label_matrix = np.zeros((self.S, self.S, self.C + 5 ))

        # eq 1: grid_index = floor( x_mid / grid_size ) as integer
        # eq 2: grid_size = image_width/ total_no_of_grids
        # eq 3: x_mid_r = x_mid / image_width
        # Sub 2 and 3 in 1 to get grid_index = floor( x_mid_r * total_no_of_grids) -eq(4)

        # Now in shape column, row
        grid_indices = orig_values[:,1:3] * self.S


        # TO find x_mid w,r,t its corresponding cell
        # eq 5: x_cell = (x_mid - grid_index * grid_size) / grid_size
        # eq 6: x_mid = x_mid_r * image_width
        # eq 7 : grid_size = image_width / no.of_grids
        # Sub 6 and 7 in eq 5, x_cell = x_mid_r * no.of.grids - grid_index eq (8)

        # Indices changed to Row, column
        grid_indices = grid_indices[:,[1,0]].astype(int)

        xy_cell = orig_values[:,1:3] * self.S - grid_indices[:,[1,0]]
        # print(xy_cell)

        # import pdb;pdb.set_trace()

        # NP Indexing


        label_matrix[grid_indices[:,0], grid_indices[:,1],20] = 1
        label_matrix[grid_indices[:,0], grid_indices[:,1],21:23] = xy_cell
        label_matrix[grid_indices[:,0], grid_indices[:,1],23:] = orig_values[:,-2:]
        label_matrix[grid_indices[:,0], grid_indices[:,1], orig_values[:,0].astype(int) - 1] = 1

        # for value in orig_values:
        #     i, j = int(self.S * )
        #     grid_indices.append((i,j))
        #     label_matrix[i,j,20] = 1
        #     label_matrix[i,j, index - 1] = 1
        #     label_matrix[i,j, 21:] = torch.Tensor(box)
        # print(grid_indices)            
        # print(label_matrix)
        label_matrix = torch.tensor(label_matrix)
        return image, label_matrix

    def test_annotations(self, index = None):
        index = randint(0, len(self.annotations)) if index is None else index
        image = Image.open(osp.join(self.image_dir, self.annotations.iloc[index,0])) .convert("RGB")
        image_width, image_height = image.size
        label_path = osp.join(self.label_dir, self.annotations.iloc[index, 1])
        # import pdb; pdb.set_trace()
        with open(label_path, 'r') as annotations_file:
            for annotation in annotations_file.readlines():
                # var_r shows that it is a ratio w.r.t image.
                class_index, x_mid_r, y_mid_r, width_r, height_r = [float(value) if float(value) != int(float(value)) else int(value) for value in annotation.split()]
                # Shape to PIL given in form [(x0, y0),(x1,y1)]
                x_mid, y_mid = x_mid_r * image_width, y_mid_r * image_height
                width, height = width_r * image_width, height_r * image_height
                x0,y0 = int(x_mid - (width / 2)), int(y_mid - (height/2))
                x1,y1 = int(x_mid + (width / 2)), int(y_mid + (height/2))

                box = [(x0,y0), (x1,y1)]
                draw = ImageDraw.Draw(image)
                draw.rectangle(box, outline="yellow")
        # import pdb; pdb.set_trace()
        image.show()

if __name__ == "__main__":
    dataset_dir = "/storage/datasets/pascal_voc"
    csv_file = osp.join(dataset_dir, "train.csv")
    image_dir = osp.join(dataset_dir, "images")
    label_dir = osp.join(dataset_dir, "labels")
    dataset = PascalVOC(csv_file, image_dir, label_dir)
    # dataset.test_annotations(index=1)

    a,b = dataset[0]

                




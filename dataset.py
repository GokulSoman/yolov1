import pandas as pd
import torch
from PIL import Image, ImageDraw
import os.path as osp
from random import randint

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
                class_index, x, y, width, height = [float(value) if float(value)!= int(value) else int(value) for value in annotation.split()]
                boxes.append([class_index, x, y, width, height])
        
        # The label is provided as a matrix, with each cell consisting of the 20 class scores, presence of an object, and the bounding box values
        label_matrix = torch.zeros((self.S, self.S, self.C + 5))

        image_width, image_height = None, None

        pass

    def test_annotations(self, index = None):
        index = randint(0, len(self.annotations)) if index is None else index
        image = Image.open(osp.join(self.image_dir, self.annotations.iloc[index,0])) .convert("RGB")
        image_width, image_height = image.size
        label_path = osp.join(self.label_dir, self.annotations.iloc[index, 1])
        with open(label_path, 'r') as annotations_file:
            for annotation in annotations_file.readlines():
                class_index, x, y, width, height = [float(value) if float(value) != int(float(value)) else int(value) for value in annotation.split()]
                box = [(int(x * image_width), int(y * image_height)), (int(width * image_width), int(height * image_height))]
                draw = ImageDraw.Draw(image)
                draw.rectangle(box, outline="yellow")
        image.show()

if __name__ == "__main__":
    dataset_dir = "/media/av/DATA/datasets/PASCAL_VOC"
    csv_file = osp.join(dataset_dir, "train.csv")
    image_dir = osp.join(dataset_dir, "images")
    label_dir = osp.join(dataset_dir, "labels")
    dataset = PascalVOC(csv_file, image_dir, label_dir)
    dataset.test_annotations(index=0)
        


                




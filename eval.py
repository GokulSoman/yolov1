import os.path as osp

import torch

from torch.utils.data import DataLoader
from dataset import PascalVOC
from model import YoloV1
# from loss import YoloV1Loss
from utils import predictions_to_bboxes

device="cuda"

dataset_dir = "/home/gokul/datasets/pascal_voc"
img_dir = osp.join(dataset_dir, "images")
label_dir = osp.join(dataset_dir, "labels")
train_csv = osp.join(dataset_dir, "train.csv")
test_csv = osp.join(dataset_dir, "test.csv")
classes_file = osp.join(dataset_dir, "classes.txt")


if __name__ == "__main__":

    # load data
    test_data = PascalVOC(csv_file=test_csv, image_dir=img_dir, 
                          label_dir=label_dir, classes_file=classes_file, pil_read=False)
    
    # dataloader

    from torch.utils.data import Subset
    sub_indices = [i for i in range(10)]

    test_dl = DataLoader(
                Subset(test_data, sub_indices),
                batch_size=1,
                shuffle=False,
                num_workers=0, 
                pin_memory=True  # faster GPU transfer
            )
    test_total_steps = len(test_dl)


    # load model
    print("Loading model...")
    model = YoloV1(split_size=7, num_boxes=2, num_classes=20).to(device)
    # compile - DISABLED FOR DEBUGGING
    model = torch.compile(model)

    print("Loading checkpoint...")
    model_dict = torch.load("model_e139_loss_5_979.pth", map_location=device)

    print("Loading state dict...")
    model.load_state_dict(model_dict["model_state"])
    # compile model

    # model = torch.compile(model)

    print("Setting to eval mode...")
    model.eval()
    count = 0
    from tqdm import tqdm
    with torch.no_grad():
        for batch_x, _ in tqdm(test_dl):
            batch_x = batch_x.to(device, non_blocking=True)
            # batch_y = batch_y.to(device, non_blocking=True)
            # with torch.autocast(device_type=device, dtype=torch.bfloat16):
            out = model(batch_x) 
            for i in range(batch_x.size(0)):
                bboxes = predictions_to_bboxes(out[i].cpu(), obj_exists=0.1, class_threshold=0.1)
                print(f"\n[Image {count}] Detected {len(bboxes)} bboxes")
                if len(bboxes) > 0:
                    print(f"  Bbox shape: {bboxes.shape}")
                    print(f"  First bbox: {bboxes[0]}")
                else:
                    print("  WARNING: No bboxes detected!")
                img = batch_x[i].cpu()
                try:
                    img, bboxes = test_data.test_sample(img_bbox=(img, bboxes), show_image=False)
                    img.save(f"eval_results/{count:06}.jpg")
                    print(f"  Saved to eval_results/{count:06}.jpg")
                except Exception as e:
                    print(f"  ERROR in test_sample: {e}")
                    import traceback
                    traceback.print_exc()
                count += 1
                if count >= 5:  # Just test on first 5 images
                    break
        print(f"\nProcessed {count} images total")
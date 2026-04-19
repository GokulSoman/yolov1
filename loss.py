import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloV1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):

        import pdb;pdb.set_trace()
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        #TODO: make suitable for more than 2 boxes
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        # There exists only one target, hence the same indices for target
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        
        ious = torch.cat([iou_b1, iou_b2], dim=-1)

        _, best_box = torch.max(ious, dim=-1, keepdim=True)

        # unsqueeze is done to keep dimensions similar
        exists_box = target[..., 20].unsqueeze(3) # identity of obj i (0 or 1)
        #TODO: use unsqueeze -1
        #TODO: assert dimension is (Batch, S, S, 1)

        ## For Box Coordinates
        #TODO: make suitable for more than 2 boxes
        box_predictions_initial = exists_box * (
            (1 - best_box) * predictions[..., 21:25]
            + best_box * predictions[..., 26:30]
        )


        box_targets_ini = exists_box * target[..., 21:25]

        # Taking root of w,h

        box_predictions_xy = box_predictions_initial[..., :2]
        box_targets_xy = box_targets_ini[..., :2]
        #TODO: Do not know why 1e-6 is added (machine precision??))
        # Predictions can be negative, hence the extra steps
        box_predictions_wh = torch.sign(box_predictions_initial[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions_initial[...,2:4]) + 1e-6)
        
        # Combine box preds
        box_predictions = torch.cat((box_predictions_xy, box_predictions_wh), dim=-1)

        # Target w,h always positive (ground truth)
        box_targets_wh = torch.sqrt(box_targets_ini[..., 2:4])

        box_targets = torch.cat((box_targets_xy, box_targets_wh), dim=-1)

        #TODO: check whether shape change is actualy required?
        # Shape change from (N, S, S, 4) --> (N * S * S, 4)
        box_loss = self.mse(torch.flatten(box_predictions, end_dim=-2),
                            torch.flatten(box_targets, end_dim=-2))

        ## For the object predictions

        # predictions in shape (N,S,S,1)
        # these are the best predictions
        object_predictions = exists_box * (
            (1- best_box) * predictions[..., 20].unsqueeze(-1) +
            best_box * predictions[..., 25].unsqueeze(-1)
        )

        # targets in shape (N, S,S,1) not  changed\
        object_loss = self.mse(torch.flatten(object_predictions),
                    torch.flatten(exists_box * target[...,[20]])
        )

        ## Loss if no object is present

        #TODO: target[...,20] is 0 if no object
        no_object_loss = self.mse(torch.flatten((1-exists_box) * predictions[..., [20]]),
                    torch.flatten((1-exists_box) * target[..., [20]]))
        
        no_object_loss += self.mse(torch.flatten((1-exists_box) * predictions[..., [25]]),
                    torch.flatten((1-exists_box) * target[..., [20]]))
        
        # only for best box
        # no_object_loss = self.mse((1-exists_box) * object_predictions, 
        #                           (1-exists_box) * target[..., [20]])
        # no_object_loss += self.mse((1-exists_box) * predictions[...,[25]], 
        #                           (1-exists_box) * target[..., [20]])


        ## Loss for the predicted class classes (classification loss)
        # from (N,S,S,20) -- > (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim = -2)
        )

        # Sum of Losses

        loss = (self.lambda_coord * box_loss
        + object_loss
        + self.lambda_noobj * no_object_loss
        + class_loss)

        return loss
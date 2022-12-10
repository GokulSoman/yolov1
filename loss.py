import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloV1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        # There exists only one target, hence the same indices for target
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        iou_maxes, best_box = torch.max(ious, dim=0)

        # unsqueeze is done to keep dimensions similar
        exists_box = target[..., 20].unsqueeze(3) # identity of obj i (0 or 1)
        

        ## For Box Coordinates

        box_predictions = exists_box * (
            (1 - best_box) * predictions[..., 21:25]
            + best_box * predictions[..., 26:30]
        )


        box_targets = exists_box * target[..., 21:25]

        # Taking root of w,h

        #TODO: Do not know why 1e-6 is added (machine precision))
        # Predictions can be negative, hence the extra steps
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[...,2:4] + 1e-6))
        # Target w,h always positive (ground truth)
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # Shape change from (N, S, S, 4) --> (N * S * S, 4)
        box_loss = self.mse(torch.flatten(box_predictions, end_dim=-2),
                            torch.flatten(box_targets, end_dim=-2))

        ## For the object predictions

        # predictions in shape (N,S,S,1)
        object_predictions = exists_box * (
            (1- best_box) * predictions[..., 20].unsqueeze(-1) +
            best_box * predictions[..., 25].unsqueeze(-1)
        )

        # targets in shape (N, S,S,1) -> changed to N*S*S
        object_loss = self.mse(torch.flatten(object_predictions),
                    torch.flatten(exists_box * target[...,20])
        )

        ## Loss if no object is present

        no_object_loss = self.mse(torch.flatten((1-exists_box) * predictions[..., 20]),
                    torch.flatten((1-exists_box) * target[..., 20]))
        
        no_object_loss += self.mse(torch.flatten((1-exists_box) * predictions[..., 25]),
                    torch.flatten((1-exists_box) * target[..., 20]))


        ## Loss for the predicted class classes (classification loss)
        # from (N,S,S,20) -- > (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., 20], end_dim = -2)
        )

        # Sum of Losses

        loss = self.lambda_coord * box_loss
        + object_loss
        + self.lambda_noobj * no_object_loss
        + class_loss

        return loss
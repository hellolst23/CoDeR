import torch
import torch.nn as nn
from util.utils import file_write

class Loss(nn.Module):
    def __init__(self, opt):
        """
        Args:
            opt
        """
        super(Loss, self).__init__()
        self.lambda_catgy = opt.lambda_catgy
        self.catgy_prediction_loss = nn.CrossEntropyLoss()

    def forward(self, l2_loss, result_click, target_item):
        """
        Args:
            l2_loss: torch.Tensor, L2 regularization loss
            result_click: torch.Tensor, batch_size * n_items
            target_item: torch.Tensor, batch_size
        Returns:
            rs_loss
            click_loss
        """
        click_loss = self.click_loss(result_click, target_item)
        rs_loss = click_loss + l2_loss
        return rs_loss, click_loss

    def click_loss(self, result_click, target_item):
        """
        Args:
            result_click: torch.Tensor, batch_size * n_items
            target: torch.Tensor, batch_size
        Returns:
            click_loss: torch.Tensor,  a scalar
        """
        batch_size, _ = result_click.shape
        result_click = torch.softmax(result_click, dim=1)
        pos_score = - torch.log(result_click[torch.arange(batch_size), target_item] + 1e-10)
        click_loss = torch.mean(pos_score)
        return click_loss
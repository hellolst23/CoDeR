# -*- coding: utf-8 -*-
"""
# @File    : loss.py
# Desc:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.utils import file_write



class Loss_Diy(nn.Module):
    """
        compute the  loss of model
    """
    def __init__(self, opt, log_path_txt, sample_stategy='random', beta=1):
        """
        Args:
            log_path_txt: .
            n_negative:
            sample_strategy = 'random'  # type=str 'random', 'category', sample strategy for negative samples
            beta: float, default=1.0, balance the negative sample
        """
        super(Loss_Diy, self).__init__()
        self.beta = beta
        self.n_negative = opt.n_negative
        self.info_lamda = opt.info_lamda
        self.catgy_lamda = opt.catgy_lamda
        self.catgy_prediction_loss = nn.CrossEntropyLoss()

        file_write(log_path_txt, 'There is CrossEntropy Loss')
        if self.n_negative == 0:
            file_write(log_path_txt, '!!! Note: no negative sample')
            self.click_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        else:
            file_write(log_path_txt, f'There is {self.n_negative} negative sample by {sample_stategy} sampling')


    def forward(self, info_loss,l2_loss, result_click, target_item, demand_sim_loss, neg_sample=None, catgy_click=None, target_catgy=None): 
        """
        Args:
            lamda: torch.scalar, control the impact of info_max_loss
            info_loss: torch.Tensor,  a scalar, the info_max loss
            l2_loss: torch.Tensor,  a scalar, the L2 regularization loss
            result_click: torch.Tensor， batch_size * n_items
            target: torch.Tensor, batch_size
            neg_sample: torch.Tensor, batch_size * n_negative, dtype=torch.long

        Returns:
            rs_loss：the whole model loss, torch.Tensor, a scalar
            click_loss: torch.Tensor, a scalar, the click loss
        """

        if self.n_negative == 0:
            #click_loss = self.click_loss_fn(result_click, target)
            click_loss = self.add_neg_loss(result_click, target_item)
        else:
            click_loss = self.add_neg_loss(result_click, target_item, neg_sample)

        # NOTE： demand mask loss(remove info_loss and demand_sim_loss)
        # rs_loss = click_loss + 0 * info_loss + l2_loss + 0 * demand_sim_loss
        # NOTE：remove demand_sim_loss
        # rs_loss = click_loss + lamda * info_loss + l2_loss + 0 * demand_sim_loss
        
        rs_loss = click_loss + self.info_lamda * info_loss + l2_loss  
        # rs_loss = click_loss + self.info_lamda * info_loss + l2_loss + 0 * demand_sim_loss
        if self.catgy_lamda > 0 :
            catgy_loss = self.catgy_lamda * self.catgy_task_loss(catgy_click, target_catgy)
        else:
            catgy_loss = self.catgy_task_loss(catgy_click, target_catgy)



        # print(f"catgy_loss:  {catgy_loss}")
        return rs_loss, click_loss, catgy_loss 


    def add_neg_loss(self, result_click, target_item, neg_sample=None):
        """
        Add negative samples into recommendation loss
        Args:
            result_click: torch.Tensor， batch_size * n_items
            target: torch.Tensor, batch_size, 1维, dtype=torch.long,
            neg_sample: torch.Tensor, batch_size * n_negative, dtype=torch.long
        Returns:
            click_loss: torch.Tensor,  a scalar, the click loss
        """
        batch_size, n_items = result_click.shape
        device = result_click.device

        result_click = torch.softmax(result_click, dim=1) # batch_size * n_items
        pos_score = - torch.log(result_click[torch.arange(batch_size), target_item])  # batch_size

        if neg_sample is None:  # model test
            click_loss = torch.mean(pos_score)
        else:  # model train
            n_negative = neg_sample.shape[1]
            neg_index = torch.arange(0,batch_size).to(device) * n_items
            neg_index = (neg_index.view(-1,1) + neg_sample).flatten()
            neg_score = result_click.take(neg_index).view(-1, n_negative)
            neg_score = - torch.log(1 - neg_score).sum(-1) # batch_size

            click_loss = torch.mean(pos_score + self.beta * neg_score)
        return click_loss

    def catgy_task_loss(self, catgy_click, target_click):
        """
        the category prediction loss
        :param catgy_click: torch.Tensor, batch_size * demand_num * catgy_num
        :param target_click: torch.Tensor, batch_size * catgy_num
        :return: catgy_loss,  a scalar,
        """
        catgy_click = catgy_click.max(1) # the max value between deamnds
        catgy_loss = self.catgy_prediction_loss(catgy_click.values,target_click)
        return catgy_loss


class BPRLoss(nn.Module):
    """
        BPR Loss
    """
    def __init__(self, log_path_txt, sample_stategy):
        super(BPRLoss, self).__init__()

        file_write(log_path_txt, 'There is BPR Loss')
        file_write(log_path_txt, f'There is 1 negative sample by {sample_stategy} sampling')

    def forward(self, lamda, info_loss,l2_loss,  result_click, target, neg_sample=None):
        """
        compute BPR loss, reference: the bpr loss in BGCN

        Args:
            l2_loss: torch.Tensor,  a scalar, the L2 regularization loss
            result_click: torch.Tensor， batch_size * n_items
            target: torch.Tensor, batch_size, dtype=torch.long
            neg_sample: torch.Tensor, batch_size * 1, dtype=torch.long
        Returns:
            click_loss: torch.Tensor,  a scalar, the click loss
        """
        batch_size = result_click.shape[0]

        result_click = torch.softmax(result_click, dim=1)  # batch_size * n_items
        pos_score = result_click[torch.arange(batch_size), target]  # batch_size

        if neg_sample is None:  # model test
            click_loss = torch.mean( - torch.log(pos_score))  
        else:# model train
            neg_score = result_click[torch.arange(batch_size), neg_sample.squeeze(-1)]
            # BPR loss
            #click_loss = torch.mean(-torch.log(torch.sigmoid(pos_score - neg_score)) )# batch_size
            click_loss = torch.mean(F.softplus(neg_score - pos_score))

        rs_loss = click_loss + lamda * info_loss + l2_loss

        return rs_loss, click_loss


if __name__ == "__main__":

    lamda = 0.2
    info_loss = torch.Tensor([3.0])
    result_click = torch.randn(5,10).sigmoid() #batch_size=5, n_items=10
    target = torch.arange(1,6)
    neg_sample = torch.cat((torch.arange(2,7), torch.arange(3,8))).view(-1,2)
    L = Loss_Diy(n_negative=2, beta=1)
    loss = L(lamda, info_loss,result_click, target, neg_sample)
    print(loss)

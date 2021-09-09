import torch
import torch.nn as nn

from ..registry import HEADS


@HEADS.register_module
class ContrastiveHead(nn.Module):
    '''Head for contrastive learning.
    '''

    def __init__(self, temperature=0.1):
        super(ContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, pos, neg):
        '''
        Args:
            pos (Tensor): Nx1 positive similarity
            neg (Tensor): Nxk negative similarity
        '''
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).cuda()
        losses = dict()
        losses['loss'] = self.criterion(logits, labels)
        return losses


@HEADS.register_module
class MultiScaleContrastiveHead(nn.Module):
    def __init__(self, temperature=0.1):
        super(MultiScaleContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, gg_pos, gg_neg, gl_pos, gl_neg, ll_pos, ll_neg):
        N = gg_pos[0].size(0)
        gg_2, gg_3, gg_4, gg_5 = self.get_splitted_logits(gg_pos, gg_neg)
        gl_2, gl_3, gl_4, gl_5 = self.get_splitted_logits(gl_pos, gl_neg)
        ll_2, ll_3, ll_4, ll_5 = self.get_splitted_logits(ll_pos, ll_neg)
        labels = torch.zeros((N,), dtype=torch.long).cuda()
        loss_2 = (self.criterion(gg_2, labels) + self.criterion(gl_2, labels) + self.criterion(ll_2, labels)) / 3.0
        loss_3 = (self.criterion(gg_3, labels) + self.criterion(gl_3, labels) + self.criterion(ll_3, labels)) / 3.0
        loss_4 = (self.criterion(gg_4, labels) + self.criterion(gl_4, labels) + self.criterion(ll_4, labels)) / 3.0
        loss_5 = (self.criterion(gg_5, labels) + self.criterion(gl_5, labels) + self.criterion(ll_5, labels)) / 3.0
        losses = dict()
        losses['loss_2'] = loss_2
        losses['loss_3'] = loss_3
        losses['loss_4'] = loss_4
        losses['loss_5'] = loss_5
        losses['loss'] = (loss_5 + 0.8 * loss_4 + 0.5 * loss_3 + 0.2 * loss_2) / 2.5
        return losses

    def get_splitted_logits(self, pos, neg):
        logits = list(zip(pos, neg))
        logits_2 = torch.cat(logits[0], dim=1) / self.temperature
        logits_3 = torch.cat(logits[1], dim=1) / self.temperature
        logits_4 = torch.cat(logits[2], dim=1) / self.temperature
        logits_5 = torch.cat(logits[3], dim=1) / self.temperature

        return logits_2, logits_3, logits_4, logits_5

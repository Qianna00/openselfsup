from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS
from .utils import GatherLayer


class TMP(nn.Module):

    def __init__(self, backbone, neck=None, head=None, pretrained=None):
        super(TMP, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            self.neck = None
        if head is not None:
            self.head = builder.build_head(head)
        else:
            self.head = None
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            # logger = get_root_logger()
            # print_log(f'load model from: {pretrained}', logger=logger)
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)
        if self.neck is not None:
            self.neck.init_weights(init_linear='kaiming')

    def forward(self, mode='train', **data):
        if mode == 'train':
            return self.forward_train(**data)
        elif mode == 'test':
            return self.forward_test(**data)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.
        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                - ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=data['img'].shape[0])

        return outputs

    def val_step(self, data, optimizer):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=data['img'].shape[0])

        return outputs

    def forward_test(self, img, **kwargs):
        return self.neck(self.backbone(img))[0]

    def forward_train(self, img, **kwargs):
        assert img.dim() == 5, f'img must be 5 dims, but got: {img.dim()}'
        batch_size = img.shape[0]

        img = torch.cat(torch.unbind(img, dim=1), dim=0)
        z = self.neck(self.backbone(img))[0]
        z = F.normalize(z, dim=1)

        z1, z2 = torch.split(z, [batch_size, batch_size], dim=0)
        z = torch.cat((z1.unsqueeze(1), z2.unsqueeze(1)), dim=1)
        z = torch.cat(GatherLayer.apply(z), dim=0)
        losses = self.head(z, **kwargs)

        return losses
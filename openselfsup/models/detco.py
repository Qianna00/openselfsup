import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS


@MODELS.register_module
class DetCo(nn.Module):
    '''MOCO.
    Part of the code is borrowed from:
        "https://github.com/facebookresearch/moco/blob/master/moco/builder.py".
    '''

    def __init__(self,
                 backbone,
                 neck_1=None,
                 neck_2=None,
                 neck_3=None,
                 neck_4=None,
                 neck_p1=None,
                 neck_p2=None,
                 neck_p3=None,
                 neck_p4=None,
                 head=None,
                 pretrained=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 **kwargs):
        super(DetCo, self).__init__()
        self.backbone_q = builder.build_backbone(backbone)
        self.backbone_k = builder.build_backbone(backbone)
        self.encoder_q_necks = nn.Sequential(builder.build_neck(neck_1),
                                             builder.build_neck(neck_2),
                                             builder.build_neck(neck_3),
                                             builder.build_neck(neck_4))
        self.encoder_k_necks = nn.Sequential(builder.build_neck(neck_1),
                                             builder.build_neck(neck_2),
                                             builder.build_neck(neck_3),
                                             builder.build_neck(neck_4))
        self.encoder_q_patch_necks = nn.Sequential(builder.build_neck(neck_p1),
                                                   builder.build_neck(neck_p2),
                                                   builder.build_neck(neck_p3),
                                                   builder.build_neck(neck_p4))
        self.encoder_k_patch_necks = nn.Sequential(builder.build_neck(neck_p1),
                                                   builder.build_neck(neck_p2),
                                                   builder.build_neck(neck_p3),
                                                   builder.build_neck(neck_p4))
        # self.backbone = self.encoder_q[0]
        for param in self.backbone_k.parameters():
            param.requires_grad = False
        for param in self.encoder_k_necks.parameters():
            param.requires_grad = False
        for param in self.encoder_k_patch_necks.parameters():
            param.requires_grad = False
        self.head = builder.build_head(head)
        self.init_weights(pretrained=pretrained)

        self.queue_len = queue_len
        self.momentum = momentum

        # create the queue
        self.register_buffer("queue_2", torch.randn(feat_dim, queue_len))
        self.queue_2 = nn.functional.normalize(self.queue_2, dim=0)
        self.register_buffer("queue_3", torch.randn(feat_dim, queue_len))
        self.queue_3 = nn.functional.normalize(self.queue_3, dim=0)
        self.register_buffer("queue_4", torch.randn(feat_dim, queue_len))
        self.queue_4 = nn.functional.normalize(self.queue_4, dim=0)
        self.register_buffer("queue_5", torch.randn(feat_dim, queue_len))
        self.queue_5 = nn.functional.normalize(self.queue_5, dim=0)
        self.register_buffer("local_queue_2", torch.randn(feat_dim, queue_len))
        self.local_queue_2 = nn.functional.normalize(self.local_queue_2, dim=0)
        self.register_buffer("local_queue_3", torch.randn(feat_dim, queue_len))
        self.local_queue_3 = nn.functional.normalize(self.local_queue_3, dim=0)
        self.register_buffer("local_queue_4", torch.randn(feat_dim, queue_len))
        self.local_queue_4 = nn.functional.normalize(self.local_queue_4, dim=0)
        self.register_buffer("local_queue_5", torch.randn(feat_dim, queue_len))
        self.local_queue_5 = nn.functional.normalize(self.local_queue_5, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # self.register_buffer("local_queue_ptr", torch.zeros(1, dtype=torch.long))

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone_q.init_weights(pretrained=pretrained)
        self.backbone_k.init_weights(pretrained=pretrained)
        for neck in self.encoder_q_necks:
            neck.init_weights(init_linear='kaiming')
        for neck in self.encoder_k_necks:
            neck.init_weights(init_linear='kaiming')
        for neck in self.encoder_q_patch_necks:
            neck.init_weights(init_linear='kaiming')
        for neck in self.encoder_k_patch_necks:
            neck.init_weights(init_linear='kaiming')

        """for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)"""

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        """for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)"""
        for param_q, param_k in zip(self.backbone_q.parameters(), self.backbone_k.parameters):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)
        for param_q, param_k in zip(self.encoder_q_necks.parameters(), self.encoder_k_necks.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)
        for param_q, param_k in zip(self.encoder_q_patch_necks.parameters(), self.encoder_k_patch_necks.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, k_2, k_3, k_4, k_5, k_l_2, k_l_3, k_l_4, k_l_5):
        # gather keys before updating queue
        keys_2 = concat_all_gather(k_2)
        keys_3 = concat_all_gather(k_3)
        keys_4 = concat_all_gather(k_4)
        keys_5 = concat_all_gather(k_5)
        local_keys_2 = concat_all_gather(k_l_2)
        local_keys_3 = concat_all_gather(k_l_3)
        local_keys_4 = concat_all_gather(k_l_4)
        local_keys_5 = concat_all_gather(k_l_5)

        batch_size = k_2.shape[0]

        ptr = int(self.queue_ptr)
        # local_ptr = int(self.local_queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_2[:, ptr:ptr + batch_size] = keys_2.transpose(0, 1)
        self.queue_3[:, ptr:ptr + batch_size] = keys_3.transpose(0, 1)
        self.queue_4[:, ptr:ptr + batch_size] = keys_4.transpose(0, 1)
        self.queue_5[:, ptr:ptr + batch_size] = keys_5.transpose(0, 1)
        self.local_queue_2[:, ptr:ptr + batch_size] = local_keys_2.transpose(0, 1)
        self.local_queue_3[:, ptr:ptr + batch_size] = local_keys_3.transpose(0, 1)
        self.local_queue_4[:, ptr:ptr + batch_size] = local_keys_4.transpose(0, 1)
        self.local_queue_5[:, ptr:ptr + batch_size] = local_keys_5.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer
        # local_ptr = (local_ptr + batch_size) % self.queue_len  # move pointer
        self.queue_ptr[0] = ptr
        # self.local_queue_ptr[0] = local_ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x, x_patch):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        x_patch_gather = concat_all_gather(x_patch)

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], x_patch_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, x_patch, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        x_patch_gather = concat_all_gather(x_patch)

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], x_patch_gather[idx_this]

    def forward_train(self, img, patch, **kwargs):
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        im_q = img[:, 0, ...].contiguous()
        im_k = img[:, 1, ...].contiguous()
        patch_q = patch[:, 0, ...].contiguous().view(-1, 3, 64, 64)
        patch_k = patch[:, 1, ...].contiguous().view(-1, 3, 64, 64)
        # compute query features
        q_2, q_3, q_4, q_5 = self.backbone_q(im_q)  # queries: NxC
        print(q_2.size())
        q_2 = nn.functional.normalize(self.encoder_q_necks[0](q_2), dim=1)
        q_3 = nn.functional.normalize(self.encoder_q_necks[1](q_3), dim=1)
        q_4 = nn.functional.normalize(self.encoder_q_necks[2](q_4), dim=1)
        q_5 = nn.functional.normalize(self.encoder_q_necks[3](q_5), dim=1)
        p_q_2, p_q_3, p_q_4, p_q_5 = self.backbone_q(patch_q)
        # p_q_2 = nn.functional.normalize(self.encoder_q_patch_neck2(p_q_2), dim=1)
        q_l_2 = nn.functional.normalize(self.encoder_q_patch_necks[0](p_q_2.view(-1, 9 * p_q_2.size(1), p_q_2.size(2), p_q_2.size(3))), dim=1)
        q_l_3 = nn.functional.normalize(self.encoder_q_patch_necks[1](p_q_3.view(-1, 9 * p_q_3.size(1), p_q_3.size(2), p_q_3.size(3))), dim=1)
        q_l_4 = nn.functional.normalize(self.encoder_q_patch_necks[2](p_q_4.view(-1, 9 * p_q_4.size(1), p_q_4.size(2), p_q_4.size(3))), dim=1)
        q_l_5 = nn.functional.normalize(self.encoder_q_patch_necks[3](p_q_5.view(-1, 9 * p_q_5.size(1), p_q_5.size(2), p_q_5.size(3))), dim=1)


        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, patch_k, idx_unshuffle = self._batch_shuffle_ddp(im_k, patch_k)

            k_2, k_3, k_4, k_5 = self.backbone_k(im_k)  # keys: NxC
            k_2 = nn.functional.normalize(self.encoder_k_necks[0](k_2), dim=1)
            k_3 = nn.functional.normalize(self.encoder_k_necks[1](k_3), dim=1)
            k_4 = nn.functional.normalize(self.encoder_k_necks[2](k_4), dim=1)
            k_5 = nn.functional.normalize(self.encoder_k_necks[3](k_5), dim=1)

            p_k_2, p_k_3, p_k_4, p_k_5 = self.backbone_k(patch_k)
            k_l_2 = nn.functional.normalize(
                self.encoder_k_patch_necks[0](p_k_2.view(-1, 9 * p_k_2.size(1), p_k_2.size(2), p_k_2.size(3))), dim=1)
            k_l_3 = nn.functional.normalize(
                self.encoder_k_patch_necks[1](p_k_3.view(-1, 9 * p_k_3.size(1), p_k_3.size(2), p_k_3.size(3))), dim=1)
            k_l_4 = nn.functional.normalize(
                self.encoder_k_patch_necks[2](p_k_4.view(-1, 9 * p_k_4.size(1), p_k_4.size(2), p_k_4.size(3))), dim=1)
            k_l_5 = nn.functional.normalize(
                self.encoder_k_patch_necks[3](p_k_5.view(-1, 9 * p_k_5.size(1), p_k_5.size(2), p_k_5.size(3))), dim=1)

            # undo shuffle
            # k, p_k = self._batch_unshuffle_ddp(k, p_k, idx_unshuffle)
            k_2, p_k_2 = self._batch_unshuffle_ddp(k_2, p_k_2, idx_unshuffle)
            k_3, p_k_3 = self._batch_unshuffle_ddp(k_3, p_k_3, idx_unshuffle)
            k_4, p_k_4 = self._batch_unshuffle_ddp(k_4, p_k_4, idx_unshuffle)
            k_5, p_k_5 = self._batch_unshuffle_ddp(k_5, p_k_5, idx_unshuffle)

        q_l = [q_l_2, q_l_3, q_l_4, q_l_5]
        q = [q_2, q_3, q_4, q_5]
        k_l = [k_l_2, k_l_3, k_l_4, k_l_5]
        k = [k_2, k_3, k_4, k_5]

        # compute logits
        # Einstein sum is more intuitive
        queue_l = [self.local_queue_2, self.local_queue_3, self.local_queue_4, self.local_queue_5]
        queue_g = [self.queue_2, self.queue_3, self.queue_4, self.queue_5]

        gg_pos, gg_neg = self.compute_logits(q, k, queue_g)
        gl_pos, gl_neg = self.compute_logits(q_l, k, queue_g)
        ll_pos, ll_neg = self.compute_logits(q_l, k_l, queue_l)

        losses = self.head(gg_pos, gg_neg, gl_pos, gl_neg, ll_pos, ll_neg)
        self._dequeue_and_enqueue(k_2, k_3, k_4, k_5, k_l_2, k_l_3, k_l_4, k_l_5)

        return losses

    def compute_logits(self, q, k, queue):
        # positive logits: Nx1
        pos_2 = torch.einsum('nc,nc->n', [q[0], k[0]]).unsqueeze(-1)
        pos_3 = torch.einsum('nc,nc->n', [q[1], k[1]]).unsqueeze(-1)
        pos_4 = torch.einsum('nc,nc->n', [q[2], k[2]]).unsqueeze(-1)
        pos_5 = torch.einsum('nc,nc->n', [q[3], k[3]]).unsqueeze(-1)
        # negative logits: NxK
        neg_2 = torch.einsum('nc,ck->nk', [q[0], queue[0].clone().detach()])
        neg_3 = torch.einsum('nc,ck->nk', [q[1], queue[1].clone().detach()])
        neg_4 = torch.einsum('nc,ck->nk', [q[2], queue[2].clone().detach()])
        neg_5 = torch.einsum('nc,ck->nk', [q[3], queue[3].clone().detach()])

        return [pos_2, pos_3, pos_4, pos_5], [neg_2, neg_3, neg_4, neg_5]

    def forward_test(self, img, **kwargs):
        pass

    def forward(self, img, patch, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, patch, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.backbone_q(img)
        else:
            raise Exception("No such mode: {}".format(mode))


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

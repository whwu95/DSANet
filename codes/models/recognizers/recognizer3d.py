from ..builder import RECOGNIZERS
from .base import BaseRecognizer
from ..modules_maker.block_maker import block_maker
import torch

@RECOGNIZERS.register_module
class Recognizer3D(BaseRecognizer):
    def __init__(self,
                 backbone,
                 cls_head,
                 backbone_type='3D',
                 fcn_testing=False,
                 train_cfg=None,
                 test_cfg=None,
                 module_cfg=None,
                 load_from=None):
        super(Recognizer3D, self).__init__(backbone, cls_head)
        self.backbone_type = backbone_type
        self.fcn_testing = fcn_testing
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.module_cfg = module_cfg
        self.load_from = load_from

        # load checkpoint
        if self.load_from:
            self.load_finetune(self.load_from)
        # insert module into backbone
        if self.module_cfg:
            self._prepare_base_model(self.module_cfg)

    def _prepare_base_model(self, module_cfg):
        module_name = module_cfg.pop('type')
        print('Adding {0} into backbone'.format(module_name))
        if 'build_function' in module_cfg and module_cfg.pop('build_function') == 'block_maker':
            block_maker(self.backbone, module_name, **module_cfg)
        else:
            NotImplementedError


    def load_finetune(self, load_from):
        # load checkpoint
        checkpoint = torch.load(load_from, map_location='cpu')
        import collections
        backbone_new_dict = collections.OrderedDict()
        cls_new_dict = collections.OrderedDict()

        for k, v in checkpoint['state_dict'].items():
            if 'backbone' in k:
                backbone_new_dict[k[9:]] = v
            if 'cls_head' in k:
                cls_new_dict[k[9:]] = v
        self.backbone.load_state_dict(backbone_new_dict)
        self.cls_head.load_state_dict(cls_new_dict)
        print("Load Finetune Done! {} ".format(load_from))

    def forward_train(self, imgs, labels, num_clips=1, num_crops=1, **kwargs):
        b, u, c, t, h, w = imgs.size()
        if type(num_clips) is torch.Tensor:
            num_clips = num_clips[0].item()
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        if self.backbone_type == '2D':
            imgs = imgs.permute(0, 2, 1, 3, 4).contiguous()
            imgs = imgs.view((-1, ) + imgs.shape[2:])
            x = self.extract_feat(imgs)
            x = x.view((-1, t) + x.shape[1:])
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        else:
            x = self.extract_feat(imgs)

        losses = dict()
        if self.with_cls_head:
            cls_score = self.cls_head(x, num_clips, num_crops)
            gt_label = labels.squeeze()
            loss_cls = self.cls_head.loss(cls_score, gt_label)
            losses.update(loss_cls)

        return losses

    def forward_test(self, imgs, return_numpy, num_clips=1, num_crops=1, **kwargs):
        if self.test_cfg is not None and 'val_method' in self.test_cfg and self.test_cfg['val_method'] == 'DSA_val_method':
            S, Crop, c, t, h, w = imgs.size()
            if type(num_clips) is torch.Tensor:
                num_clips = num_clips[0].item()
            if type(num_crops) is torch.Tensor:
                num_crops = num_crops[0].item()

            assert num_clips % 4 == 0, ("num_clips {}".format(num_clips))
            imgs = imgs.reshape((-1, num_crops) + imgs.shape[2:])

            cls_score_list = []
            for i in range(0, num_clips // 4):
                img_group = []
                imgs_part_list = []

                for j in range(4):
                    imgs_part = imgs[(j * num_clips // 4) + i, :]
                    imgs_part_list.append(imgs_part)
                img_group = torch.stack(imgs_part_list, dim=0)
                img_group = img_group.reshape((-1,) + img_group.shape[2:])

                if self.backbone_type == '2D':
                    img_group = img_group.permute(0, 2, 1, 3, 4).contiguous()
                    img_group = img_group.view((-1,) + img_group.shape[2:])
                    x = self.extract_feat(img_group)
                    x = x.view((-1, t) + x.shape[1:])
                    x = x.permute(0, 2, 1, 3, 4).contiguous()
                else:
                    x = self.extract_feat(img_group)

                if self.with_cls_head:
                    cls_score_part = self.cls_head(x, 4, num_crops)
                    cls_score_list.append(cls_score_part)
            cls_score = torch.stack(cls_score_list, dim=0)
            cls_score = cls_score.mean(dim=0, keepdim=True)
            cls_score = cls_score.mean(dim=1, keepdim=True)
            cls_score = cls_score.squeeze(0)

            if return_numpy:
                return cls_score.cpu().numpy()
            else:
                return cls_score

        else:
            S, Crop, c, t, h, w = imgs.size()
            if type(num_clips) is torch.Tensor:
                num_clips = num_clips[0].item()
            if type(num_crops) is torch.Tensor:
                num_crops = num_crops[0].item()

            imgs = imgs.reshape((-1, ) + imgs.shape[2:])

            if self.backbone_type == '2D':
                imgs = imgs.permute(0, 2, 1, 3, 4).contiguous()
                imgs = imgs.view((-1,) + imgs.shape[2:])
                x = self.extract_feat(imgs)
                x = x.view((-1, t) + x.shape[1:])
                x = x.permute(0, 2, 1, 3, 4).contiguous()
            else:
                x = self.extract_feat(imgs)

            if self.with_cls_head:
                cls_score = self.cls_head(x, num_clips, num_crops)
                cls_score = self.average_clip(cls_score)

            if return_numpy:
                return cls_score.cpu().numpy()
            else:
                return cls_score

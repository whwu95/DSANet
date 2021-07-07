import torch
import torch.nn as nn
import codes.models.modules_maker as modules_maker


def block_maker(net,
                  module_name=None,
                  insert_freq=(1, 1, 1, 1),
                  **kwargs):
    make_function_block = modules_maker.__dict__[module_name]
    if module_name is None:
        raise NotImplementedError

    def _make_block(stage, insert_freq, **kwargs):
        if isinstance(insert_freq, int) and insert_freq == 1:
            blocks = list(stage.children())
            print('=> Processing stage with {} blocks residual'.format(len(blocks)))
            for i, b in enumerate(blocks):
                blocks[i] = make_function_block(b, **kwargs)
                blocks[i].init_weights()
            return nn.Sequential(*blocks)

        elif isinstance(insert_freq, tuple) and len(insert_freq) > 1:
            blocks = list(stage.children())
            assert len(insert_freq) == len(blocks)
            print('=> Processing stage with {} blocks residual'.format(len(blocks)))
            for i, b in enumerate(blocks):
                if insert_freq[i] == 1:
                    blocks[i] = make_function_block(b, **kwargs)
                    blocks[i].init_weights()
            return nn.Sequential(*blocks)
        else:
            raise NotImplementedError

    net.layer1 = _make_block(
        net.layer1, insert_freq[0], **kwargs) if insert_freq[0] else net.layer1
    net.layer2 = _make_block(
        net.layer2, insert_freq[1], **kwargs) if insert_freq[1] else net.layer2
    net.layer3 = _make_block(
        net.layer3, insert_freq[2], **kwargs) if insert_freq[2] else net.layer3
    net.layer4 = _make_block(
        net.layer4, insert_freq[3], **kwargs) if insert_freq[3] else net.layer4


if __name__ == '__main__':
    from codes.models.backbones.resnet_i3d import ResNet_I3D

    input = torch.rand(32, 3, 4, 224, 224)
    model = ResNet_I3D(pretrained=None,
        pretrained2d=True,
        depth=50,
        num_stages=4,
        out_indices=[3],
        frozen_stages=-1,
        inflate_freq=(0, 0, 1, 1),
        inflate_style='3x1x1',
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_kernel_t=1,
        pool1_stride_t=1,
        no_pool2=True,
        norm_eval=False,
        partial_norm=False,
        norm_cfg=dict(type='BN3d', requires_grad=True),
        style='pytorch',
        zero_init_residual=False,
        with_cp=False)
    block_maker(model,
                module_name='Make_DSA',
                insert_freq=(0, 0, 0, (1,0,0)),
                depth=50,
                insert_place='res_2')

    print(model)
    output = model.forward(input)
    print(output.shape)
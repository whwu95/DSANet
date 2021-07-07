# Copyright (c) X-Lab, Sensetime Inc.
# Hao Shao, Manyuan Zhang, Yu Liu.
# If you have. any quastion please concat shaohao@sensetime.com

import argparse

parser = argparse.ArgumentParser(description="PyTorch of Video Framework")
parser.add_argument('dataset', type=str)
parser.add_argument('modality', type=str, choices=['RGB', 'Flow'])
parser.add_argument('--train_list', type=str, default="")
parser.add_argument('--val_list', type=str, default="")
parser.add_argument('--root_path', type=str, default="")
parser.add_argument('--store_name', type=str, default="")
parser.add_argument('--suffix', type=str, default=None)
parser.add_argument('--video_source', default=False, action='store_true',
                    help='the training data comes from video directly')

# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--num_segments', type=int,
                    default=1, help='the number of input frames')
parser.add_argument('--new_length', type=int,
                    default=1, help='the length of input clip')
parser.add_argument('--consensus_type', type=str, default='avg')
parser.add_argument('--model_type', type=str,
                    default='2D', help='the model is based on 2D or 3D')
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll', 'bce'])
parser.add_argument('--img_feature_dim', default=256,
                    type=int, help="the feature dimension for each frame")
parser.add_argument('--pretrain', type=str, default='imagenet')
parser.add_argument('--tune_from', type=str,
                    default=None, help='fine-tune from checkpoint')
parser.add_argument('--use_ema', default=False,
                    action="store_true", help='use ema')
parser.add_argument('--use_warmup', default=False,
                    action="store_true", help='use warmup')
parser.add_argument('--use_syncbn', default=False, action="store_true")
# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_type', default='step', type=str,
                    metavar='LRtype', help='learning rate type')
parser.add_argument('--lr_steps', default=[50, 100], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay LR by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping')
parser.add_argument('--no_partialbn', '--npb',
                    default=False, action="store_true")
parser.add_argument('--multi_class', default=False, type=bool,
                    help='the task and dataset is multi-class or not')
parser.add_argument('--loss_factor', default=1.0, type=float,
                    help='the ratio which multiply with loss')
parser.add_argument('--reweight', default=False, action='store_true')
parser.add_argument('--warmup_epochs', default=-1.0, type=float,
                    help='warmup epochss')

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')

# ========================= Runtime Configs ==========================
parser.add_argument('--test_mode', action='store_true')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--resume', default=False, action='store_true',
                    help='load latest checkpoint or not')
parser.add_argument('--resume_path', default='', type=str)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', default="", type=str)
parser.add_argument('--root_log', type=str, default='experiments')
parser.add_argument('--root_model', type=str, default='experiments')

# ============================ Model Configs ================================
parser.add_argument('--temporal_pool', default=False, action="store_true",
                    help='add temporal pooling')
parser.add_argument('--non_local', default=False, action="store_true",
                    help='add non local block')
parser.add_argument('--dense_sample', default=False, action="store_true",
                    help='use dense sample for video dataset')
parser.add_argument('--max_pooling', default=False, action="store_true",
                    help='use max pooling after 1st conv layer')


#  ========================== learnable shift Configs =======================
parser.add_argument('--shift', default=False, action="store_true",
                    help='use shift for models')
parser.add_argument('--shift_div', default=8, type=int,
                    help='number of div for shift (default: 8)')
parser.add_argument('--shift_place', default='blockres', type=str,
                    help='place for shift (default: stageres)')

parser.add_argument('--is_dtn', default=False, action="store_true",
                    help='add temporal deformable')
parser.add_argument('--bias_model', default='conv_2fc', type=str,
                    help='arch of bias model')
parser.add_argument('--is_bias_sigmoid', default=False, action='store_true',
                    help='use sigmoid at the end of bias model')
parser.add_argument('--weight_model', default='conv_fc', type=str,
                    help='arch of weight model')
parser.add_argument('--is_weight_sigmoid', default=False, action='store_true',
                    help='use sigmoid at the end of weight model')
parser.add_argument('--scale_model', default='empty', type=str,
                    help='arch of scale model')
parser.add_argument('--is_scale_sigmoid', default=False, action='store_true',
                    help='use sigmoid at the end of scale model')
parser.add_argument('--deform_group', default=2, type=int,
                    help='groups of deforamble bias channel')
parser.add_argument('--bias_weight', default=0.5, type=float,
                    help='the weight which use to multi bias')
parser.add_argument('--bias_init_std', default=-1, type=float,
                    help='the std of normal std in bias conv')
parser.add_argument('--extended', default=True, type=bool,
                    help='use extended feature or not')
parser.add_argument('--before_weight', default=False, type=bool,
                    help='multi weight to data before inpolote')
parser.add_argument('--deform_mode', default='bias', type=str)
parser.add_argument('--share_network', default=False, type=bool,
                    help='use share interplote network')
parser.add_argument('--bias_conv_kernel', default=3, type=int,
                    help='the kernel size of conv layer')
parser.add_argument('--bias_conv_padding', default=1, type=int,
                    help='the padding of conv layer')

# ============================ Test Configs ==================================
parser.add_argument('--test_samples', default=1, type=int,
                    help='the test crops numbers in temporal dimension')
parser.add_argument('--dup_samples', default=1, type=int,
                    help='the number of samples for each video')
parser.add_argument('--random_crops', default=1, type=int,
                    help='the spatial crops for each video')
parser.add_argument('--input_size', default=224, type=int,
                    help='the input size of crops for each video')
parser.add_argument('--scale_size', default=256, type=int,
                    help='the scaled size of crops for each video')
parser.add_argument('--crop_reverse_aug', default=False, type=int,
                    help='the aug of reverse crops for each video')
parser.add_argument('--new_step', default=1, type=int,
                    help='the temporal stride when sample frames from dataset')
parser.add_argument('--test_crops', type=int, default=1,
                    help='get 1/3/5/10 crops from single image')
# ========================== Apex Configs ====================================
# the param is from torch.distributed.launch, local_rank means gpu_id
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--opt-level', type=str, default='O0')
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)

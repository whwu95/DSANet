# Copyright (c) X-Lab, Sensetime Inc.
# Hao Shao, Manyuan Zhang, Yu Liu.
# If you have. any quastion please concat shaohao@sensetime.com


ROOT_DATASET = '/data/wuwenhao/'  # dataset root


def return_ucf101(modality):
    filename_categories = 'datalist/ucf101/classInd.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'UCF101/ucf101_rgb_img_340'
        filename_imglist_train = \
            'datalist/ucf101/ucf101_rgb_train_split_1.txt'
        filename_imglist_val = '.datalist/ucf101/ucf101_rgb_val_split_1.txt'
        filename_imglist_test = 'datalist/ucf101/val_videofolder.txt'
        prefix = 'image_{:04d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'UCF101/ucf101_flow_img_tvl1_gpu'
        filename_imglist_train = \
            'datalist/ucf101/ucf101_flow_train_split_1.txt'
        filename_imglist_val = 'datalist/ucf101/ucf101_flow_val_split_1.txt'
        prefix = 'flow_{}_{:04d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, \
        filename_imglist_test, root_data, prefix


def return_hmdb51(modality):
    filename_categories = 51
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'HMDB51/hmdb51_rgb_img_256_340'
        filename_imglist_train = \
            'datalist/hmdb51/hmdb51_rgb_train_split_1.txt'
        filename_imglist_val = 'datalist/hmdb51/hmdb51_rgb_val_split_1.txt'
        filename_imglist_test = 'datalist/hmdb51/test_videofolder.txt'
        prefix = 'image_{:06d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'HMDB51/images'  # don't exist yet
        filename_imglist_train = 'datalist/hmdb51/train_videofolder.txt'
        filename_imglist_val = 'datalist/hmdb51/test_videofolder.txt'
        filename_imglist_test = 'datalist/hmdb51/test_videofolder.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, \
        filename_imglist_test, root_data, prefix


def return_something(modality):
    filename_categories = 'datalist/something/v1/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1'
        filename_imglist_train = \
            'datalist/something/v1/train_videofolder.txt'
        filename_imglist_test = 'datalist/something/v1/test_videofolder.txt'
        filename_imglist_val = 'datalist/something/v1/val_videofolder.txt'
        prefix = '{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'datalist/something/v1/flow'
        filename_imglist_train = \
            'datalist/something/v1/train_videofolder_flow.txt'
        filename_imglist_val = 'datalist/something/v1/val_videofolder_flow.txt'
        prefix = '{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, \
        filename_imglist_test, root_data, prefix


def return_somethingv2(modality):
    filename_categories = '../datalist/something/v2/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + \
            'something/v2/20bn-something-something-v2-frames'
        filename_imglist_train = \
            'datalist/something/v2/train_videofolder.txt'
        filename_imglist_val = 'datalist/something/v2/val_videofolder.txt'
        filename_imglist_test = 'datalist/something/v1/test_videofolder.txt'
        prefix = '{:06d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + \
            'something/v2/20bn-something-something-v2-flow'
        filename_imglist_train = 'something/v2/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v2/val_videofolder_flow.txt'
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, \
        filename_imglist_test, root_data, prefix


def return_jester(modality):
    filename_categories = 'jester/category.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = 'datalist/jester/20bn-jester-v1'
        filename_imglist_train = 'datalist/jester/train_videofolder.txt'
        filename_imglist_test = 'datalist/something/v1/test_videofolder.txt'
        filename_imglist_val = 'datalist/jester/val_videofolder.txt'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, \
        filename_imglist_test, root_data, prefix


def return_kinetics_400(modality):
    filename_categories = 400
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'Kinetics400'
        filename_imglist_train = \
            'datalist/kinetics-400/kinetics_img_train_list.txt'
        filename_imglist_val = \
            'datalist/kinetics-400/kinetics_img_val_list.txt'
        filename_imglist_test = \
            'datalist/kinetics-400/kinetics_img_val_list.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, \
        filename_imglist_test, root_data, prefix


def return_kinetics_700(modality):
    filename_categories = 700
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'kinetics_700/train_frames'
        filename_imglist_train = 'datalist/kinetics-700/train_videofolder.txt'
        filename_imglist_val = 'datalist/kinetics-700/val_videofolder.txt'
        filename_imglist_test = ''
        prefix = 'image_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, \
        filename_imglist_test, root_data, prefix


def return_kinetics_600(modality):
    filename_categories = 600
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'kinetics_600/train_frames'
        filename_imglist_train = 'datalist/kinetics-600/train_videofolder.txt'
        filename_imglist_val = 'datalist/kinetics-600/val_videofolder.txt'
        filename_imglist_test = ''
        prefix = 'image_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, \
        filename_imglist_test, root_data, prefix


def return_mmit(modality):
    filename_categories = 313
    if modality == 'RGB':
        prefix = 'image_{:05d}.jpg'
        root_data = ROOT_DATASET + 'Multi_Moments_in_Time/video_frames'
        filename_imglist_train = \
            'datalist/mmit/train_videofolder.txt'  # .balance'
        filename_imglist_val = 'datalist/mmit/val_videofolder.txt'
        filename_imglist_test = 'datalist/mmit/test_videofolder.txt'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, \
        filename_imglist_test, root_data, prefix


def return_mit(modality):
    filename_categories = 339
    if modality == 'RGB':
        prefix = 'image_{:05d}.jpg'
        root_data = 'datasets/mit2/'
        filename_imglist_train = \
            'datalist/mit/train_videofolder.txt'  # .balance'
        filename_imglist_val = 'datalist/mit/val_videofolder.txt'
        filename_imglist_test = 'datalist/mit/test_videofolder.txt'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, \
        filename_imglist_test, root_data, prefix


def return_dataset(dataset, modality):
    dict_single = {'jester': return_jester, 'something': return_something,
                   'somethingv2': return_somethingv2, 'ucf101': return_ucf101,
                   'hmdb51': return_hmdb51,
                   'kinetics_400': return_kinetics_400,
                   'kinetics_700': return_kinetics_700,
                   'kinetics_600': return_kinetics_600,
                   'mmit': return_mmit, 'mit': return_mit}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, \
            file_imglist_test, root_data, prefix = dict_single[dataset](
                modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    # file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    # file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    # file_imglist_test = os.path.join(ROOT_DATASET, file_imglist_test)
    if isinstance(file_categories, str):
        # file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    return n_class, file_imglist_train, file_imglist_val,\
        file_imglist_test, root_data, prefix

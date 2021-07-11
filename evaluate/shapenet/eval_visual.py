import argparse
import os
import random
import sys

import numba
import numpy as np

from visualizer.visual_shapenet import shapenet_color_classes
from pathlib import Path

sys.path.append(os.getcwd())

__all__ = ['evaluate']


def prepare():
    from utils.common import get_save_path
    from utils.config import configs
    from utils.device import set_cuda_visible_devices

    # since PyTorch jams device selection, we have to parse args before import torch (issue #26790)
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', nargs='+')
    parser.add_argument('--devices', default=None)
    args, opts = parser.parse_known_args()
    if args.devices is not None and args.devices != 'cpu':
        gpus = set_cuda_visible_devices(args.devices)
    else:
        gpus = []

    print(f'==> loading configs from {args.configs}')
    configs.update_from_modules(*args.configs)
    # define save path
    save_path = get_save_path(*args.configs, prefix='runs')
    os.makedirs(save_path, exist_ok=True)
    configs.train.save_path = save_path
    configs.train.checkpoint_path = os.path.join(save_path, 'latest.pth.tar')
    configs.train.best_checkpoint_path = os.path.join(save_path, 'best.pth.tar')

    # override configs with args
    configs.update_from_arguments(*opts)
    if len(gpus) == 0:
        configs.device = 'cpu'
        configs.device_ids = []
    else:
        configs.device = 'cuda'
        configs.device_ids = gpus
    configs.dataset.split = configs.evaluate.dataset.split
    if 'best_checkpoint_path' not in configs.evaluate or configs.evaluate.best_checkpoint_path is None:
        if 'best_checkpoint_path' in configs.train and configs.train.best_checkpoint_path is not None:
            configs.evaluate.best_checkpoint_path = configs.train.best_checkpoint_path
        else:
            configs.evaluate.best_checkpoint_path = os.path.join(configs.train.save_path, 'best.pth.tar')
    assert configs.evaluate.best_checkpoint_path.endswith('.pth.tar')
    configs.evaluate.stats_path = configs.evaluate.best_checkpoint_path.replace('.pth.tar', '.eval.npy')

    return configs


def evaluate(configs=None):
    configs = prepare() if configs is None else configs

    import math
    import torch
    import torch.backends.cudnn as cudnn
    import torch.nn.functional as F
    from tqdm import tqdm

    from meters.shapenet import MeterShapeNet

    ###########
    # Prepare #
    ###########

    if configs.device == 'cuda':
        cudnn.benchmark = True
        if configs.get('deterministic', False):
            cudnn.deterministic = True
            cudnn.benchmark = False
    if ('seed' not in configs) or (configs.seed is None):
        configs.seed = torch.initial_seed() % (2 ** 32 - 1)
    seed = configs.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(configs)

    if os.path.exists(configs.evaluate.stats_path):
        stats = np.load(configs.evaluate.stats_path)
        print('clssIoU: {}'.format('  '.join(map('{:>8.2f}'.format, stats[:, 0] / stats[:, 1] * 100))))
        print('meanIoU: {:4.2f}'.format(stats[:, 0].sum() / stats[:, 1].sum() * 100))
        return

    #################################
    # Initialize DataLoaders, Model #
    #################################

    print(f'\n==> loading dataset "{configs.dataset}"')
    dataset = configs.dataset()[configs.dataset.split]
    meter = MeterShapeNet()

    print(f'\n==> creating model "{configs.model}"')
    model = configs.model()
    if configs.device == 'cuda':
        model = torch.nn.DataParallel(model)
    model = model.to(configs.device)

    if os.path.exists(configs.evaluate.best_checkpoint_path):
        print(f'==> loading checkpoint "{configs.evaluate.best_checkpoint_path}"')
        checkpoint = torch.load(configs.evaluate.best_checkpoint_path)
        model.load_state_dict(checkpoint.pop('model'))
        del checkpoint
    else:
        return

    model.eval()

    ##############
    # Evaluation #
    ##############

    stats = np.zeros((configs.data.num_shapes, 2))

    # define the output directory for visualization
    visual_dir = 'visual/shapenet/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)
    # output the iou of all models with .txt file
    all_iou_save = open(os.path.join(visual_dir, 'shapenet_iou' + '.txt'), 'w')

    for shape_index, (file_path, shape_id) in enumerate(tqdm(dataset.file_paths, desc='eval', ncols=0)):
        data = np.loadtxt(file_path).astype(np.float32)
        total_num_points_in_shape = data.shape[0]
        confidences = np.zeros(total_num_points_in_shape, dtype=np.float32)
        predictions = np.full(total_num_points_in_shape, -1, dtype=np.int64)

        coords = data[:, :3]
        if dataset.normalize:
            coords = dataset.normalize_point_cloud(coords)
        coords = coords.transpose()
        ground_truth = data[:, -1].astype(np.int64)
        if dataset.with_normal:
            normal = data[:, 3:6].transpose()
            if dataset.with_one_hot_shape_id:
                shape_one_hot = np.zeros((dataset.num_shapes, coords.shape[-1]), dtype=np.float32)
                shape_one_hot[shape_id, :] = 1.0
                point_set = np.concatenate([coords, normal, shape_one_hot])
            else:
                point_set = np.concatenate([coords, normal])
        else:
            if dataset.with_one_hot_shape_id:
                shape_one_hot = np.zeros((dataset.num_shapes, coords.shape[-1]), dtype=np.float32)
                shape_one_hot[shape_id, :] = 1.0
                point_set = np.concatenate([coords, shape_one_hot])
            else:
                point_set = coords
        extra_batch_size = configs.evaluate.num_votes * math.ceil(total_num_points_in_shape / dataset.num_points)
        total_num_voted_points = extra_batch_size * dataset.num_points
        num_repeats = math.ceil(total_num_voted_points / total_num_points_in_shape)
        shuffled_point_indices = np.tile(np.arange(total_num_points_in_shape), num_repeats)
        shuffled_point_indices = shuffled_point_indices[:total_num_voted_points]
        np.random.shuffle(shuffled_point_indices)
        start_class, end_class = meter.part_class_to_shape_part_classes[ground_truth[0]]

        # model inference
        inputs = torch.from_numpy(
            point_set[:, shuffled_point_indices].reshape(-1, extra_batch_size, dataset.num_points).transpose(1, 0, 2)
        ).float().to(configs.device)
        with torch.no_grad():
            vote_confidences = F.softmax(model(inputs), dim=1)
            vote_confidences, vote_predictions = vote_confidences[:, start_class:end_class, :].max(dim=1)
            vote_confidences = vote_confidences.view(total_num_voted_points).cpu().numpy()
            vote_predictions = (vote_predictions + start_class).view(total_num_voted_points).cpu().numpy()

        update_shape_predictions(vote_confidences, vote_predictions, shuffled_point_indices,
                                 confidences, predictions, total_num_voted_points)
        iou = update_stats(stats, ground_truth, predictions, shape_id, start_class, end_class)

        # output the visualization results with .obj file
        fout = open(os.path.join(visual_dir, str(shape_id) + '_' + str(shape_index) + '_pred.obj'), 'w')
        fout_gt = open(os.path.join(visual_dir, str(shape_id) + '_' + str(shape_index) + '_gt.obj'), 'w')

        #filename_visual = os.path.join(visual_dir, str(shape_id) + '_' + str(shape_index) + '.txt')
        #with open(filename_visual, 'w') as pl_save:
         #   for pre_i in predictions:
          #      pl_save.write(str(int(pre_i)) + '\n')
           # pl_save.close()
        #filename_iou = os.path.join(visual_dir, str(shape_id) + '_' + str(shape_index) + 'iou' + str(format(iou, '.4f')) + '.txt')
        #with open(filename_iou, 'w') as single_iou_save:
         #   single_iou_save.write(str(iou))
          #  single_iou_save.close()

        all_iou_save.write(str(shape_id) + '_' + str(shape_index) + ' ' + str(format(iou, '.4f')) + '\n')
        for i in range(ground_truth.shape[0]):
            if predictions[i] == -1:
                color = shapenet_color_classes[50]
            else:
                color = shapenet_color_classes[predictions[i]]
                # color = g_label2color[predictions[i]]
            color_gt = shapenet_color_classes[ground_truth[i]]
            if 1:
                fout.write('v %f %f %f %d %d %d\n' % (
                    data[i, 0], data[i, 1], data[i, 2], color[0], color[1],
                    color[2]))
                fout_gt.write(
                    'v %f %f %f %d %d %d\n' % (
                        data[i, 0], data[i, 1], data[i, 2], color_gt[0],
                        color_gt[1], color_gt[2]))
        fout.close()
        fout_gt.close()
    all_iou_save.close()
    np.save(configs.evaluate.stats_path, stats)
    print('clssIoU: {}'.format('  '.join(map('{:>8.2f}'.format, stats[:, 0] / stats[:, 1] * 100))))
    print('meanIoU: {:4.2f}'.format(stats[:, 0].sum() / stats[:, 1].sum() * 100))


@numba.jit()
def update_shape_predictions(vote_confidences, vote_predictions, shuffled_point_indices,
                             shape_confidences, shape_predictions, total_num_voted_points):
    for p in range(total_num_voted_points):
        point_index = shuffled_point_indices[p]
        current_confidence = vote_confidences[p]
        if current_confidence > shape_confidences[point_index]:
            shape_confidences[point_index] = current_confidence
            shape_predictions[point_index] = vote_predictions[p]


@numba.jit()
def update_stats(stats, ground_truth, predictions, shape_id, start_class, end_class):
    iou = 0.0
    for i in range(start_class, end_class):
        igt = (ground_truth == i)
        ipd = (predictions == i)
        union = np.sum(igt | ipd)
        intersection = np.sum(igt & ipd)
        if union == 0:
            iou += 1
        else:
            iou += intersection / union
    iou /= (end_class - start_class)
    stats[shape_id][0] += iou
    stats[shape_id][1] += 1
    return iou


if __name__ == '__main__':
    evaluate()

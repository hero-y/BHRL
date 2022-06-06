import argparse
import os
import os.path as osp
import shutil
import tempfile
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.core import coco_eval, results2json
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
import numpy as np

def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    img_ids = []
    img_labels = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)
        img_id = data['img_metas'][0].data[0][0]['img_info']['id']
        label = data['img_metas'][0].data[0][0]['label']
        img_ids.append(img_id)
        img_labels.append(label)

        if rank == 0:
            batch_size = data['img'][0][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()
    # collect results from all ranks
    results, img_ids, img_labels = collect_results_id(results, len(dataset), img_ids, img_labels, tmpdir)

    return results, img_ids, img_labels

def collect_results_id(result_part, size, img_ids_part, img_labels_part, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    mmcv.dump(img_ids_part, osp.join(tmpdir, 'id_part_{}.pkl'.format(rank)))
    mmcv.dump(img_labels_part, osp.join(tmpdir, 'label_part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None, None, None
    else:
        # load results of all parts from tmp dir
        part_list = []
        id_part_list = []
        label_part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            id_part = osp.join(tmpdir, 'id_part_{}.pkl'.format(i))
            label_part = osp.join(tmpdir, 'label_part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
            id_part_list.append(mmcv.load(id_part))
            label_part_list.append(mmcv.load(label_part))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        ordered_ids = []
        for res in zip(*id_part_list):
            ordered_ids.extend(list(res))
        ordered_labels = []
        for res in zip(*label_part_list):
            ordered_labels.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        ordered_ids = ordered_ids[:size]
        ordered_labels = ordered_labels[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results, ordered_ids, ordered_labels

def parse_args():
    parser = argparse.ArgumentParser(description='BHRL test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--average', type=int, default=1)
    parser.add_argument('--test_seen_classes', action='store_true', help='test seen classes')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()

    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]

    avg = args.average

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if args.test_seen_classes:
        cfg.data.test.test_seen_classes = True
    else:
        cfg.data.test.test_seen_classes = False
    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    for i in range(avg):
        cfg.data.test.position = i 
        dataset = build_dataset(cfg.data.test) 
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        # build the model and load checkpoint
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))

        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES
        # --------------------------------
        model.CLASSES = dataset.CLASSES
        # --------------------------------
        
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs, img_ids, img_labels = multi_gpu_test(model, data_loader, args.tmpdir)

        rank, _ = get_dist_info()
        if args.out and rank == 0:
            print('\nwriting results to {}'.format(args.out))
            mmcv.dump(outputs, args.out)
            eval_types = args.eval
            if eval_types:
                print('Starting evaluate {}'.format(' and '.join(eval_types)))
                if eval_types == ['proposal_fast']:
                    result_file = args.out
                    coco_eval(result_file, eval_types, dataset.coco)
                else:
                    if not isinstance(outputs[0], dict):
                        result_files = results2json(dataset, outputs, img_ids, img_labels, args.out)
                        coco_eval(result_files, eval_types, dataset.coco, img_ids=img_ids, img_labels=img_labels)

if __name__ == '__main__':
    main()
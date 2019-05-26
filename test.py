#!/usr/bin/env python3
from argparse import ArgumentParser

import common
import embed
import evaluate
from store_args import Arguments

parser = ArgumentParser(description='Embed a dataset using a trained network.')

# Required

parser.add_argument(
    '--experiment_root', required=True,
    help='Location used to store checkpoints and dumped data.')

parser.add_argument(
    '--dataset', required=True, choices=('market1501', 'vehicle', 'cuhk03', 'veri', 'PVUD'),
    help='The dataset to be tested')

parser.add_argument(
    '--image_root', required=True,  default=None, type=common.readable_directory,
    help='Path that will be pre-pended to the filenames in the train_set csv.')

# Optional

parser.add_argument(
    '--model_name', type=str, default='resnet_v1_50',
    help='Backend model')

parser.add_argument(
    '--checkpoint', default=None,
    help='Name of checkpoint file of the trained network within the experiment '
         'root. Uses the last checkpoint if not provided.')

parser.add_argument(
    '--output_name', default="",
    help='The end of the txt and image file output name.')

parser.add_argument(
    '--gpu', default=0,
    help='ID of GPU which should be use for running.')

parser.add_argument(
    '--b4_layers', default= 1,
    help='Number of layers in the final block of ResNet (has to be in [1,2,3]')

parser.add_argument(
    '--resnet_stride', required=False, default=1, choices=('1','2'),
    help='Resnet stride on block 3')

def run_test(stored_args):
    filename = stored_args.filename
    stored_args.gallery_dataset = os.path.join('datasets', stored_args.dataset + '_gallery.txt')
    stored_args.query_dataset = os.path.join('datasets', stored_args.dataset + '_query.txt')
    if stored_args.dataset == 'vehicle': stored_args.excluder = 'diagonal'
    elif stored_args.dataset == 'cuhk03': stored_args.excluder = 'PVUD'
    else: stored_args.excluder = stored_args.dataset
    output_file = open(os.path.join(stored_args.experiment_root, stored_args.output_name + '.txt'), "a")
    query_embs = embed.run_embedding(stored_args, stored_args.query_dataset)
    stored_args.filename = filename
    gallery_embs = embed.run_embedding(stored_args, stored_args.gallery_dataset)
    [mAP, rank1] = evaluate.run_evaluation(stored_args, query_embs, gallery_embs)
    print("mAP: " + str(mAP) + "; rank-1: " + str(rank1))
    output_file.write("checkpoint: " + "mAP: " + str("%0.2f" % (mAP * 100)) + "; rank-1: " + str(
        "%0.2f" % (rank1 * 100)) + "\n")
    output_file.close()
    return [mAP, rank1]

if __name__ == '__main__':
    import os
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    stored_args = Arguments()
    stored_args.save_args(args)
    run_test(stored_args)


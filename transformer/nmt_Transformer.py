
"""


Reference: https://docs.dgl.ai/tutorials/models/4_old_wines/7_transformer.html
"""

import re
import numpy as np
import pandas as pd
import pickle
from nltk.translate.bleu_score import corpus_bleu
from sklearn.model_selection import train_test_split
import time
import argparse

# from tqdm import tqdm
import torch as T
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

import dgl.function as fn

from functools import partial

from transformer_header import LabelSmoothing, SimpleLossCompute, clones, make_model, NoamOpt
from transformer_header import UTransformer, src_dot_dst, scaled_exp, GraphPool, MultiGPULossCompute
from transformer_header import draw_atts
from data_preprocessing import get_dataset
# from dgl.contrib.transformer import get_dataset, GraphPool
import os
from dgl.data.utils import *


# Hyperparameters
DIM_MODEL = 256
NUM_EPOCHS = 100
from transformer_header import VIZ_IDX # 3


def message_func(edges):
    return {'score': ((edges.src['k'] * edges.dst['q'])
                      .sum(-1, keepdim=True)),
            'v': edges.src['v']}


def reduce_func(nodes, d_k=64):
    v = nodes.mailbox['v']
    att = F.softmax(nodes.mailbox['score'] / T.sqrt(d_k), 1)
    return {'dx': (att * v).sum(1)}


def run_epoch(epoch, data_iter, dev_rank, ndev, model, loss_compute, is_train=True):
    universal = isinstance(model, UTransformer)
    with loss_compute:
        for i, g in enumerate(data_iter):
            with T.set_grad_enabled(is_train):
                if universal:
                    output, loss_act = model(g)
                    if is_train:
                        loss_act.backward(retain_graph=True)
                else:
                    output = model(g)
                tgt_y = g.tgt_y
                n_tokens = g.n_tokens
                loss = loss_compute(output, tgt_y, n_tokens)

    if universal:
        for step in range(1, model.MAX_DEPTH + 1):
            print("nodes entering step {}: {:.2f}%".format(step, (1.0 * model.stat[step] / model.stat[0])))
        model.reset_stat()
    print('Epoch {} {}: Dev {} average loss: {}, accuracy {}'.format(
        epoch, "Training" if is_train else "Evaluating",
        dev_rank, loss_compute.avg_loss, loss_compute.accuracy))


# TODO: See why this is needed
def run(dev_id, args):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip=args.master_ip, master_port=args.master_port)
    world_size = args.ngpu
    T.distributed.init_process_group(backend="nccl",
                                         init_method=dist_init_method,
                                         world_size=world_size,
                                         rank=dev_id)
    gpu_rank = T.distributed.get_rank()
    assert gpu_rank == dev_id
    main(dev_id, args)


def main(dev_id, args):
    if dev_id == -1:
        device = T.device('cpu')
    else:
        device = T.device('cuda:{}'.format(dev_id))
    # Set current device
    T.cuda.set_device(device)
    # Prepare dataset
    dataset = get_dataset(args.dataset)
    V = dataset.vocab_size
    criterion = LabelSmoothing(V, padding_idx=dataset.pad_id, smoothing=0.1)
    dim_model = DIM_MODEL
    # Build graph pool
    graph_pool = GraphPool()
    # Create model
    model = make_model(V, V, N=args.N, dim_model=dim_model,
                       universal=args.universal)
    # Sharing weights between Encoder & Decoder
    model.src_embed.lut.weight = model.tgt_embed.lut.weight
    model.generator.proj.weight = model.tgt_embed.lut.weight
    # Move model to corresponding device
    model, criterion = model.to(device), criterion.to(device)
    # Loss function
    if args.ngpu > 1:
        dev_rank = dev_id # current device id
        ndev = args.ngpu # number of devices (including cpu)
        loss_compute = partial(MultiGPULossCompute, criterion, args.ngpu,
                               args.grad_accum, model)
    else: # cpu or single gpu case
        dev_rank = 0
        ndev = 1
        loss_compute = partial(SimpleLossCompute, criterion, args.grad_accum)

    if ndev > 1:
        for param in model.parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= ndev

    # Optimizer
    model_opt = NoamOpt(model_size=dim_model, factor=0.1, warmup=4000,
                        optimizer=T.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9))

    # Train & evaluate
    for epoch in range(NUM_EPOCHS):
        start = time.time()
        train_iter = dataset(graph_pool, mode='train', batch_size=args.batch,
                             device=device, dev_rank=dev_rank, ndev=ndev)
        model.train(True)
        run_epoch(epoch, train_iter, dev_rank, ndev, model,
                  loss_compute(opt=model_opt), is_train=True)
        if dev_rank == 0:

            model.att_weight_map = None
            model.eval()
            valid_iter = dataset(graph_pool, mode='valid', batch_size=args.batch,
                                 device=device, dev_rank=dev_rank, ndev=1)
            run_epoch(epoch, valid_iter, dev_rank, 1, model,
                      loss_compute(opt=None), is_train=False)
            end = time.time()
            print("epoch time: {}".format(end - start))

            # Visualize attention
            if args.viz and model.att_weight_map is not None:
                src_seq = dataset.get_seq_by_id(VIZ_IDX, mode='valid', field='src')
                tgt_seq = dataset.get_seq_by_id(VIZ_IDX, mode='valid', field='tgt')[:-1]
                draw_atts(model.att_weight_map, src_seq, tgt_seq, 'epochs', 'epoch_{}'.format(epoch))

            args_filter = ['batch', 'gpus', 'viz', 'master_ip', 'master_port', 'grad_accum', 'ngpu']
            exp_setting = '-'.join('{}'.format(v) for k, v in vars(args).items() if k not in args_filter)
            with open('checkpoints/viz_{}-{}.pkl'.format(exp_setting, epoch), 'wb') as f:
                T.save(model.state_dict(), f)


def train_model(dataset="anki"):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    np.random.seed(2019)

    # Use defaults
    argparser = argparse.ArgumentParser('training translation model')
    argparser.add_argument('--gpus', default='-1', type=str, help='gpu id')
    argparser.add_argument('--N', default=2, type=int, help='enc/dec layers')
    # argparser.add_argument('--dataset', default='multi30k', help='dataset')
    argparser.add_argument('--batch', default=128, type=int, help='batch size')
    argparser.add_argument('--viz', default=True, action='store_true',
                        help='visualize attention')
    argparser.add_argument('--universal', action='store_true',
                        help='use universal transformer')
    argparser.add_argument('--master-ip', type=str, default='127.0.0.1',
                        help='master ip address')
    argparser.add_argument('--master-port', type=str, default='12345',
                        help='master port')
    argparser.add_argument('--grad-accum', type=int, default=1,
                        help='accumulate gradients for this many times '
                                'then update weights')
    args = argparser.parse_args()
    args.dataset = dataset

    # devices = list(map(int, args.gpus.split(',')))
    devices = [0]
    if len(devices) == 1:
        args.ngpu = 0 if devices[0] < 0 else 1
        main(devices[0], args)
    else:
        args.ngpu = len(devices)
        mp = T.multiprocessing.get_context('spawn')
        procs = []
        for dev_id in devices:
            procs.append(mp.Process(target=run, args=(dev_id, args), daemon=True))
            procs[-1].start()
        for p in procs:
            p.join()


def test_model(k=5, dataset="anki", checkpoint=99):
    argparser = argparse.ArgumentParser('testing translation model')
    argparser.add_argument('--gpu', default=-1, help='gpu id')
    argparser.add_argument('--N', default=2, type=int, help='num of layers')
    # argparser.add_argument('--dataset', default='multi30k', help='dataset')
    argparser.add_argument('--batch', default=64, help='batch size')
    argparser.add_argument('--universal', action='store_true', help='use universal transformer')
    # argparser.add_argument('--checkpoint', default=95, type=int, help='checkpoint: you must specify it')
    argparser.add_argument('--print', default=True, action='store_true', help='whether to print translated text')
    argparser.add_argument('--printn', default=10, type=int, help='how many examples to print')
    args = argparser.parse_args()
    args.dataset = dataset
    args.checkpoint = checkpoint
    args_filter = ['batch', 'gpu', 'print', 'printn']
    exp_setting = '-'.join('{}'.format(v) for k, v in vars(args).items() if k not in args_filter)

    # device = 'cpu' if args.gpu == -1 else 'cuda:{}'.format(args.gpu)
    device = T.device('cuda:0')
    T.cuda.set_device(device)

    dataset = get_dataset(args.dataset)
    V = dataset.vocab_size
    dim_model = DIM_MODEL

    fpred = open('data/{}_pred_viz.txt'.format(args.dataset), 'w')
    # fref = open('data/ref.txt', 'w')
    try:
        graph_pool = GraphPool()
        model = make_model(V, V, N=args.N, dim_model=dim_model, universal=False)
        with open('checkpoints/{}.pkl'.format(exp_setting), 'rb') as f:
            model.load_state_dict(T.load(f, map_location=lambda storage, loc: storage))
        model = model.to(device)
        model.eval()
        test_iter = dataset(graph_pool, mode='test', batch_size=args.batch, device=device, k=k)
        pred, ytrue, src = list(), list(), list()
        for i, g in enumerate(test_iter):
            with T.no_grad():
                output = model.infer(g, dataset.MAX_LENGTH, dataset.eos_id, k, alpha=0.6)
            for line in dataset.get_sequence(output):
                # if args.print:
                #     print(line)
                print(line, file=fpred)
                pred.append(line)
    finally:
        fpred.close()
        # fref.close()

    for line in dataset.tgt['test']:
        # print(line.strip(), file=fref)
        ytrue.append(line.strip())
    for line in dataset.src['test']:
        src.append(line.strip())

    # Compute the bleu scores
    bleu_1 = corpus_bleu(ytrue, pred, weights=(1.0, 0, 0, 0))
    bleu_2 = corpus_bleu(ytrue, pred, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(ytrue, pred, weights=(0.3, 0.3, 0.3, 0))
    bleu_4 = corpus_bleu(ytrue, pred, weights=(0.25, 0.25, 0.25, 0.25))

    # Save BLEU scores
    with(open(os.path.join("data", args.dataset, "bleu_scores_viz.txt"), "w")) as f:
        f.write("BLEU 1 score: {}\n".format(bleu_1))
        f.write("BLEU 2 score: {}\n".format(bleu_2))
        f.write("BLEU 3 score: {}\n".format(bleu_3))
        f.write("BLEU 4 score: {}\n".format(bleu_4))

    if args.print:
        print("Printing {} examples:".format(args.printn))
        for i in np.random.randint(len(ytrue), size=args.printn):# range(len(args.printn)):
            print('src=[%s], target=[%s], predicted=[%s]' % (src[i], ytrue[i], pred[i]))

        # df = pd.DataFrame({'source': src, 'target': ytrue, 'predicted': pred})
        # print(df.sample(n=args.printn))

        print("\n\nAchieving the BLEU scores:\n")
        # calculate BLEU score
        print('BLEU-1: %f' % bleu_1)
        print('BLEU-2: %f' % bleu_2)
        print('BLEU-3: %f' % bleu_3)
        print('BLEU-4: %f' % bleu_4)
    #os.system(r'bash scripts/bleu.sh pred.txt ref.txt')
    # os.remove('pred.txt')
    # os.remove('ref.txt')


if __name__ == "__main__":
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    np.random.seed(1111)
    argparser = argparse.ArgumentParser('training translation model')
    argparser.add_argument('--gpus', default='-1', type=str, help='gpu id')
    argparser.add_argument('--N', default=6, type=int, help='enc/dec layers')
    argparser.add_argument('--dataset', default='multi30k', help='dataset')
    argparser.add_argument('--batch', default=128, type=int, help='batch size')
    argparser.add_argument('--viz', action='store_true',
                           help='visualize attention')
    argparser.add_argument('--universal', action='store_true',
                           help='use universal transformer')
    argparser.add_argument('--master-ip', type=str, default='127.0.0.1',
                           help='master ip address')
    argparser.add_argument('--master-port', type=str, default='12345',
                           help='master port')
    argparser.add_argument('--grad-accum', type=int, default=1,
                           help='accumulate gradients for this many times '
                                'then update weights')
    args = argparser.parse_args()
    print(args)

    devices = list(map(int, args.gpus.split(',')))
    if len(devices) == 1:
        args.ngpu = 0 if devices[0] < 0 else 1
        main(devices[0], args)
    else:
        args.ngpu = len(devices)
        mp = T.multiprocessing.get_context('spawn')
        procs = []
        for dev_id in devices:
            procs.append(mp.Process(target=run, args=(dev_id, args),
                                    daemon=True))
            procs[-1].start()
        for p in procs:
            p.join()

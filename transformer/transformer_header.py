# -*- coding: utf-8 -*-
"""
Classes and functions implemented for the NMT Graph translation model

@author: David Cecchini
@author2: Steve Beattie
"""

import numpy as np
import random
import itertools
import time
from collections import namedtuple

import torch as T
import torch.nn as nn
from torch.nn import LayerNorm
import torch.nn.init as INIT
import torch.nn.functional as F
import torch.distributed as dist
from dgl import DGLGraph, batch
from dgl.init import zero_initializer as dgl_zero_initializer
import dgl.function as fn
import copy
# import threading

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import ConnectionStyle,FancyArrowPatch
import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import is_string_like

import os

# Define some global constants/variables
VIZ_IDX = 3
mode2id = {'e2e': 0, 'e2d': 1, 'd2d': 2}
colorbar = None
Graph = namedtuple('Graph',
                   ['g', 'src', 'tgt', 'tgt_y', 'nids', 'eids', 'nid_arr', 'n_nodes', 'n_edges', 'n_tokens'])


def clones(module, k):
    return nn.ModuleList(
        copy.deepcopy(module) for _ in range(k)
    )


def src_dot_dst(src_field, dst_field, out_field):
    """
    This function serves as a surrogate for `src_dot_dst` built-in apply_edge function.
    """
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func


def scaled_exp(field, c):
    """
    This function applies $exp(x / c)$ for input $x$, which is required by *Scaled Dot-Product Attention* mentioned in the paper.
    """
    def func(edges):
        return {field: T.exp((edges.data[field] / c).clamp(-10, 10))}
    return func


def get_attention_map(g, src_nodes, dst_nodes, h):
    """
    To visualize the attention score between two set of nodes.
    """
    n, m = len(src_nodes), len(dst_nodes)
    weight = T.zeros(n, m, h).fill_(-1e8)
    for i, src in enumerate(src_nodes.tolist()):
        for j, dst in enumerate(dst_nodes.tolist()):
            if not g.has_edge_between(src, dst):
                continue
            eid = g.edge_id(src, dst)
            weight[i][j] = g.edata['score'][eid].squeeze(-1).cpu().detach()

    weight = weight.transpose(0, 2)
    att = T.softmax(weight, -2)
    return att.numpy()


def draw_heatmap(array, input_seq, output_seq, dirname, name):
    dirname = os.path.join('images', dirname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    fig, axes = plt.subplots(2, 4)
    cnt = 0
    for i in range(2):
        for j in range(4):
            axes[i, j].imshow(array[cnt].transpose(-1, -2))
            axes[i, j].set_yticks(np.arange(len(input_seq)))
            axes[i, j].set_xticks(np.arange(len(output_seq)))
            axes[i, j].set_yticklabels(input_seq, fontsize=4)
            axes[i, j].set_xticklabels(output_seq, fontsize=4)
            axes[i, j].set_title('head_{}'.format(cnt), fontsize=10)
            plt.setp(axes[i, j].get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")
            cnt += 1

    fig.suptitle(name, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(dirname, '{}.pdf'.format(name)))
    plt.close()


def draw_atts(maps, src, tgt, dirname, prefix):
    '''
    maps[0]: encoder self-attention
    maps[1]: encoder-decoder attention
    maps[2]: decoder self-attention
    '''
    draw_heatmap(maps[0], src, src, dirname, '{}_enc_self_attn'.format(prefix))
    draw_heatmap(maps[1], src, tgt, dirname, '{}_enc_dec_attn'.format(prefix))
    draw_heatmap(maps[2], tgt, tgt, dirname, '{}_dec_self_attn'.format(prefix))


def graph_att_head(M, N, weight, ax, title):
    "credit: Jinjing Zhou"
    in_nodes=len(M)
    out_nodes=len(N)

    g = nx.bipartite.generators.complete_bipartite_graph(in_nodes,out_nodes)
    X, Y = bipartite.sets(g)
    height_in = 10
    height_out = height_in
    height_in_y = np.linspace(0, height_in, in_nodes)
    height_out_y = np.linspace((height_in - height_out) / 2, height_out, out_nodes)
    pos = dict()
    pos.update((n, (1, i)) for i, n in zip(height_in_y, X))  # put nodes from X at x=1
    pos.update((n, (3, i)) for i, n in zip(height_out_y, Y))  # put nodes from Y at x=2
    ax.axis('off')
    ax.set_xlim(-1,4)
    ax.set_title(title)
    nx.draw_networkx_nodes(g, pos, nodelist=range(in_nodes), node_color='r', node_size=50, ax=ax)
    nx.draw_networkx_nodes(g, pos, nodelist=range(in_nodes, in_nodes + out_nodes), node_color='b', node_size=50, ax=ax)
    for edge in g.edges():
        nx.draw_networkx_edges(g, pos, edgelist=[edge], width=weight[edge[0], edge[1] - in_nodes] * 1.5, ax=ax)
    nx.draw_networkx_labels(g, pos, {i:label + '  ' for i,label in enumerate(M)},horizontalalignment='right', font_size=8, ax=ax)
    nx.draw_networkx_labels(g, pos, {i+in_nodes:'  ' + label for i,label in enumerate(N)},horizontalalignment='left', font_size=8, ax=ax)


def att_animation(maps_array, mode, src, tgt, head_id):
    weights = [maps[mode2id[mode]][head_id] for maps in maps_array]
    fig, axes = plt.subplots(1, 2)

    def weight_animate(i):
        global colorbar
        if colorbar:
            colorbar.remove()
        plt.cla()
        axes[0].set_title('heatmap')
        axes[0].set_yticks(np.arange(len(src)))
        axes[0].set_xticks(np.arange(len(tgt)))
        axes[0].set_yticklabels(src)
        axes[0].set_xticklabels(tgt)
        plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        fig.suptitle('epoch {}'.format(i))
        weight = weights[i].transpose(-1, -2)
        heatmap = axes[0].pcolor(weight, vmin=0, vmax=1, cmap=plt.cm.Blues)
        colorbar = plt.colorbar(heatmap, ax=axes[0], fraction=0.046, pad=0.04)
        axes[0].set_aspect('equal')
        axes[1].axis("off")
        graph_att_head(src, tgt, weight, axes[1], 'graph')

    ani = animation.FuncAnimation(fig, weight_animate, frames=len(weights), interval=500, repeat_delay=2000)
    return ani


"The following function was modified from the source code of networkx"
def draw_networkx_edges(G, pos, edgelist=None, width=1.0, edge_color='k',
                        style='solid', alpha=1.0, arrowstyle='-|>', arrowsize=10,
                        edge_cmap=None, edge_vmin=None, edge_vmax=None, ax=None,
                        arrows=True, label=None, node_size=300, nodelist=None,
                        node_shape="o", connectionstyle='arc3', **kwds):
    """Draw the edges of the graph G.
    This draws only the edges of the graph G.
    Parameters
    ----------
    G : graph
       A networkx graph
    pos : dictionary
       A dictionary with nodes as keys and positions as values.
       Positions should be sequences of length 2.
    edgelist : collection of edge tuples
       Draw only specified edges(default=G.edges())
    width : float, or array of floats
       Line width of edges (default=1.0)
    edge_color : color string, or array of floats
       Edge color. Can be a single color format string (default='r'),
       or a sequence of colors with the same length as edgelist.
       If numeric values are specified they will be mapped to
       colors using the edge_cmap and edge_vmin,edge_vmax parameters.
    style : string
       Edge line style (default='solid') (solid|dashed|dotted,dashdot)
    alpha : float
       The edge transparency (default=1.0)
    edge_ cmap : Matplotlib colormap
       Colormap for mapping intensities of edges (default=None)
    edge_vmin,edge_vmax : floats
       Minimum and maximum for edge colormap scaling (default=None)
    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.
    arrows : bool, optional (default=True)
       For directed graphs, if True draw arrowheads.
       Note: Arrows will be the same color as edges.
    arrowstyle : str, optional (default='-|>')
       For directed graphs, choose the style of the arrow heads.
       See :py:class: `matplotlib.patches.ArrowStyle` for more
       options.
    arrowsize : int, optional (default=10)
       For directed graphs, choose the size of the arrow head head's length and
       widT. See :py:class: `matplotlib.patches.FancyArrowPatch` for attribute
       `mutation_scale` for more info.
    label : [None| string]
       Label for legend
    Returns
    -------
    matplotlib.collection.LineCollection
        `LineCollection` of the edges
    list of matplotlib.patches.FancyArrowPatch
        `FancyArrowPatch` instances of the directed edges
    Depending whether the drawing includes arrows or not.
    Notes
    -----
    For directed graphs, arrows are drawn at the head end.  Arrows can be
    turned off with keyword arrows=False. Be sure to include `node_size' as a
    keyword argument; arrows are drawn considering the size of nodes.
    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edges = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (1, 3), (2, 3)])
    >>> arcs = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))
    >>> alphas = [0.3, 0.4, 0.5]
    >>> for i, arc in enumerate(arcs):  # change alpha values of arcs
    ...     arc.set_alpha(alphas[i])
    Also see the NetworkX drawing examples at
    https://networkx.github.io/documentation/latest/auto_examples/index.html
    See Also
    --------
    draw()
    draw_networkx()
    draw_networkx_nodes()
    draw_networkx_labels()
    draw_networkx_edge_labels()
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.cbook as cb
        from matplotlib.colors import colorConverter, Colormap, Normalize
        from matplotlib.collections import LineCollection
        from matplotlib.patches import FancyArrowPatch, ConnectionStyle
        import numpy as np
    except ImportError:
        raise ImportError("Matplotlib required for draw()")
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    if ax is None:
        ax = plt.gca()

    if edgelist is None:
        edgelist = list(G.edges())

    if not edgelist or len(edgelist) == 0:  # no edges!
        return None

    if nodelist is None:
        nodelist = list(G.nodes())

    # set edge positions
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])

    if not cb.iterable(width):
        lw = (width,)
    else:
        lw = width

    if not is_string_like(edge_color) \
            and cb.iterable(edge_color) \
            and len(edge_color) == len(edge_pos):
        if np.alltrue([is_string_like(c) for c in edge_color]):
            # (should check ALL elements)
            # list of color letters such as ['k','r','k',...]
            edge_colors = tuple([colorConverter.to_rgba(c, alpha)
                                 for c in edge_color])
        elif np.alltrue([not is_string_like(c) for c in edge_color]):
            # If color specs are given as (rgb) or (rgba) tuples, we're OK
            if np.alltrue([cb.iterable(c) and len(c) in (3, 4)
                           for c in edge_color]):
                edge_colors = tuple(edge_color)
            else:
                # numbers (which are going to be mapped with a colormap)
                edge_colors = None
        else:
            raise ValueError('edge_color must contain color names or numbers')
    else:
        if is_string_like(edge_color) or len(edge_color) == 1:
            edge_colors = (colorConverter.to_rgba(edge_color, alpha), )
        else:
            msg = 'edge_color must be a color or list of one color per edge'
            raise ValueError(msg)

    if (not G.is_directed() or not arrows):
        edge_collection = LineCollection(edge_pos,
                                         colors=edge_colors,
                                         linewidths=lw,
                                         antialiaseds=(1,),
                                         linestyle=style,
                                         transOffset=ax.transData,
                                         )

        edge_collection.set_zorder(1)  # edges go behind nodes
        edge_collection.set_label(label)
        ax.add_collection(edge_collection)

        # Note: there was a bug in mpl regarding the handling of alpha values
        # for each line in a LineCollection. It was fixed in matplotlib by
        # r7184 and r7189 (June 6 2009). We should then not set the alpha
        # value globally, since the user can instead provide per-edge alphas
        # now.  Only set it globally if provided as a scalar.
        if cb.is_numlike(alpha):
            edge_collection.set_alpha(alpha)

        if edge_colors is None:
            if edge_cmap is not None:
                assert(isinstance(edge_cmap, Colormap))
            edge_collection.set_array(np.asarray(edge_color))
            edge_collection.set_cmap(edge_cmap)
            if edge_vmin is not None or edge_vmax is not None:
                edge_collection.set_clim(edge_vmin, edge_vmax)
            else:
                edge_collection.autoscale()
        return edge_collection

    arrow_collection = None

    if G.is_directed() and arrows:
        # Note: Waiting for someone to implement arrow to intersection with
        # marker.  Meanwhile, this works well for polygons with more than 4
        # sides and circle.

        def to_marker_edge(marker_size, marker):
            if marker in "s^>v<d":  # `large` markers need extra space
                return np.sqrt(2 * marker_size) / 2
            else:
                return np.sqrt(marker_size) / 2

        # Draw arrows with `matplotlib.patches.FancyarrowPatch`
        arrow_collection = []
        mutation_scale = arrowsize  # scale factor of arrow head
        arrow_colors = edge_colors
        if arrow_colors is None:
            if edge_cmap is not None:
                assert(isinstance(edge_cmap, Colormap))
            else:
                edge_cmap = plt.get_cmap()  # default matplotlib colormap
            if edge_vmin is None:
                edge_vmin = min(edge_color)
            if edge_vmax is None:
                edge_vmax = max(edge_color)
            color_normal = Normalize(vmin=edge_vmin, vmax=edge_vmax)

        for i, (src, dst) in enumerate(edge_pos):
            x1, y1 = src
            x2, y2 = dst
            arrow_color = None
            line_width = None
            shrink_source = 0  # space from source to tail
            shrink_target = 0  # space from  head to target
            if cb.iterable(node_size):  # many node sizes
                src_node, dst_node = edgelist[i]
                index_node = nodelist.index(dst_node)
                marker_size = node_size[index_node]
                shrink_target = to_marker_edge(marker_size, node_shape)
            else:
                shrink_target = to_marker_edge(node_size, node_shape)
            if arrow_colors is None:
                arrow_color = edge_cmap(color_normal(edge_color[i]))
            elif len(arrow_colors) > 1:
                arrow_color = arrow_colors[i]
            else:
                arrow_color = arrow_colors[0]
            if len(lw) > 1:
                line_width = lw[i]
            else:
                line_width = lw[0]
            arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                    arrowstyle=arrowstyle,
                                    shrinkA=shrink_source,
                                    shrinkB=shrink_target,
                                    mutation_scale=mutation_scale,
                                    connectionstyle=connectionstyle,
                                    color=arrow_color,
                                    linewidth=line_width,
                                    zorder=1)  # arrows go behind nodes

            # There seems to be a bug in matplotlib to make collections of
            # FancyArrowPatch instances. Until fixed, the patches are added
            # individually to the axes instance.
            arrow_collection.append(arrow)
            ax.add_patch(arrow)

    # update view
    minx = np.amin(np.ravel(edge_pos[:, :, 0]))
    maxx = np.amax(np.ravel(edge_pos[:, :, 0]))
    miny = np.amin(np.ravel(edge_pos[:, :, 1]))
    maxy = np.amax(np.ravel(edge_pos[:, :, 1]))

    w = maxx - minx
    h = maxy - miny
    padx,  pady = 0.05 * w, 0.05 * h
    corners = (minx - padx, miny - pady), (maxx + padx, maxy + pady)
    ax.update_datalim(corners)
    ax.autoscale_view()

    return arrow_collection


def draw_g(graph):
    g=graph.g.to_networkx()
    fig=plt.figure(figsize=(8,4),dpi=150)
    ax=fig.subplots()
    ax.axis('off')
    ax.set_ylim(-1,1.5)
    en_indx=graph.nids['enc'].tolist()
    de_indx=graph.nids['dec'].tolist()
    en_l={i:np.array([i,0]) for i in en_indx}
    de_l={i:np.array([i+2,1]) for i in de_indx}
    en_de_s=[]
    for i in en_indx:
        for j in de_indx:
            en_de_s.append((i,j))
            g.add_edge(i,j)
    en_s=[]
    for i in en_indx:
        for j in en_indx:
            g.add_edge(i,j)
            en_s.append((i,j))

    de_s=[]
    for idx,i in enumerate(de_indx):
        for j in de_indx[idx:]:
            g.add_edge(i,j)
            de_s.append((i,j))

    nx.draw_networkx_nodes(g, en_l, nodelist=en_indx, node_color='r', node_size=60, ax=ax)
    nx.draw_networkx_nodes(g, de_l, nodelist=de_indx, node_color='r', node_size=60, ax=ax)
    draw_networkx_edges(g,en_l,edgelist=en_s, ax=ax,connectionstyle="arc3,rad=-0.3",width=0.5)
    draw_networkx_edges(g,de_l,edgelist=de_s, ax=ax,connectionstyle="arc3,rad=-0.3",width=0.5)
    draw_networkx_edges(g,{**en_l,**de_l},edgelist=en_de_s,width=0.3, ax=ax)
    # ax.add_patch()
    ax.text(len(en_indx)+0.5,0,"Encoder", verticalalignment='center', horizontalalignment='left')

    ax.text(len(en_indx)+0.5,1,"Decoder", verticalalignment='center', horizontalalignment='right')
    delta=0.03
    for value in {**en_l,**de_l}.values():
        x,y=value
        ax.add_patch(FancyArrowPatch((x-delta,y+delta),(x-delta,y-delta),arrowstyle="->",mutation_scale=8,connectionstyle="arc3,rad=3"))
    plt.show(fig)


class LabelSmoothing(nn.Module):
    """
    Computer loss at one time step.
    """
    def __init__(self, size, padding_idx, smoothing=0.0):
        """Label Smoothing module
        args:
            size: vocab_size
            padding_idx: index for symbol `padding`
            smoothing: smoothing ratio
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.size = size
        self.padding_idx = padding_idx
        self.smoothing = smoothing

    def forward(self, x, target):
        # x: (*, n_classes)
        # target: (*)
        assert x.size(1) == self.size
        with T.no_grad():
            tgt_dist = T.zeros_like(x, dtype=T.float)
            tgt_dist.fill_(self.smoothing / (self.size - 2)) # one for padding, another for label
            tgt_dist[:, self.padding_idx] = 0
            tgt_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)

            mask = T.nonzero(target == self.padding_idx)
            if mask.shape[0] > 0:
                tgt_dist.index_fill_(0, mask.squeeze(), 0)

        return self.criterion(x, tgt_dist)


class SimpleLossCompute(nn.Module):
    eps=1e-8
    def __init__(self, criterion, grad_accum, opt=None):
        """Loss function and optimizer for single device
        Parameters
        ----------
        criterion: torch.nn.Module
            criterion to compute loss
        grad_accum: int
            number of batches to accumulate gradients
        opt: Optimizer
            Model optimizer to use. If None, then no backward and update will be
            performed
        """
        super(SimpleLossCompute, self).__init__()
        self.criterion = criterion
        self.opt = opt
        self.acc_loss = 0
        self.n_correct = 0
        self.norm_term = 0
        self.loss = 0
        self.batch_count = 0
        self.grad_accum = grad_accum

    def __enter__(self):
        self.batch_count = 0

    def __exit__(self, type, value, traceback):
        # if not enough batches accumulated and there are gradients not applied,
        # do one more step
        if self.batch_count > 0:
            self.step()

    @property
    def avg_loss(self):
        return (self.acc_loss + self.eps) / (self.norm_term + self.eps)

    @property
    def accuracy(self):
        return (self.n_correct + self.eps) / (self.norm_term + self.eps)

    def step(self):
        self.opt.step()
        self.opt.optimizer.zero_grad()

    def backward_and_step(self):
        self.loss.backward()
        self.batch_count += 1
        # accumulate self.grad_accum times then synchronize and update
        if self.batch_count == self.grad_accum:
            self.step()
            self.batch_count = 0

    def __call__(self, y_pred, y, norm):
        y_pred = y_pred.contiguous().view(-1, y_pred.shape[-1])
        y = y.contiguous().view(-1)
        self.loss = self.criterion(y_pred, y) / norm
        if self.opt is not None:
            self.backward_and_step()
        self.n_correct += ((y_pred.max(dim=-1)[1] == y) & (y != self.criterion.padding_idx)).sum().item()
        self.acc_loss += self.loss.item() * norm
        self.norm_term += norm
        return self.loss.item() * norm


class MultiGPULossCompute(SimpleLossCompute):
    def __init__(self, criterion, ndev, grad_accum, model, opt=None):
        """Loss function and optimizer for multiple devices
        Parameters
        ----------
        criterion: torch.nn.Module
            criterion to compute loss
        ndev: int
            number of devices used
        grad_accum: int
            number of batches to accumulate gradients
        model: torch.nn.Module
            model to optimizer (needed to iterate and synchronize all parameters)
        opt: Optimizer
            Model optimizer to use. If None, then no backward and update will be
            performed
        """
        super(MultiGPULossCompute, self).__init__(criterion, grad_accum, opt=opt)
        self.ndev = ndev
        self.model = model

    def step(self):
        # multi-gpu synchronize gradients
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.ndev
        self.opt.step()
        self.opt.optimizer.zero_grad()


class Generator(nn.Module):
    '''
    Generate next token from the representation. This part is separated from the decoder, mostly for the convenience of sharing weight between embedding and generator.
    log(softmax(Wx + b))
    '''
    def __init__(self, dim_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(dim_model, vocab_size)

    def forward(self, x):
        return T.log_softmax(
            self.proj(x), dim=-1
        )


class SubLayerWrapper(nn.Module):
    '''
    The module wraps normalization, dropout, residual connection into one equation:
    sublayerwrapper(sublayer)(x) = x + dropout(sublayer(norm(x)))
    '''
    def __init__(self, size, dropout):
        super(SubLayerWrapper, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    '''
    This module implements feed-forward network(after the Multi-Head Network) equation:
    FFN(x) = max(0, x @ W_1 + b_1) @ W_2 + b_2
    '''
    def __init__(self, dim_model, dim_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim_model, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(T.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn # (key, query, value, mask)
        self.feed_forward = feed_forward
        self.sublayer = clones(SubLayerWrapper(size, dropout), 2)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn # (key, query, value, mask)
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SubLayerWrapper(size, dropout), 3)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def pre_func(self, i, fields='qkv'):
        layer = self.layers[i]
        def func(nodes):
            x = nodes.data['x']
            norm_x = layer.sublayer[0].norm(x)
            return layer.self_attn.get(norm_x, fields=fields)
        return func

    def post_func(self, i):
        layer = self.layers[i]
        def func(nodes):
            x, wv, z = nodes.data['x'], nodes.data['wv'], nodes.data['z']
            o = layer.self_attn.get_o(wv / z)
            x = x + layer.sublayer[0].dropout(o)
            x = layer.sublayer[1](x, layer.feed_forward)
            return {'x': x if i < self.N - 1 else self.norm(x)}
        return func


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.N = N
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def pre_func(self, i, fields='qkv', l=0):
        layer = self.layers[i]
        def func(nodes):
            x = nodes.data['x']
            norm_x = layer.sublayer[l].norm(x) if fields.startswith('q') else x
            if fields != 'qkv':
                return layer.src_attn.get(norm_x, fields)
            else:
                return layer.self_attn.get(norm_x, fields)
        return func

    def post_func(self, i, l=0):
        layer = self.layers[i]
        def func(nodes):
            x, wv, z = nodes.data['x'], nodes.data['wv'], nodes.data['z']
            o = layer.self_attn.get_o(wv / z)
            x = x + layer.sublayer[l].dropout(o)
            if l == 1:
                x = layer.sublayer[2](x, layer.feed_forward)
            return {'x': x if i < self.N - 1 else self.norm(x)}
        return func


class UEncoder(nn.Module):
    def __init__(self, layer):
        super(UEncoder, self).__init__()
        self.layer = layer
        self.norm = LayerNorm(layer.size)

    def pre_func(self, fields='qkv'):
        layer = self.layer
        def func(nodes):
            x = nodes.data['x']
            norm_x = layer.sublayer[0].norm(x)
            return layer.self_attn.get(norm_x, fields=fields)
        return func

    def post_func(self):
        layer = self.layer
        def func(nodes):
            x, wv, z = nodes.data['x'], nodes.data['wv'], nodes.data['z']
            o = layer.self_attn.get_o(wv / z)
            x = x + layer.sublayer[0].dropout(o)
            x = layer.sublayer[1](x, layer.feed_forward)
            return {'x': x}
        return func


class UDecoder(nn.Module):
    def __init__(self, layer):
        super(UDecoder, self).__init__()
        self.layer = layer
        self.norm = LayerNorm(layer.size)

    def pre_func(self, fields='qkv', l=0):
        layer = self.layer
        def func(nodes):
            x = nodes.data['x']
            if fields == 'kv':
                norm_x = x
            else:
                norm_x = layer.sublayer[l].norm(x)
            return layer.self_attn.get(norm_x, fields)
        return func

    def post_func(self, l=0):
        layer = self.layer
        def func(nodes):
            x, wv, z = nodes.data['x'], nodes.data['wv'], nodes.data['z']
            o = layer.self_attn.get_o(wv / z)
            x = x + layer.sublayer[l].dropout(o)
            if l == 1:
                x = layer.sublayer[2](x, layer.feed_forward)
            return {'x': x}
        return func


class HaltingUnit(nn.Module):
    halting_bias_init = 1.0
    def __init__(self, dim_model):
        super(HaltingUnit, self).__init__()
        self.linear = nn.Linear(dim_model, 1)
        self.norm = LayerNorm(dim_model)
        INIT.constant_(self.linear.bias, self.halting_bias_init)

    def forward(self, x):
        return T.sigmoid(self.linear(self.norm(x)))


class UTransformer(nn.Module):
    "Universal Transformer(https://arxiv.org/pdf/1807.03819.pdf) with ACT(https://arxiv.org/pdf/1603.08983.pdf)."
    MAX_DEPTH = 8
    thres = 0.99
    act_loss_weight = 0.01
    def __init__(self, encoder, decoder, src_embed, tgt_embed, pos_enc, time_enc, generator, h, d_k):
        super(UTransformer, self).__init__()
        self.encoder,  self.decoder = encoder, decoder
        self.src_embed, self.tgt_embed = src_embed, tgt_embed
        self.pos_enc, self.time_enc = pos_enc, time_enc
        self.halt_enc = HaltingUnit(h * d_k)
        self.halt_dec = HaltingUnit(h * d_k)
        self.generator = generator
        self.h, self.d_k = h, d_k
        self.reset_stat()

    def reset_stat(self):
        self.stat = [0] * (self.MAX_DEPTH + 1)

    def step_forward(self, nodes):
        x = nodes.data['x']
        step = nodes.data['step']
        pos = nodes.data['pos']
        return {'x': self.pos_enc.dropout(x + self.pos_enc(pos.view(-1)) + self.time_enc(step.view(-1))),
                'step': step + 1}

    def halt_and_accum(self, name, end=False):
        "field: 'enc' or 'dec'"
        halt = self.halt_enc if name == 'enc' else self.halt_dec
        thres = self.thres
        def func(nodes):
            p = halt(nodes.data['x'])
            sum_p = nodes.data['sum_p'] + p
            active = (sum_p < thres) & (1 - end)
            _continue = active.float()
            r = nodes.data['r'] * (1 - _continue) + (1 - sum_p) * _continue
            s = nodes.data['s'] + ((1 - _continue) * r + _continue * p) * nodes.data['x']
            return {'p': p, 'sum_p': sum_p, 'r': r, 's': s, 'active': active}
        return func

    def propagate_attention(self, g, eids):
        # Compute attention score
        g.apply_edges(src_dot_dst('k', 'q', 'score'), eids)
        g.apply_edges(scaled_exp('score', np.sqrt(self.d_k)), eids)
        # Send weighted values to target nodes
        g.send_and_recv(eids,
                        [fn.src_mul_edge('v', 'score', 'v'), fn.copy_edge('score', 'score')],
                        [fn.sum('v', 'wv'), fn.sum('score', 'z')])

    def update_graph(self, g, eids, pre_pairs, post_pairs):
        "Update the node states and edge states of the graph."
        # Pre-compute queries and key-value pairs.
        for pre_func, nids in pre_pairs:
            g.apply_nodes(pre_func, nids)
        self.propagate_attention(g, eids)
        # Further calculation after attention mechanism
        for post_func, nids in post_pairs:
            g.apply_nodes(post_func, nids)

    def forward(self, graph):
        g = graph.g
        N, E = graph.n_nodes, graph.n_edges
        nids, eids = graph.nids, graph.eids

        # embed & pos
        g.nodes[nids['enc']].data['x'] = self.src_embed(graph.src[0])
        g.nodes[nids['dec']].data['x'] = self.tgt_embed(graph.tgt[0])
        g.nodes[nids['enc']].data['pos'] = graph.src[1]
        g.nodes[nids['dec']].data['pos'] = graph.tgt[1]

        # init step
        device = next(self.parameters()).device
        g.ndata['s'] = T.zeros(N, self.h * self.d_k, dtype=T.float, device=device)    # accumulated state
        g.ndata['p'] = T.zeros(N, 1, dtype=T.float, device=device)                    # halting prob
        g.ndata['r'] = T.ones(N, 1, dtype=T.float, device=device)                     # remainder
        g.ndata['sum_p'] = T.zeros(N, 1, dtype=T.float, device=device)                # sum of pondering values
        g.ndata['step'] = T.zeros(N, 1, dtype=T.long, device=device)                  # step
        g.ndata['active'] = T.ones(N, 1, dtype=T.uint8, device=device)                # active

        for step in range(self.MAX_DEPTH):
            pre_func = self.encoder.pre_func('qkv')
            post_func = self.encoder.post_func()
            nodes = g.filter_nodes(lambda v: v.data['active'].view(-1), nids['enc'])
            if len(nodes) == 0: break
            edges = g.filter_edges(lambda e: e.dst['active'].view(-1), eids['ee'])
            end = step == self.MAX_DEPTH - 1
            self.update_graph(g, edges,
                              [(self.step_forward, nodes), (pre_func, nodes)],
                              [(post_func, nodes), (self.halt_and_accum('enc', end), nodes)])

        g.nodes[nids['enc']].data['x'] = self.encoder.norm(g.nodes[nids['enc']].data['s'])

        for step in range(self.MAX_DEPTH):
            pre_func = self.decoder.pre_func('qkv')
            post_func = self.decoder.post_func()
            nodes = g.filter_nodes(lambda v: v.data['active'].view(-1), nids['dec'])
            if len(nodes) == 0: break
            edges = g.filter_edges(lambda e: e.dst['active'].view(-1), eids['dd'])
            self.update_graph(g, edges,
                              [(self.step_forward, nodes), (pre_func, nodes)],
                              [(post_func, nodes)])

            pre_q = self.decoder.pre_func('q', 1)
            pre_kv = self.decoder.pre_func('kv', 1)
            post_func = self.decoder.post_func(1)
            nodes_e = nids['enc']
            edges = g.filter_edges(lambda e: e.dst['active'].view(-1), eids['ed'])
            end = step == self.MAX_DEPTH - 1
            self.update_graph(g, edges,
                              [(pre_q, nodes), (pre_kv, nodes_e)],
                              [(post_func, nodes), (self.halt_and_accum('dec', end), nodes)])

        g.nodes[nids['dec']].data['x'] = self.decoder.norm(g.nodes[nids['dec']].data['s'])
        act_loss = T.mean(g.ndata['r']) # ACT loss

        self.stat[0] += N
        for step in range(1, self.MAX_DEPTH + 1):
            self.stat[step] += T.sum(g.ndata['step'] >= step).item()

        return self.generator(g.ndata['x'][nids['dec']]), act_loss * self.act_loss_weight

    def infer(self, *args, **kwargs):
        raise NotImplementedError


def make_universal_model(src_vocab, tgt_vocab, dim_model=512, dim_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(h, dim_model)
    ff = PositionwiseFeedForward(dim_model, dim_ff)
    pos_enc = PositionalEncoding(dim_model, dropout)
    time_enc = PositionalEncoding(dim_model, dropout)
    encoder = UEncoder(EncoderLayer((dim_model), c(attn), c(ff), dropout))
    decoder = UDecoder(DecoderLayer((dim_model), c(attn), c(attn), c(ff), dropout))
    src_embed = Embeddings(src_vocab, dim_model)
    tgt_embed = Embeddings(tgt_vocab, dim_model)
    generator = Generator(dim_model, tgt_vocab)
    model = UTransformer(
        encoder, decoder, src_embed, tgt_embed, pos_enc, time_enc, generator, h, dim_model // h)
    # xavier init
    for p in model.parameters():
        if p.dim() > 1:
            INIT.xavier_uniform_(p)
    return model


class MultiHeadAttention(nn.Module):
    "Multi-Head Attention"
    def __init__(self, h, dim_model):
        "h: number of heads; dim_model: hidden dimension"
        super(MultiHeadAttention, self).__init__()
        self.d_k = dim_model // h
        self.h = h
        # W_q, W_k, W_v, W_o
        self.linears = clones(nn.Linear(dim_model, dim_model), 4)

    def get(self, x, fields='qkv'):
        "Return a dict of queries / keys / values."
        batch_size = x.shape[0]
        ret = {}
        if 'q' in fields:
            ret['q'] = self.linears[0](x).view(batch_size, self.h, self.d_k)
        if 'k' in fields:
            ret['k'] = self.linears[1](x).view(batch_size, self.h, self.d_k)
        if 'v' in fields:
            ret['v'] = self.linears[2](x).view(batch_size, self.h, self.d_k)
        return ret

    def get_o(self, x):
        "get output of the multi-head attention"
        batch_size = x.shape[0]
        return self.linears[3](x.view(batch_size, -1))


class PositionalEncoding(nn.Module):
    "Position Encoding module"
    def __init__(self, dim_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = T.zeros(max_len, dim_model, dtype=T.float)
        position = T.arange(0, max_len, dtype=T.float).unsqueeze(1)
        div_term = T.exp(T.arange(0, dim_model, 2, dtype=T.float) *
                             -(np.log(10000.0) / dim_model))
        pe[:, 0::2] = T.sin(position * div_term)
        pe[:, 1::2] = T.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # Not a parameter but should be in state_dict

    def forward(self, pos):
        return T.index_select(self.pe, 1, pos).squeeze(0)


class Embeddings(nn.Module):
    "Word Embedding module"
    def __init__(self, vocab_size, dim_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, dim_model)
        self.dim_model = dim_model

    def forward(self, x):
        return self.lut(x) * np.sqrt(self.dim_model)


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, pos_enc, generator, h, d_k):
        super(Transformer, self).__init__()
        self.encoder,  self.decoder = encoder, decoder
        self.src_embed, self.tgt_embed = src_embed, tgt_embed
        self.pos_enc = pos_enc
        self.generator = generator
        self.h, self.d_k = h, d_k
        self.att_weight_map = None

    def propagate_attention(self, g, eids):
        # Compute attention score
        g.apply_edges(src_dot_dst('k', 'q', 'score'), eids)
        g.apply_edges(scaled_exp('score', np.sqrt(self.d_k)), eids)
        # Send weighted values to target nodes
        g.send_and_recv(eids,
                        [fn.src_mul_edge('v', 'score', 'v'), fn.copy_edge('score', 'score')],
                        [fn.sum('v', 'wv'), fn.sum('score', 'z')])


    def update_graph(self, g, eids, pre_pairs, post_pairs):
        "Update the node states and edge states of the graph."

        # Pre-compute queries and key-value pairs.
        for pre_func, nids in pre_pairs:
            g.apply_nodes(pre_func, nids)
        self.propagate_attention(g, eids)
        # Further calculation after attention mechanism
        for post_func, nids in post_pairs:
            g.apply_nodes(post_func, nids)


    def forward(self, graph):
        g = graph.g
        nids, eids = graph.nids, graph.eids

        # embed
        src_embed, src_pos = self.src_embed(graph.src[0]), self.pos_enc(graph.src[1])
        tgt_embed, tgt_pos = self.tgt_embed(graph.tgt[0]), self.pos_enc(graph.tgt[1])
        g.nodes[nids['enc']].data['x'] = self.pos_enc.dropout(src_embed + src_pos)
        g.nodes[nids['dec']].data['x'] = self.pos_enc.dropout(tgt_embed + tgt_pos)

        for i in range(self.encoder.N):
            pre_func = self.encoder.pre_func(i, 'qkv')
            post_func = self.encoder.post_func(i)
            nodes, edges = nids['enc'], eids['ee']
            self.update_graph(g, edges, [(pre_func, nodes)], [(post_func, nodes)])

        for i in range(self.decoder.N):
            pre_func = self.decoder.pre_func(i, 'qkv')
            post_func = self.decoder.post_func(i)
            nodes, edges = nids['dec'], eids['dd']
            self.update_graph(g, edges, [(pre_func, nodes)], [(post_func, nodes)])
            pre_q = self.decoder.pre_func(i, 'q', 1)
            pre_kv = self.decoder.pre_func(i, 'kv', 1)
            post_func = self.decoder.post_func(i, 1)
            nodes_e, edges = nids['enc'], eids['ed']
            self.update_graph(g, edges, [(pre_q, nodes), (pre_kv, nodes_e)], [(post_func, nodes)])

        # visualize attention

        if self.att_weight_map is None:
            self._register_att_map(g, graph.nid_arr['enc'][VIZ_IDX], graph.nid_arr['dec'][VIZ_IDX])


        return self.generator(g.ndata['x'][nids['dec']])


    def infer(self, graph, max_len, eos_id, k, alpha=1.0):
        '''
        This function implements Beam Search in DGL, which is required in inference phase.
        Length normalization is given by (5 + len) ^ alpha / 6 ^ alpha. Please refer to https://arxiv.org/pdf/1609.08144.pdf.
        args:
            graph: a `Graph` object defined in `dgl.contrib.transformer.graph`.
            max_len: the maximum length of decoding.
            eos_id: the index of end-of-sequence symbol.
            k: beam size
        return:
            ret: a list of index array correspond to the input sequence specified by `graph``.
        '''
        g = graph.g
        N, E = graph.n_nodes, graph.n_edges
        nids, eids = graph.nids, graph.eids

        # embed & pos
        src_embed = self.src_embed(graph.src[0])
        src_pos = self.pos_enc(graph.src[1])
        g.nodes[nids['enc']].data['pos'] = graph.src[1]
        g.nodes[nids['enc']].data['x'] = self.pos_enc.dropout(src_embed + src_pos)
        tgt_pos = self.pos_enc(graph.tgt[1])
        g.nodes[nids['dec']].data['pos'] = graph.tgt[1]

        # init mask
        device = next(self.parameters()).device
        g.ndata['mask'] = T.zeros(N, dtype=T.uint8, device=device)

        # encode
        for i in range(self.encoder.N):
            pre_func = self.encoder.pre_func(i, 'qkv')
            post_func = self.encoder.post_func(i)
            nodes, edges = nids['enc'], eids['ee']
            self.update_graph(g, edges, [(pre_func, nodes)], [(post_func, nodes)])

        # decode
        log_prob = None
        y = graph.tgt[0]
        for step in range(1, max_len):
            y = y.view(-1)
            tgt_embed = self.tgt_embed(y)
            g.ndata['x'][nids['dec']] = self.pos_enc.dropout(tgt_embed + tgt_pos)
            edges_ed = g.filter_edges(lambda e: (e.dst['pos'] < step) & ~e.dst['mask'] , eids['ed'])
            edges_dd = g.filter_edges(lambda e: (e.dst['pos'] < step) & ~e.dst['mask'], eids['dd'])
            nodes_d = g.filter_nodes(lambda v: (v.data['pos'] < step) & ~v.data['mask'], nids['dec'])
            for i in range(self.decoder.N):
                pre_func, post_func = self.decoder.pre_func(i, 'qkv'), self.decoder.post_func(i)
                nodes, edges = nodes_d, edges_dd
                self.update_graph(g, edges, [(pre_func, nodes)], [(post_func, nodes)])
                pre_q, pre_kv = self.decoder.pre_func(i, 'q', 1), self.decoder.pre_func(i, 'kv', 1)
                post_func = self.decoder.post_func(i, 1)
                nodes_e, nodes_d, edges = nids['enc'], nodes_d, edges_ed
                self.update_graph(g, edges, [(pre_q, nodes_d), (pre_kv, nodes_e)], [(post_func, nodes_d)])

            frontiers = g.filter_nodes(lambda v: v.data['pos'] == step - 1, nids['dec'])
            out = self.generator(g.ndata['x'][frontiers])
            batch_size = frontiers.shape[0] // k
            vocab_size = out.shape[-1]
            # Mask output for complete sequence
            one_hot = T.zeros(vocab_size).fill_(-1e9).to(device)
            one_hot[eos_id] = 0
            mask = g.ndata['mask'][frontiers].unsqueeze(-1).float()
            out = out * (1 - mask) + one_hot.unsqueeze(0) * mask

            if log_prob is None:
                log_prob, pos = out.view(batch_size, k, -1)[:, 0, :].topk(k, dim=-1)
                eos = T.zeros(batch_size, k).byte()
            else:
                norm_old = eos.float().to(device) + (1 - eos.float().to(device)) * np.power((4. + step) / 6, alpha)
                norm_new = eos.float().to(device) + (1 - eos.float().to(device)) * np.power((5. + step) / 6, alpha)
                log_prob, pos = ((out.view(batch_size, k, -1) + (log_prob * norm_old).unsqueeze(-1)) / norm_new.unsqueeze(-1)).view(batch_size, -1).topk(k, dim=-1)

            _y = y.view(batch_size * k, -1)
            y = T.zeros_like(_y)
            _eos = eos.clone()
            for i in range(batch_size):
                for j in range(k):
                    _j = pos[i, j].item() // vocab_size
                    token = pos[i, j].item() % vocab_size
                    y[i*k+j, :] = _y[i*k+_j, :]
                    y[i*k+j, step] = token
                    eos[i, j] = _eos[i, _j] | (token == eos_id)

            if eos.all():
                break
            else:
                g.ndata['mask'][nids['dec']] = eos.unsqueeze(-1).repeat(1, 1, max_len).view(-1).to(device)
        return y.view(batch_size, k, -1)[:, 0, :].tolist()


    def _register_att_map(self, g, enc_ids, dec_ids):
        self.att_weight_map = [
            get_attention_map(g, enc_ids, enc_ids, self.h),
            get_attention_map(g, enc_ids, dec_ids, self.h),
            get_attention_map(g, dec_ids, dec_ids, self.h),
        ]


def make_model(src_vocab, tgt_vocab, N=6,
                   dim_model=512, dim_ff=2048, h=8, dropout=0.1, universal=False):
    if universal:
        return make_universal_model(src_vocab, tgt_vocab, dim_model, dim_ff, h, dropout)
    c = copy.deepcopy
    attn = MultiHeadAttention(h, dim_model)
    ff = PositionwiseFeedForward(dim_model, dim_ff)
    pos_enc = PositionalEncoding(dim_model, dropout)

    encoder = Encoder(EncoderLayer(dim_model, c(attn), c(ff), dropout), N)
    decoder = Decoder(DecoderLayer(dim_model, c(attn), c(attn), c(ff), dropout), N)
    src_embed = Embeddings(src_vocab, dim_model)
    tgt_embed = Embeddings(tgt_vocab, dim_model)
    generator = Generator(dim_model, tgt_vocab)
    model = Transformer(
        encoder, decoder, src_embed, tgt_embed, pos_enc, generator, h, dim_model // h)
    # xavier init
    for p in model.parameters():
        if p.dim() > 1:
            INIT.xavier_uniform_(p)
    return model


class NoamOpt(object):
    def __init__(self, model_size, factor, warmup, optimizer):
        """
        model_size: hidden size
        factor: coefficient
        warmup: warm up steps(step ** (-0.5) == step * warmup ** (-1.5) holds when warmup equals step)
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5))
            )

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()


class Vocab:
    def __init__(self, init_token=None, eos_token=None, pad_token=None, unk_token=None):
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.vocab_lst = []
        self.vocab_dict = None

    def load(self, path):
        if self.init_token is not None:
            self.vocab_lst.append(self.init_token)
        if self.eos_token is not None:
            self.vocab_lst.append(self.eos_token)
        if self.pad_token is not None:
            self.vocab_lst.append(self.pad_token)
        if self.unk_token is not None:
            self.vocab_lst.append(self.unk_token)
        with open(path, 'r', encoding='utf-8') as f:
            for token in f.readlines():
                token = token.strip()
                self.vocab_lst.append(token)
        self.vocab_dict = {
            v: k for k, v in enumerate(self.vocab_lst)
        }

    def __len__(self):
        return len(self.vocab_lst)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self.vocab_dict:
                return self.vocab_dict[key]
            else:
                return self.vocab_dict[self.unk_token]
        else:
            return self.vocab_lst[key]


class Field:
    def __init__(self, vocab, preprocessing=None, postprocessing=None):
        self.vocab = vocab
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def preprocess(self, x):
        if self.preprocessing is not None:
            return self.preprocessing(x)
        return x

    def postprocess(self, x):
        if self.postprocessing is not None:
            return self.postprocessing(x)
        return x

    def numericalize(self, x):
        return [self.vocab[token] for token in x]

    def __call__(self, x):
        return self.postprocess(
            self.numericalize(
                self.preprocess(x)
            )
        )


class TranslationDataset(object):
    '''
    Dataset class for translation task.
    By default, the source language shares the same vocabulary with the target language.
    '''
    INIT_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    PAD_TOKEN = '<pad>'
    MAX_LENGTH = 50
    def __init__(self, path, exts, train='train', valid='valid', test='test', vocab='vocab.txt', replace_oov=None):
        vocab_path = os.path.join(path, vocab)
        self.src = {}
        self.tgt = {}
        with open(os.path.join(path, train + '.' + exts[0]), 'r', encoding='utf-8') as f:
            self.src['train'] = f.readlines()
        with open(os.path.join(path, train + '.' + exts[1]), 'r', encoding='utf-8') as f:
            self.tgt['train'] = f.readlines()
        with open(os.path.join(path, valid + '.' + exts[0]), 'r', encoding='utf-8') as f:
            self.src['valid'] = f.readlines()
        with open(os.path.join(path, valid + '.' + exts[1]), 'r', encoding='utf-8') as f:
            self.tgt['valid'] = f.readlines()
        with open(os.path.join(path, test + '.' + exts[0]), 'r', encoding='utf-8') as f:
            self.src['test'] = f.readlines()
        with open(os.path.join(path, test + '.' + exts[1]), 'r', encoding='utf-8') as f:
            self.tgt['test'] = f.readlines()

        if not os.path.exists(vocab_path):
            self._make_vocab(vocab_path)

        vocab = Vocab(init_token=self.INIT_TOKEN,
                      eos_token=self.EOS_TOKEN,
                      pad_token=self.PAD_TOKEN,
                      unk_token=replace_oov)
        vocab.load(vocab_path)
        self.vocab = vocab
        strip_func = lambda x: x[:self.MAX_LENGTH]
        self.src_field = Field(vocab,
                               preprocessing=None,
                               postprocessing=strip_func)
        self.tgt_field = Field(vocab,
                               preprocessing=lambda seq: [self.INIT_TOKEN] + seq + [self.EOS_TOKEN],
                               postprocessing=strip_func)

    def get_seq_by_id(self, idx, mode='train', field='src'):
        "get raw sequence in dataset by specifying index, mode(train/valid/test), field(src/tgt)"
        if field == 'src':
            return self.src[mode][idx].strip().split()
        else:
            return [self.INIT_TOKEN] + self.tgt[mode][idx].strip().split() + [self.EOS_TOKEN]

    def _make_vocab(self, path, thres=2):
        word_dict = {}
        for mode in ['train', 'valid', 'test']:
            for line in self.src[mode] + self.tgt[mode]:
                for token in line.strip().split():
                    if token not in word_dict:
                        word_dict[token] = 0
                    else:
                        word_dict[token] += 1

        with open(path, 'w') as f:
            for k, v in word_dict.items():
                if v > 2:
                    print(k, file=f)

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def pad_id(self):
        return self.vocab[self.PAD_TOKEN]

    @property
    def sos_id(self):
        return self.vocab[self.INIT_TOKEN]

    @property
    def eos_id(self):
        return self.vocab[self.EOS_TOKEN]

    def __call__(self, graph_pool, mode='train', batch_size=32, k=1,
                 device='cpu', dev_rank=0, ndev=1):
        '''
        Create a batched graph correspond to the mini-batch of the dataset.
        args:
            graph_pool: a GraphPool object for accelerating.
            mode: train/valid/test
            batch_size: batch size
            k: beam size(only required for test)
            device: str or torch.device
            dev_rank: rank (id) of current device
            ndev: number of devices
        '''
        src_data, tgt_data = self.src[mode], self.tgt[mode]
        n = len(src_data)
        # make sure all devices have the same number of batch
        n = n // ndev * ndev

        # XXX: partition then shuffle may not be equivalent to shuffle then
        # partition
        order = list(range(dev_rank, n, ndev))
        if mode == 'train':
            random.shuffle(order)

        src_buf, tgt_buf = [], []

        for idx in order:
            src_sample = self.src_field(
                src_data[idx].strip().split())
            tgt_sample = self.tgt_field(
                tgt_data[idx].strip().split())
            src_buf.append(src_sample)
            tgt_buf.append(tgt_sample)
            if len(src_buf) == batch_size:
                if mode == 'test':
                    yield graph_pool.beam(src_buf, self.sos_id, self.MAX_LENGTH, k, device=device)
                else:
                    yield graph_pool(src_buf, tgt_buf, device=device)
                src_buf, tgt_buf = [], []

        if len(src_buf) != 0:
            if mode == 'test':
                yield graph_pool.beam(src_buf, self.sos_id, self.MAX_LENGTH, k, device=device)
            else:
                yield graph_pool(src_buf, tgt_buf, device=device)


    def get_sequence(self, batch):
        "return a list of sequence from a list of index arrays"
        ret = []
        filter_list = set([self.pad_id, self.sos_id, self.eos_id])
        for seq in batch:
            try:
                l = seq.index(self.eos_id)
            except:
                l = len(seq)
            ret.append(' '.join(self.vocab[token] for token in seq[:l] if not token in filter_list))
        return ret


class GraphPool:
    "Create a graph pool in advance to accelerate graph building phase in Transformer."
    def __init__(self, n=50, m=50):
        '''
        args:
            n: maximum length of input sequence.
            m: maximum length of output sequence.
        '''
        print('start creating graph pool...')
        tic = time.time()
        self.n, self.m = n, m
        g_pool = [[DGLGraph() for _ in range(m)] for _ in range(n)]
        num_edges = {
            'ee': np.zeros((n, n)).astype(int),
            'ed': np.zeros((n, m)).astype(int),
            'dd': np.zeros((m, m)).astype(int)
        }
        for i, j in itertools.product(range(n), range(m)):
            src_length = i + 1
            tgt_length = j + 1

            g_pool[i][j].add_nodes(src_length + tgt_length)
            enc_nodes = T.arange(src_length, dtype=T.long)
            dec_nodes = T.arange(tgt_length, dtype=T.long) + src_length

            # enc -> enc
            us = enc_nodes.unsqueeze(-1).repeat(1, src_length).view(-1)
            vs = enc_nodes.repeat(src_length)
            g_pool[i][j].add_edges(us, vs)
            num_edges['ee'][i][j] = len(us)
            # enc -> dec
            us = enc_nodes.unsqueeze(-1).repeat(1, tgt_length).view(-1)
            vs = dec_nodes.repeat(src_length)
            g_pool[i][j].add_edges(us, vs)
            num_edges['ed'][i][j] = len(us)
            # dec -> dec
            indices = T.triu(T.ones(tgt_length, tgt_length)) == 1
            us = dec_nodes.unsqueeze(-1).repeat(1, tgt_length)[indices]
            vs = dec_nodes.unsqueeze(0).repeat(tgt_length, 1)[indices]
            g_pool[i][j].add_edges(us, vs)
            num_edges['dd'][i][j] = len(us)

        print('successfully created graph pool, time: {0:0.3f}s'.format(time.time() - tic))
        self.g_pool = g_pool
        self.num_edges = num_edges

    def beam(self, src_buf, start_sym, max_len, k, device='cpu'):
        '''
        Return a batched graph for beam search during inference of Transformer.
        args:
            src_buf: a list of input sequence
            start_sym: the index of start-of-sequence symbol
            max_len: maximum length for decoding
            k: beam size
            device: 'cpu' or 'cuda:*'
        '''
        g_list = []
        src_lens = [len(_) for _ in src_buf]
        tgt_lens = [max_len] * len(src_buf)
        num_edges = {'ee': [], 'ed': [], 'dd': []}
        for src_len, tgt_len in zip(src_lens, tgt_lens):
            i, j = src_len - 1, tgt_len - 1
            for _ in range(k):
                g_list.append(self.g_pool[i][j])
            for key in ['ee', 'ed', 'dd']:
                num_edges[key].append(int(self.num_edges[key][i][j]))

        g = batch(g_list)
        src, tgt = [], []
        src_pos, tgt_pos = [], []
        enc_ids, dec_ids = [], []
        e2e_eids, e2d_eids, d2d_eids = [], [], []
        n_nodes, n_edges, n_tokens = 0, 0, 0
        for src_sample, n, n_ee, n_ed, n_dd in zip(src_buf, src_lens, num_edges['ee'], num_edges['ed'], num_edges['dd']):
            for _ in range(k):
                src.append(T.tensor(src_sample, dtype=T.long, device=device))
                src_pos.append(T.arange(n, dtype=T.long, device=device))
                enc_ids.append(T.arange(n_nodes, n_nodes + n, dtype=T.long, device=device))
                n_nodes += n
                e2e_eids.append(T.arange(n_edges, n_edges + n_ee, dtype=T.long, device=device))
                n_edges += n_ee
                tgt_seq = T.zeros(max_len, dtype=T.long, device=device)
                tgt_seq[0] = start_sym
                tgt.append(tgt_seq)
                tgt_pos.append(T.arange(max_len, dtype=T.long, device=device))

                dec_ids.append(T.arange(n_nodes, n_nodes + max_len, dtype=T.long, device=device))
                n_nodes += max_len
                e2d_eids.append(T.arange(n_edges, n_edges + n_ed, dtype=T.long, device=device))
                n_edges += n_ed
                d2d_eids.append(T.arange(n_edges, n_edges + n_dd, dtype=T.long, device=device))
                n_edges += n_dd

        g.set_n_initializer(dgl_zero_initializer)
        g.set_e_initializer(dgl_zero_initializer)

        return Graph(g=g,
                     src=(T.cat(src), T.cat(src_pos)),
                     tgt=(T.cat(tgt), T.cat(tgt_pos)),
                     tgt_y=None,
                     nids = {'enc': T.cat(enc_ids), 'dec': T.cat(dec_ids)},
                     eids = {'ee': T.cat(e2e_eids), 'ed': T.cat(e2d_eids), 'dd': T.cat(d2d_eids)},
                     nid_arr = {'enc': enc_ids, 'dec': dec_ids},
                     n_nodes=n_nodes,
                     n_edges=n_edges,
                     n_tokens=n_tokens)

    def __call__(self, src_buf, tgt_buf, device='cpu'):
        '''
        Return a batched graph for the training phase of Transformer.
        args:
            src_buf: a set of input sequence arrays.
            tgt_buf: a set of output sequence arrays.
            device: 'cpu' or 'cuda:*'
        '''
        g_list = []
        src_lens = [len(_) for _ in src_buf]
        tgt_lens = [len(_) - 1 for _ in tgt_buf]
        num_edges = {'ee': [], 'ed': [], 'dd': []}
        for src_len, tgt_len in zip(src_lens, tgt_lens):
            i, j = src_len - 1, tgt_len - 1
            g_list.append(self.g_pool[i][j])
            for key in ['ee', 'ed', 'dd']:
                num_edges[key].append(int(self.num_edges[key][i][j]))

        g = batch(g_list)
        src, tgt, tgt_y = [], [], []
        src_pos, tgt_pos = [], []
        enc_ids, dec_ids = [], []
        e2e_eids, d2d_eids, e2d_eids = [], [], []
        n_nodes, n_edges, n_tokens = 0, 0, 0
        for src_sample, tgt_sample, n, m, n_ee, n_ed, n_dd in zip(src_buf, tgt_buf, src_lens, tgt_lens, num_edges['ee'], num_edges['ed'], num_edges['dd']):
            src.append(T.tensor(src_sample, dtype=T.long, device=device))
            tgt.append(T.tensor(tgt_sample[:-1], dtype=T.long, device=device))
            tgt_y.append(T.tensor(tgt_sample[1:], dtype=T.long, device=device))
            src_pos.append(T.arange(n, dtype=T.long, device=device))
            tgt_pos.append(T.arange(m, dtype=T.long, device=device))
            enc_ids.append(T.arange(n_nodes, n_nodes + n, dtype=T.long, device=device))
            n_nodes += n
            dec_ids.append(T.arange(n_nodes, n_nodes + m, dtype=T.long, device=device))
            n_nodes += m
            e2e_eids.append(T.arange(n_edges, n_edges + n_ee, dtype=T.long, device=device))
            n_edges += n_ee
            e2d_eids.append(T.arange(n_edges, n_edges + n_ed, dtype=T.long, device=device))
            n_edges += n_ed
            d2d_eids.append(T.arange(n_edges, n_edges + n_dd, dtype=T.long, device=device))
            n_edges += n_dd
            n_tokens += m


        g.set_n_initializer(dgl_zero_initializer)
        g.set_e_initializer(dgl_zero_initializer)

        return Graph(g=g,
                     src=(T.cat(src), T.cat(src_pos)),
                     tgt=(T.cat(tgt), T.cat(tgt_pos)),
                     tgt_y=T.cat(tgt_y),
                     nids = {'enc': T.cat(enc_ids), 'dec': T.cat(dec_ids)},
                     eids = {'ee': T.cat(e2e_eids), 'ed': T.cat(e2d_eids), 'dd': T.cat(d2d_eids)},
                     nid_arr = {'enc': enc_ids, 'dec': dec_ids},
                     n_nodes=n_nodes,
                     n_edges=n_edges,
                     n_tokens=n_tokens)




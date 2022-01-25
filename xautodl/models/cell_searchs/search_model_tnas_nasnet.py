####################
# DARTS, ICLR 2019 #
####################
import torch
import torch.nn as nn
from copy import deepcopy
from typing import List, Text, Dict
from .search_cells import NASNetSearchCell as SearchCell


# The macro structure is based on NASNet
class TNASNetworkDARTS(nn.Module):
    def __init__(
        self,
        C: int,
        N: int,
        steps: int,
        multiplier: int,
        stem_multiplier: int,
        num_classes: int,
        search_space: List[Text],
        affine: bool,
        track_running_stats: bool,
        train_arch_parameters=False
    ):
        super(TNASNetworkDARTS, self).__init__()
        self._C = C
        self._layerN = N
        self._steps = steps
        self._multiplier = multiplier
        self.stem = nn.Sequential(
            nn.Conv2d(3, C * stem_multiplier, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C * stem_multiplier),
        )

        # config for each layer
        layer_channels = (
            [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        )
        layer_reductions = (
            [False] * N + [True] + [False] * N + [True] + [False] * N
        )

        num_edge, edge2index = None, None
        C_prev_prev, C_prev, C_curr, reduction_prev = (
            C * stem_multiplier,
            C * stem_multiplier,
            C,
            False,
        )

        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(
            zip(layer_channels, layer_reductions)
        ):
            cell = SearchCell(
                search_space,
                steps,
                multiplier,
                C_prev_prev,
                C_prev,
                C_curr,
                reduction,
                reduction_prev,
                affine,
                track_running_stats,
            )
            if num_edge is None:
                num_edge, edge2index = cell.num_edges, cell.edge2index
            else:
                assert (
                    num_edge == cell.num_edges and edge2index == cell.edge2index
                ), "invalid {:} vs. {:}.".format(num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev_prev, C_prev, reduction_prev = C_prev, multiplier * C_curr, reduction
        self.op_names = deepcopy(search_space)
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        
        self.train_arch_parameters = train_arch_parameters
        if self.train_arch_parameters:
            self.arch_normal_parameters = nn.Parameter(
                1e-3 * torch.randn(num_edge, len(search_space))
            )
            self.arch_reduce_parameters = nn.Parameter(
                1e-3 * torch.randn(num_edge, len(search_space))
            )
            self.normalizer = 'softmax'
        else:
            self.arch_normal_parameters = torch.ones(num_edge, len(search_space), requires_grad=False)
            self.arch_reduce_parameters = torch.ones(num_edge, len(search_space), requires_grad=False)
            self.normalizer = 'avg'
            
    def get_weights(self) -> List[torch.nn.Parameter]:
        xlist = list(self.stem.parameters()) + list(self.cells.parameters())
        xlist += list(self.lastact.parameters()) + list(
            self.global_pooling.parameters()
        )
        xlist += list(self.classifier.parameters())
        return xlist

    def get_alphas(self) -> List[torch.nn.Parameter]:
        return [self.arch_normal_parameters, self.arch_reduce_parameters]

    def show_alphas(self) -> Text:
        with torch.no_grad():
            A = "arch-normal-parameters :\n{:}".format(
                self.arch_normal_parameters.cpu()
            )
            B = "arch-reduce-parameters :\n{:}".format(
                self.arch_reduce_parameters.cpu()
            )
        return "{:}\n{:}".format(A, B)

    def get_message(self) -> Text:
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += "\n {:02d}/{:02d} :: {:}".format(
                i, len(self.cells), cell.extra_repr()
            )
        return string

    def extra_repr(self) -> Text:
        return "{name}(C={_C}, N={_layerN}, steps={_steps}, multiplier={_multiplier}, L={_Layer})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def genotype(self) -> Dict[Text, List]:
        def _parse(weights, k=2):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                edges = weights[start:end].clone()
                edge_max, primitive_indices = torch.topk(edges[:, 1:], 1)
                topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)  # get the topk input states of each node
                node_gene = []
                for edge_idx in topk_edge_indices:
                    prim_idx = primitive_indices[edge_idx]+1
                    prim = self.op_names[prim_idx]
                    node_gene.append((prim, edge_idx.item()))
                start = end
                n += 1
                gene.append(tuple(node_gene))
            return gene

        with torch.no_grad():
            gene_normal = _parse(
                self.arch_normal_parameters
            )
            gene_reduce = _parse(
                self.arch_reduce_parameters
            )
        return {
            "normal": gene_normal,
            "normal_concat": list(
                range(2 + self._steps - self._multiplier, self._steps + 2)
            ),
            "reduce": gene_reduce,
            "reduce_concat": list(
                range(2 + self._steps - self._multiplier, self._steps + 2)
            ),
        }

    def normalize_archp(self,):
        if self.normalizer == 'softmax':
            normal_w = nn.functional.softmax(self.arch_normal_parameters, dim=1)
            reduce_w = nn.functional.softmax(self.arch_reduce_parameters, dim=1)   
        else:
            none_zero = torch.sum(self.arch_normal_parameters[:, 1:], dim=1, keepdim=True)  # weight non zero operations
            none_zero[none_zero == 0] = 1.
            normal_w = self.arch_normal_parameters / none_zero
            
            none_zero = torch.sum(self.arch_reduce_parameters[:, 1:], dim=1, keepdim=True)  # weight non zero operations
            none_zero[none_zero == 0] = 1.
            reduce_w = self.arch_reduce_parameters / none_zero
        return normal_w, reduce_w    

    def forward(self, inputs):
        normal_w, reduce_w = self.normalize_archp()

        s0 = s1 = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                ww = reduce_w
            else:
                ww = normal_w
            s0, s1 = s1, cell.forward_tnas(s0, s1, ww)
        out = self.lastact(s1)  # TODO: why need this last activation layer?
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return out, logits

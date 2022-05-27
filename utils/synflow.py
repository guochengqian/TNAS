# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import torch
import torch.nn as nn
import time
import numpy as np


def get_layer_metric_array(net, metric):
    metric_array = []
    op_names_array = []
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))
            op_names_array.append(name)
    return metric_array, op_names_array


def compute_synflow_per_weight(net, input_dim=[3, 32, 32], device=torch.device("cuda")):
    # convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    # convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    # keep signs of all params
    signs = linearize(net)

    # Compute gradients with input of 1s 
    net.zero_grad()
    net.double()
    inputs = torch.ones([1] + input_dim).double().to(device)
    output = net.forward(inputs)
    if isinstance(output, tuple):
        output = output[-1]
    torch.sum(output).backward()

    # select the gradients that we want to use for search/prune
    def synflow(layer):
        if layer.weight.grad is not None:
            return torch.abs(layer.weight.data * layer.weight.grad.data).sum()
        else:
            return torch.zeros_like(layer.weight).sum()

    grads_abs = get_layer_metric_array(net, synflow)  # synflow values for each layer.

    # apply signs of all params
    nonlinearize(net, signs)
    net.float()
    return grads_abs


def synflow_edge(model, n_node=4, input_size=32, input_channels=3, device=torch.device('cuda')):
    model = model.to(device)
    synflow_weight, op_names_array = compute_synflow_per_weight(model, [input_channels, input_size, input_size], device)
    edge_scores = np.zeros(len(model.edge2index))
    for node_i in range(1, n_node):
        for edge_j in range(node_i):
            node_str = "{:}<-{:}".format(node_i, edge_j)
            idx = [i for i, op_name in enumerate(op_names_array) if node_str in op_name]
            edge_scores[model.edge2index[node_str]] = sum(synflow_weight[i] for i in idx)
    return edge_scores


def synflow_model(model, input_size=32, input_channels=3, device=torch.device('cuda')):
    start_time = time.time()
    model = model.to(device)
    model_value = sum(compute_synflow_per_weight(model, [input_channels, input_size, input_size], device)[0])
    return model_value.data.cpu().numpy(), time.time() - start_time

#
# if __name__ == "__main__":
#     import utils
#     from config import SearchConfig
#     from models.search_cnn import SearchCNNController
#     from genotypes import N_TOPOLOGY, N_CONV, PRIM_GROUPS, PRIMITIVES
#
#     config = SearchConfig()
#     device = torch.device("cuda")
#
#     input_size, input_channels, n_classes = 32, 3, 10
#     # get data with meta info
#     # input_size, input_channels, n_classes, train_data = utils.get_data(
#     #     config.dataset, config.data_dir, cutout_length=0, validation=False)
#
#     # train_loader = torch.utils.data.DataLoader(train_data,
#     #                                            batch_size=config.batch_size,
#     #                                            shuffle=True,
#     #                                            num_workers=config.workers,
#     #                                            pin_memory=True)
#
#     model = SearchCNNController(input_channels, config.init_channels, n_classes, config.layers,
#                                 n_nodes=config.nodes, device_ids=config.gpus)
#     model = model.to(device)
#
#     # inputs = get_some_data(train_loader, device=device)
#     synflow_weight, op_names_array = compute_synflow_per_weight(model, [input_channels, input_size, input_size], device)
#
#     # # only calculate convolution and linear layer.
#     # print(op_names_array)
#     #
#     n_node = config.nodes
#     n_cell = config.layers
#
#     edge_scores = []
#     for node_i in range(n_node):
#         for edge_j in range(node_i + 2):
#             pattern = f'dag.{node_i}.{edge_j}'
#             idx = [i for i, op_name in enumerate(op_names_array) if pattern in op_name]
#             edge_scores.append(sum(synflow_weight[i] for i in idx))
#
#     print(edge_scores)

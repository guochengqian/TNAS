'''
Copyright 2021@Guocheng Qian
File Description: PyTorch Implementation of TNAS on NAS-BENCH-201 dataset
'''
######################################################################################
# python exps/NAS-Bench-201-algos/TNAS.py --cfg cfgs/search_darts/tnas.yaml
######################################################################################


import os, sys, time, random, argparse, json
import itertools
from collections import Iterable
import copy

import numpy as np
import torch, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from xautodl.config_utils import dict2config
from xautodl.datasets import get_datasets, get_nas_search_loaders
from xautodl.procedures import (
    get_optim_scheduler,
    prepare_logger,
    save_checkpoint
)
from xautodl.utils import count_parameters_in_MB, obtain_accuracy
from xautodl.log_utils import AverageMeter, time_string, convert_secs2time
from xautodl.models import get_cell_based_tiny_net, get_search_spaces

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import config, set_seed, Wandb, generate_exp_directory


NONE_ENCODING = [1, 0, 0, 0, 0, 0, 0, 0]


def train_func(
        train_loader,
        model,
        criterion,
        scheduler,
        w_optimizer,
        epoch_str,
        print_freq,
        logger,
):
    data_time, batch_time = AverageMeter(), AverageMeter()
    base_losses = AverageMeter()
    end = time.time()
    model.train()
    # four inputs: train, train_label, test, test_label
    for step, (base_inputs, base_targets) in enumerate(
            train_loader
    ):
        scheduler.update(None, 1.0 * step / len(train_loader))

        base_inputs = base_inputs.cuda(non_blocking=True)
        base_targets = base_targets.cuda(non_blocking=True)

        # measure data loading time
        data_time.update(time.time() - end)
        model.zero_grad()
        _, logits = model(base_inputs)
        base_loss = criterion(logits, base_targets)
        base_loss.backward()
        w_optimizer.step()
        base_losses.update(base_loss.item(), base_inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % print_freq == 0 or step + 1 == len(train_loader):
            Sstr = (
                    "*SEARCH* "
                    + " [{:}][{:03d}/{:03d}]".format(epoch_str, step, len(train_loader))
            )
            Tstr = "Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})".format(
                batch_time=batch_time, data_time=data_time
            )
            Wstr = "Base [Loss {loss.val:.3f} ({loss.avg:.3f}) ".format(
                loss=base_losses
            )
            strs = Sstr + " " + Tstr + " " + Wstr
            logger.log(strs)
    return base_losses.avg


def valid_func(xloader, model):
    data_time, batch_time = AverageMeter(), AverageMeter()
    arch_top1, arch_top5 = AverageMeter(), AverageMeter()
    end = time.time()
    with torch.no_grad():
        model.eval()
        for step, (arch_inputs, arch_targets) in enumerate(xloader):
            arch_targets = arch_targets.cuda(non_blocking=True)
            # measure data loading time
            data_time.update(time.time() - end)
            # prediction
            _, logits = model(arch_inputs.cuda(non_blocking=True))
            # record
            arch_prec1, arch_prec5 = obtain_accuracy(
                logits.data, arch_targets.data, topk=(1, 5)
            )
            arch_top1.update(arch_prec1.item(), arch_inputs.size(0))
            arch_top5.update(arch_prec5.item(), arch_inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    return arch_top1.avg, arch_top5.avg


def train_val_epochs(start_epoch, total_epoch, global_epoch,
                     train_loader, valid_loader,
                     model, model_idx,
                     criterion, w_scheduler, w_optimizer,
                     show_alpha=False,
                     enable_valid=True
                     ):
    best_a_top1 = 0.
    best_w_loss = np.inf
    for epoch in range(start_epoch, total_epoch):
        global_epoch += 1
        w_scheduler.update(epoch, 0.0)
        epoch_str = "{:03d}-{:03d}".format(epoch + 1, total_epoch)

        search_w_loss = train_func(train_loader, model, criterion,
                                   w_scheduler, w_optimizer,
                                   epoch_str, config.print_freq, logger,
                                   )
        strs = "[{:}] search [base] : loss={:.2f}".format(epoch_str, search_w_loss)
        if search_w_loss < best_w_loss:
            best_w_loss = search_w_loss

        # only validate at the last training epoch
        if enable_valid and epoch == total_epoch - 1:
            search_a_top1, _ = valid_func(valid_loader, model)
            if search_a_top1 > best_a_top1:
                best_a_top1 = search_a_top1
            strs += "search [arch] : accuracy@1={:.2f}%".format(search_a_top1)
            summary_writer.add_scalar(f'train/a_top1', search_a_top1, global_epoch)

        logger.log(strs)

        if show_alpha:
            with torch.no_grad():
                logger.log("{:}".format(model.show_alphas()))
        summary_writer.add_scalar(f'train/w_loss', search_w_loss, global_epoch)
        summary_writer.add_scalar(f'train/w_lr', w_scheduler.get_lr()[-1], global_epoch)
        summary_writer.add_scalar('train/subnet_idx', model_idx, global_epoch)
    return best_w_loss, best_a_top1


def check_model_valid(arch_parameters, edge2index, max_nodes=4):
    for i in range(1, max_nodes):
        none_flag = True
        for j in range(i):
            node_str = "{:}<-{:}".format(i, j)
            with torch.no_grad():
                weights = arch_parameters[edge2index[node_str]]
                none_flag = none_flag and torch.all(weights == torch.FloatTensor(NONE_ENCODING))
        if none_flag:
            return False
    return True


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


def single_path_sample_models(model,
                              edge_indices,
                              group_list,
                              cell='normal',
                              check_valid=False
                              ):
    """
        edge_indicies: list of edge idx. indicate which edges to split
            (if only one edge, then Greedy Search;
            if multiple edges, then Tree Search with depth >1;
            if all edges, then global search;
        group list: indicate how to group (one list indicates one group, list of groups) for each edge in edge_indicies
                    e.g. [[[0], [1, 2], [3, 4]], [[0], [1, 2], [3, 4]]
        cell: normal or reduce
    """
    # give the edge indices for spliting.
    model = model.to('cpu')  # move to CPU at first
    group_models = []

    n_layers = len(group_list)
    assert len(edge_indices) == n_layers  # group_list should have the group for each layer

    n_group_list = [list(range(len(g))) for g in group_list]  # possible sub group idx for each layer.
    model_group_indicies = [i for i in itertools.product(*n_group_list)]
    assert len(model_group_indicies) > 1
    model_group_list = []

    for model_i_indicies in model_group_indicies:
        if cell == 'normal':
            arch_parameters_copy = copy.deepcopy(model.arch_normal_parameters)
        else:
            arch_parameters_copy = copy.deepcopy(model.arch_reduce_parameters)
        model_i_group_list = []

        none_num = 0
        for idx, edge_i in enumerate(edge_indices):
            arch_parameters_copy[edge_i, :] = 0
            group_idx = model_i_indicies[idx]
            op_indicies = group_list[idx][group_idx]
            model_i_group_list.append(op_indicies)
            # check op_indicies is list or not? if list means a mixed op (group)
            if isinstance(op_indicies, list):
                op_indicies = list(flatten(op_indicies))
                for op_idx in op_indicies:
                    if op_idx == 0:  # ZERO Index
                        none_num += 1
                    arch_parameters_copy[edge_i, op_idx] = 1
            else:
                arch_parameters_copy[edge_i, op_indicies] = 1

        # DARTS Rule! must have two NOT None
        if len(edge_indices) - none_num != 2 and check_valid:
            valid_model = False
        else:
            valid_model = True
        # check valid! at least two incoming edge.
        # valid_model = check_model_valid(arch_parameters_copy, model.alpha_split_indicies)
        if valid_model:
            model_copy = copy.deepcopy(model)
            if cell == 'normal':
                model_copy.arch_normal_parameters = arch_parameters_copy
            else:
                model_copy.arch_reduce_parameters = arch_parameters_copy
            model_group_list.append(model_i_group_list)  # here, it should come from the models.
            group_models.append(model_copy)
    return group_models, model_group_list


def check_single_path_model(model):
    n_edges = len(model.arch_normal_parameters)
    normal_flag = [False] * n_edges
    reduce_flag = [False] * n_edges
    for i in range(n_edges):
        normal_flag[i] = model.arch_normal_parameters[i].sum() == 1
        reduce_flag[i] = model.arch_reduce_parameters[i].sum() == 1
    model_flag = all(normal_flag) and all(reduce_flag)
    return model_flag, np.array(normal_flag), np.array(reduce_flag)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()


def main(config):
    train_data, valid_data, xshape, class_num = get_datasets(
        config.data.dataset, config.data.data_path, -1)
    config.xshape = xshape
    config.class_num = class_num

    # create nats-bench 
    search_space = get_search_spaces("cell", config.search_space)
    model_config = dict2config(
        dict(
            super_type=config.model.super_type,
            name=config.model.name,
            C=config.model.C,
            N=config.model.N,
            steps=config.model.steps,
            multiplier=config.model.multiplier,
            stem_multiplier=config.model.stem_multiplier,
            num_classes=class_num,
            space=search_space,
            affine=bool(config.model.affine),
            track_running_stats=bool(config.model.track_running_stats),
            train_arch_parameters=config.model.train_arch_parameters,
        ),
        None,
    )
    logger.log("search space : {:}".format(search_space))
    logger.log("model config : {:}".format(model_config))
    supernet = get_cell_based_tiny_net(model_config)
    logger.log("{:}".format(supernet))

    # warmup. default: False
    config.epochs = config.warmup_epochs
    config.LR = config.warmup_lr
    config.lr_min = config.warmup_lr_min
    w_optimizer, w_scheduler, criterion = get_optim_scheduler(
        supernet.get_weights(), config
    )
    logger.log("w-optimizer : {:}".format(w_optimizer))
    logger.log("w-scheduler : {:}".format(w_scheduler))
    logger.log("criterion   : {:}".format(criterion))
    params = count_parameters_in_MB(supernet)
    logger.log("The parameters of the search model = {:.2f} MB".format(params))
    logger.log("search-space : {:}".format(search_space))

    supernet, criterion = supernet.cuda(), criterion.cuda()  # use a single GPU

    if config.load_path is not None:
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start".format(config.load_path)
        )
        checkpoint = torch.load(config.load_path, map_location='cpu')
        start_epoch = checkpoint["epoch"]
        supernet.load_state_dict(checkpoint["model"])
        w_scheduler.load_state_dict(checkpoint["w_scheduler"])
        w_optimizer.load_state_dict(checkpoint["w_optimizer"])
        logger.log(
            "=> loading checkpoint start with {:}-th epoch.".format(
                start_epoch
            )
        )
    else:
        start_epoch = 0

    _, train_loader, valid_loader = get_nas_search_loaders(
        train_data,
        valid_data,
        config.data.dataset,
        "configs/nas-benchmark/",
        (config.warmup_batch_size, config.test_batch_size),
        config.workers,
        config.debug
    )

    # start training
    start_time = (time.time())
    supernet.apply(init_weights)
    # warm up epochs.
    global_epoch=0
    train_val_epochs(start_epoch, config.warmup_epochs, global_epoch,
                     train_loader, valid_loader,
                     supernet, 0,
                     criterion, w_scheduler, w_optimizer,
                     enable_valid=False
                     )
    global_epoch += config.warmup_epochs
    # save checkpoint of the warmup
    save_checkpoint(
        {
            "epoch": global_epoch,
            "config": copy.deepcopy(config),
            "model": supernet.state_dict(),
            "w_optimizer": w_optimizer.state_dict(),
            "w_scheduler": w_scheduler.state_dict(),
        },
        os.path.join(config.ckpt_dir, config.logname + '_supernet.pth'),
        logger,
    )

    logger.log("\n\n========== Start TNAS Branching ============= ")
    del train_loader
    torch.cuda.empty_cache()
    _, train_loader, valid_loader = get_nas_search_loaders(
        train_data,
        valid_data,
        config.data.dataset,
        "configs/nas-benchmark/",
        (config.train_batch_size, config.test_batch_size),
        config.workers,
        debug=config.debug
    )
    (test_val_idx, argbest) = (0, np.argmin) if 'loss' in config.metric else (1, np.argmax)

    epoch = 0
    config.epochs = config.train_epochs
    config.LR = config.train_lr
    config.LR_min = config.train_lr_min
    n_edges = len(supernet.edge2index)

    if config.d_o == 1:
        groups = [[0], [[[1], [6, 7]], [[2, 3], [4, 5]]]]
    elif config.d_o == 2:
        groups = [[0], [[1, 6, 7], [2, 3, 4, 5]]]
    elif config.d_o == 3:
        groups = [[0], [1, 2, 3, 4, 5, 6, 7]]
    else:
        groups = [0, 1, 2, 3, 4, 5, 6, 7]

    normal_group_lists = [groups] * n_edges  # the groups to choose for each edge
    reduce_group_lists = [groups] * n_edges  # the groups to choose for each edge
    stages = int(np.ceil(np.log2(len(search_space))))
    # check whether edge is single path or not?
    model_flag, normal_flag, reduce_flag = check_single_path_model(supernet)

    # TODO: here, has to percell at first. 
    alphas = ['reduce', 'normal']
    for cell in alphas:
        edge_flag = normal_flag if cell == 'normal' else reduce_flag
        stage = -1 
        while not np.all(edge_flag):
            stage += 1
            step = -1
            if stage == 0:
                depths = range(2, 2 + config.model.steps)
                check_valid = True
            else:
                depths = [config.d_a] * int(np.ceil(n_edges / config.d_a))
                check_valid = False

            stage_edge_flag = copy.deepcopy(edge_flag)
            logger.log(f'edge to decide in normal cell is {normal_flag}'
                    f'edge to decide in reduce cell is {reduce_flag}'
                    )
            logger.log(f'alphas: \nalpha_normal:\n{supernet.arch_normal_parameters}; '
                    f'\nalpha_reduce:\n{supernet.arch_reduce_parameters}')

            while not np.all(stage_edge_flag):
                step += 1
            
                if cell == 'normal':
                    arch_parameters = supernet.arch_normal_parameters
                    group_lists = normal_group_lists
                else:
                    arch_parameters = supernet.arch_reduce_parameters
                    group_lists = reduce_group_lists   

                edge_to_decide = [i for i, x in enumerate(stage_edge_flag) if not x] 
                edge_indicies = edge_to_decide[:depths[step]]
                logger.log(
                    f"\n======= {cell} Cell, Stage:{stage}, Step:{step}, "
                    f"Edge: {edge_indicies} ========")
                logger.log(f"current alpha of {cell} cell is: \n{arch_parameters}")

                group_list = [group_lists[edge_idx] for edge_idx in edge_indicies]
                group_models, group_indicies = single_path_sample_models(supernet, edge_indicies, group_list, cell=cell,
                                                                            check_valid=check_valid)

                if len(group_models) > 1:
                    group_metrics = []
                    group_info = {}
                    for model_idx, (model_c, group_idx_list) in enumerate(zip(group_models, group_indicies)):
                        set_seed(config.rand_seed)
                        logger.log(f"===> {cell} Cell, Stage:{stage}, Step:{step}, "
                                    f"Train and Evaluate {model_idx}/{len(group_indicies)}\n"
                                    f"{group_idx_list} for {edge_indicies}")
                        torch.cuda.empty_cache()
                        model_c = model_c.to(device)
                        if cell == 'normal':
                            logger.log(f"alpha is \n{model_c.arch_normal_parameters}")
                        else:
                            logger.log(f"alpha is \n{model_c.arch_reduce_parameters}")

                        w_optimizer, w_scheduler, criterion = get_optim_scheduler(
                            model_c.get_weights(), config
                        )
                        best_metric_copy = train_val_epochs(epoch, epoch + config.decision_epochs, global_epoch,
                                                            train_loader, valid_loader,
                                                            model_c, model_idx,
                                                            criterion, w_scheduler, w_optimizer,
                                                            enable_valid=not ('loss' in config.metric)
                                                            )[test_val_idx]
                        global_epoch += config.decision_epochs

                        group_metrics.append(best_metric_copy)
                        group_info[str(group_idx_list)] = best_metric_copy
                        model_c = model_c.to('cpu')
                        global_epoch += config.decision_epochs

                    results = ""
                    for i, key in enumerate(group_info):
                        results += "\ngenotype {}, idx {}: {} ;".format(key, i, group_info[key])
                    logger.log(f"Stage: {stage}/{stages} Step: {step} Cell: {cell}, "
                                f"compare the best {config.metric}: {results}")

                    if "loss" in config.metric:
                        best_indices = np.argsort(group_metrics)[:config.topk]
                    else:
                        best_indices = np.argsort(group_metrics)[::-1][:config.topk]

                    # no repeat for DARTS for saving time.
                    best_idx = best_indices[0]
                    # do not use the weights from joint model.
                    supernet = copy.deepcopy(group_models[best_idx])

                else:
                    best_idx = 0
                    supernet.arch_normal_parameters = group_models[best_idx].arch_normal_parameters
                    supernet.arch_reduce_parameters = group_models[best_idx].arch_reduce_parameters

                best_group_idx_list = group_indicies[best_idx]
                # update the edge list
                for i, group_list in enumerate(best_group_idx_list):
                    edge_i = edge_indicies[i]
                    group_lists[edge_i] = best_group_idx_list[i]

                model_flag, normal_flag, reduce_flag = check_single_path_model(supernet)
                if cell == 'normal':
                    edge_flag = normal_flag
                else:
                    edge_flag = reduce_flag
                # update the choose edges as True flag. (means decided)
                for edge_i in edge_indicies:
                    stage_edge_flag[edge_i] = True

                # update the single edges as True flag.
                for i in range(len(stage_edge_flag)):
                    stage_edge_flag[i] = any([stage_edge_flag[i], edge_flag[i]])

                logger.log(
                    f"{cell} Cell, Edge {edge_indicies} choose the best idx {best_idx}, and the best group: {best_group_idx_list}"
                )
                logger.log(
                    f'Finish {cell} Cell, Stage {stage}, Step {step}, current group list: \n {group_lists} \n'
                    f'current edge_flag: {edge_flag}')
                torch.cuda.empty_cache()
        
        # if config.re_init and stage == 0:
        #     supernet.apply(init_weights)
    # the final post procedure : count the time
    genotype = supernet.genotype()
    print(f'===> Finished searching! The final normal alpha is: \n {supernet.arch_normal_parameters}\n'
          f'The final reduce alpha is:\n {supernet.arch_reduce_parameters}\n'
          f'The final genotype is: \n {genotype}\n')
    logger.log("\n" + "-" * 100)
    end_time = time.time()
    total_time = end_time - start_time
    logger.log(f"total search time: {total_time}")


def parse_option():
    parser = argparse.ArgumentParser('search cell')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
    args, opts = parser.parse_known_args()
    config.load(args.cfg, recursive=True)
    config.update(opts)
    config.debug = args.debug
    config.enable_valid = 'loss' not in config.metric
    if config.rand_seed is None or config.rand_seed < 0:
        config.rand_seed = random.randint(1, 100000)
    return args, config


if __name__ == "__main__":
    opt, config = parse_option()

    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(config.rand_seed)

    # if config.load_path is None:
    tags = [config.search_space,
            config.data.dataset,
            config.algo,
            f'N{config.model.N}', f'C{config.model.C}',
            f'WE{config.warmup_epochs}', f'WBS{config.warmup_batch_size}',
            f'DE{config.decision_epochs}', f'BS{config.train_batch_size}',
            f'LR{config.LR}',
            f'{config.metric}', f'd_a{config.d_a}', f'd_o{config.d_o}',
            f'order_{config.order}'
            ]
    if config.re_init:
        tags.append('reinit')
    if config.group:
        tags.append('group')
    tags.append(f'Seed{config.rand_seed}')
    generate_exp_directory(config, tags)
    config.wandb.tags = tags
    # else:  # resume from the existing ckpt and reuse the folder.
    #    resume_exp_directory(config, config.load_path)
    #    config.wandb.tags = ['resume']
    logger = prepare_logger(config)
    # wandb and tensorboard
    cfg_path = os.path.join(config.log_dir, "config.json")
    with open(cfg_path, 'w') as f:
        json.dump(vars(opt), f, indent=2)
        json.dump(vars(config), f, indent=2)
        os.system('cp %s %s' % (opt.cfg, config.log_dir))
    config.cfg_path = cfg_path

    # wandb config
    config.wandb.name = config.logname
    Wandb.launch(config, config.wandb.use_wandb)

    # tensorboard
    summary_writer = SummaryWriter(log_dir=config.log_dir)

    logger.log(config)

    main(config)

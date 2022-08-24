import numpy as np
import os
import datetime
import argparse

import torch
import torchnet as tnt
import json
from src.utils.utils import *
from src.utils.metrics import ConfusionMatrix, CustomMeter
import yaml
from sklearn.metrics import adjusted_rand_score
import torch.utils.data as data
import torch.nn.functional as F
from src.utils.supervised_utils import VisdomLinePlotter
import time
from src.utils.path import CONFIGS_PATH, RESULTS_PATH

FIRST_DATE = 16
LAST_DATE = 421
CLOUD_THRESHOLD = 0.999
IMG_SIZE = 128


def iterate(input_seq, label, mask, config, model, optimizer, mode='train', device=None, use_variance=False):
    proto_count = None
    if config['model']['name'] == 'upssits':
        output_seq, input_seq, loss, indices, label, mask = model.forward(input_seq, label, mask)
        if config['training']['loss'] == "recons":
            logits = - loss
            training_loss = ((mask * ((output_seq - input_seq) ** 2).mean(2)).sum(1) / mask.sum(1)).mean()
            # logits = torch.softmax(-100. * loss, dim=1)  # N x K
            # training_loss = (
            #             ((logits[..., None, None] * model.module.prototypes[None].expand(logits.size(0), -1, -1, -1)
            #               ).sum(1) - output_seq) ** 2).mean()

        elif config['training']['loss'] in ["logreg", "ce"]:
            # logits = torch.softmax(-loss, dim=1)  # N x K
            # training_loss = (((logits[..., None, None] * model.module.prototypes[None].expand(logits.size(0), -1, -1, -1)
            #                   ).sum(1) - output_seq)**2).mean()
            logits = - F.softplus(model.module.beta) * loss
            training_loss = F.cross_entropy(logits, label, reduction='none').mean()

        elif config['training']['loss'] == "mixed":
            logits = loss
            training_loss_logreg = F.cross_entropy(logits, label, reduction='none').mean()
            training_loss_recons = ((output_seq - input_seq) ** 2).mean(2).mean(1).mean()
            training_loss = training_loss_logreg + 0.1 * training_loss_recons
    elif config['model']['name'] == 'mlp':
        logits, label = model.forward(input_seq, label)
        training_loss = F.cross_entropy(logits, label, reduction='none').mean()
    elif config['model']['name'] == 'means':
        input_seq, mask, output_seq, indices, loss, label = model.module.predict(input_seq, mask, label, device,
                                                                                 use_variance=use_variance)
        logits = - loss
        training_loss = ((mask * ((output_seq - input_seq) ** 2).mean(2)).sum(1) / mask.sum(1)).mean()
    elif config['model']['name'] == 'kmeans':
        output_seq, input_seq, loss, indices, label, mask = model.forward(input_seq, label, mask)
        logits = - loss
        training_loss = ((mask * ((output_seq - input_seq) ** 2).mean(2)).sum(1) / torch.where(mask.sum(1) == 0, torch.ones_like(mask.sum(1)), mask.sum(1))).mean()
        proto_count = torch.stack([(indices == k).float().sum() for k in range(config['model']['num_prototypes'])])
    if mode == 'train':
        training_loss.backward()
        optimizer.step()
    _, pred_indices = torch.max(logits, dim=1)
    acc = (label == pred_indices).float().mean() * 100
    class_count = [(label == k).float().sum() for k in range(config['model']['num_classes'])]
    dividers = [class_count[k] if class_count[k] > 0 else 1. for k in range(config['model']['num_classes'])]
    class_count = [1. if class_count[k] > 0 else 0. for k in range(config['model']['num_classes'])]
    acc_per_class = torch.stack([((label == k) * (pred_indices == k)).float().sum() / dividers[k] * 100
                                 for k in range(config['model']['num_classes'])])
    return training_loss, acc, acc_per_class, class_count, proto_count, label, pred_indices


class Logger:
    def __init__(
            self,
            path,
            loss='recons',
            learn_weights=False,
            model_name='upssits',
    ):
        super(Logger, self).__init__()
        self.path = path
        self.loss = loss
        self.learn_weights = learn_weights
        self.model_name = model_name
        self.metrics = {'loss': {},
                        'acc': {},
                        'mean_acc': {},
                        'ari': {},
                        'acc_per_class': {},
                        'lr': {}}
        if self.model_name == 'kmeans':
            self.metrics['proportions'] = {}
        if self.loss in ['logreg', 'mixed', 'ce']:
            self.metrics['lambda'] = {}
        if self.learn_weights:
            self.metrics['weights'] = {}

    def update(self, metrics, n_iter):
        for metric in metrics.keys():
            self.metrics[metric][n_iter] = metrics[metric]
        with open(self.path, "w") as file:
            file.write(json.dumps(self.metrics, indent=4))


def validate(val_loader, model, optimizer, config, device=None, use_variance=False, matching=None):
    loss_meter = tnt.meter.AverageValueMeter()
    acc_per_class_meter = CustomMeter(num_classes=config['model']['num_classes'])
    acc_meter = tnt.meter.AverageValueMeter()
    ari_meter = tnt.meter.AverageValueMeter()
    if config['model']['name'] == 'kmeans':
        conf_matrix = ConfusionMatrix(config['model']['num_classes'], config['model']['num_prototypes'])
    n_iter = 0
    with_img_id = config['dataset'].get('with_img_id', False)
    t_start = time.time()

    for i, batch in enumerate(val_loader):
        input_seq, mask, y = batch
        input_seq = input_seq.view(-1, config['model']['num_steps'], config['model']['input_dim']).to(torch.float32)
        label = y.view(-1).long()
        mask = mask.view(-1, config['model']['num_steps']).int()
        input_seq = input_seq.to(torch.float32)
        with torch.no_grad():
            loss, acc, acc_per_class, class_count, proto_count, label, pred_indices = iterate(input_seq, label, mask,
                                                            config, model, optimizer, mode='test', device=device,
                                                            use_variance=use_variance)
        n_iter += 1
        loss_meter.add(loss.item())
        acc_per_class_meter.add(acc_per_class, class_count)
        acc_meter.add(acc.item())
        ari_meter.add(adjusted_rand_score(label.detach().cpu().numpy(), pred_indices.detach().cpu().numpy()))
        if config['model']['name'] == 'kmeans':
            conf_matrix.add(pred_indices, label)
    t_end = time.time()
    conf_matrix.purity_assignment(matching=matching)
    curr_acc, curr_acc_per_class = acc_meter.value()[0], acc_per_class_meter.value()
    if config['model']['name'] == 'kmeans':
        curr_acc = conf_matrix.get_acc()
        curr_acc_per_class = conf_matrix.get_acc_per_class()
    val_metrics = {'loss': loss_meter.value()[0],
                   'acc': curr_acc,
                   'mean_acc': np.mean(curr_acc_per_class),
                   'ari': ari_meter.value()[0],
                   'acc_per_class': curr_acc_per_class,
                   }
    print('Validation time: {:.2f}s'.format(t_end - t_start))
    print('   Val Loss: {:.3f}, Val Acc: {:.2f}, Val mAcc: {:.2f}'.format(val_metrics['loss'],
                                                                          val_metrics['acc'], val_metrics['mean_acc']))
    if config['training']['loss'] in ['logreg', 'mixed', 'ce']:
        val_metrics['lambda'] = 0
    if config['model'].get('learn_weights', False):
        val_metrics['weights'] = [0 for _ in range(config['model']['num_steps'] * config['model']['input_dim'])]
    return val_metrics


def validate_per_cropid(val_loader, model, optimizer, config, device=None, use_variance=False, matching=None):
    loss_meter = tnt.meter.AverageValueMeter()
    acc_per_class_meter = CustomMeter(num_classes=config['model']['num_classes'])
    acc_meter = tnt.meter.AverageValueMeter()
    ari_meter = tnt.meter.AverageValueMeter()
    if config['model']['name'] == 'kmeans':
        conf_matrix = ConfusionMatrix(config['model']['num_classes'], config['model']['num_prototypes'])
    n_iter = 0
    with_img_id = config['dataset'].get('with_img_id', False)
    t_start = time.time()
    # img_label_dict = torch.zeros((2064, config['model']['num_classes']))
    # img_label_gt = torch.zeros(2064, dtype=torch.int)
    img_label_dict = torch.zeros((2417, config['model']['num_classes']))
    img_label_gt = torch.zeros(2417, dtype=torch.int)

    mapping = {x: i for x, i in enumerate(matching)}

    for i, batch in enumerate(val_loader):
        input_seq, mask, y = batch
        input_seq = input_seq.view(-1, config['model']['num_steps'], config['model']['input_dim']).to(torch.float32)
        y = y.view(-1, 2).long()
        label, img_id = y[:, 0], y[:, 1]
        mask = mask.view(-1, config['model']['num_steps']).int()
        input_seq = input_seq.to(torch.float32)
        with torch.no_grad():
            loss, acc, acc_per_class, class_count, proto_count, label, pred_indices = iterate(input_seq, label, mask,
                                                            config, model, optimizer, mode='test', device=device,
                                                            use_variance=use_variance)
            img_label_dict[img_id, torch.tensor([mapping[x.item()] for x in pred_indices])] += 1
            img_label_gt[img_id] = label.detach().cpu().int()
        n_iter += 1
        loss_meter.add(loss.item())
        acc_per_class_meter.add(acc_per_class, class_count)
        acc_meter.add(acc.item())
        ari_meter.add(adjusted_rand_score(label.detach().cpu().numpy(), pred_indices.detach().cpu().numpy()))
        if config['model']['name'] == 'kmeans':
            conf_matrix.add(pred_indices, label)
    preds = torch.argmax(img_label_dict, dim=1)
    acc_per_class_crops = [(((preds == img_label_gt).float() * (img_label_gt == class_id).float()).sum() / (img_label_gt == class_id).float().sum()).item() * 100 for class_id in range(config['model']['num_classes'])]
    mean_acc_crops = np.mean(acc_per_class_crops)
    acc_crops = (preds == img_label_gt).float().mean().item() * 100
    print(acc_crops)
    print(mean_acc_crops)
    t_end = time.time()
    conf_matrix.purity_assignment(matching=matching)
    curr_acc, curr_acc_per_class = acc_meter.value()[0], acc_per_class_meter.value()
    if config['model']['name'] == 'kmeans':
        curr_acc = conf_matrix.get_acc()
        curr_acc_per_class = conf_matrix.get_acc_per_class()
    val_metrics = {'loss': loss_meter.value()[0],
                   'acc': curr_acc,
                   'mean_acc': np.mean(curr_acc_per_class),
                   'ari': ari_meter.value()[0],
                   'acc_per_class': curr_acc_per_class,
                   'acc_crops': acc_crops,
                   'mean_acc_crops': mean_acc_crops,
                   'acc_per_class_crops': acc_per_class_crops,
                   }
    print('Validation time: {:.2f}s'.format(t_end - t_start))
    print('   Val Loss: {:.3f}, Val Acc: {:.2f}, Val mAcc: {:.2f}'.format(val_metrics['loss'],
                                                                          val_metrics['acc'], val_metrics['mean_acc']))
    if config['training']['loss'] in ['logreg', 'mixed', 'ce']:
        val_metrics['lambda'] = 0
    if config['model'].get('learn_weights', False):
        val_metrics['weights'] = [0 for _ in range(config['model']['num_steps'] * config['model']['input_dim'])]
    return val_metrics


def update_viz(viz, metrics, mode, n_iter, model_name='upssits'):
    viz.plot('loss', mode, 'Loss', n_iter, metrics['loss'])
    viz.plot('ari', mode, 'Adjusted Rand Index', n_iter, metrics['ari'])
    viz.plot('acc', mode, 'Overall Accuracy', n_iter, metrics['acc'])
    viz.plot('m_acc', mode, 'Mean Accuracy', n_iter, metrics['mean_acc'])
    if mode == 'train':
        viz.plot('lr', mode, 'Learning rate', n_iter, np.log(metrics['lr']) / np.log(10))
        if 'lambda' in list(metrics.keys()):
            viz.plot('lambda', mode, 'Lambda', n_iter, metrics['lambda'])
        if 'weights' in list(metrics.keys()):
            viz.histogram('weights', mode, 'Weights', metrics['weights'])
        if model_name == 'kmeans':
            viz.bar('proportions', mode, 'Proto Prop', np.array(metrics['proportions']))


def main(config, res_dir, exp_id, viz=None):
    coerce_to_path_and_create_dir(res_dir)

    if torch.cuda.is_available():
        type_device = "cuda"
        nb_device = torch.cuda.device_count()
    else:
        type_device = "cpu"
        nb_device = None

    print("Using {} device, nb_device is {}".format(type_device, nb_device))

    np.random.seed(config['training'].get("seed", 621))
    torch.manual_seed(config['training'].get("seed", 621))
    device = torch.device(config['training'].get("device", 'cuda'))

    train_dataset = get_train_dataset(config["dataset"])
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=config['training'].get("batch_size"),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    val_dataset = get_val_dataset(config["dataset"])
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=config['training'].get("batch_size"),
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    model = get_model(config["model"])
    model = torch.nn.DataParallel(model.to(device), device_ids=[0, 1, 2, 3])
    config["N_params"] = get_ntrainparams(model)
    print(f"Model has {config['N_params']} parameters.")

    with open(os.path.join(res_dir, "conf.json"), "w") as file:
        file.write(json.dumps(config, indent=4))

    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), **config['training']['optimizer'])

    if config['model']['name'] == 'kmeans':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', **config['training']['scheduler'])
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', **config['training']['scheduler'])

    train_log_path = os.path.join(res_dir, "train_logs.json")
    val_log_path = os.path.join(res_dir, "val_logs.json")
    if config['model']['name'] != 'means':
        train_logger = Logger(train_log_path,
                              loss=config['model']['learn_weights'],
                              learn_weights=config['model']['learn_weights'],
                              model_name=config['model']['name'])
        val_logger = Logger(val_log_path,
                            loss=config['model']['learn_weights'],
                            learn_weights=config['model']['learn_weights'],
                            model_name=config['model']['name'])

    n_iter = 0
    best_acc = 0
    best_loss = np.inf
    loss_meter = tnt.meter.AverageValueMeter()
    if config['model']['name'] == 'kmeans':
        conf_matrix = ConfusionMatrix(config['model']['num_classes'], config['model']['num_prototypes'])
        proto_count_meter = CustomMeter(num_classes=config['model']['num_prototypes'])
    ari_meter = tnt.meter.AverageValueMeter()
    acc_per_class_meter = CustomMeter(num_classes=config['model']['num_classes'])
    acc_meter = tnt.meter.AverageValueMeter()
    for epoch in range(1, config['training']['n_epochs'] + 1):
        print('Epoch {}/{}:'.format(epoch, config['training']['n_epochs']))
        t_start = time.time()
        for i, batch in enumerate(train_loader):
            input_seq, mask, y = batch
            input_seq = input_seq.view(-1, config['model']['num_steps'], config['model']['input_dim']).to(torch.float32)
            label = y.view(-1).long()
            mask = mask.view(-1, config['model']['num_steps']).int()
            if config['model']['name'] == 'means':
                if i % 1000 == 0:
                    print(f'Iter {i}/10042')
                model(input_seq, label, mask)
            else:
                loss, acc, acc_per_class, class_count, proto_count, label, pred_indices = iterate(input_seq, label, mask,
                                                                                                  config, model, optimizer)
                n_iter += 1
                loss_meter.add(loss.item())
                acc_per_class_meter.add(acc_per_class, class_count)
                acc_meter.add(acc.item())
                ari_meter.add(adjusted_rand_score(label.detach().cpu().numpy(), pred_indices.detach().cpu().numpy()))
                if config['model']['name'] == 'kmeans':
                    conf_matrix.add(pred_indices, label)
                    proto_count_meter.add(proto_count, [0 for _ in range(config['model']['num_prototypes'])])
                    if n_iter % config['training']['check_cluster_step'] == 0:
                        curr_prop = proto_count_meter.value(mode='density')
                        model.module.reassign_empty_clusters(curr_prop)
                        proto_count_meter.reset()
                if n_iter % config['training']['print_step'] == 0:
                    curr_acc = acc_meter.value()[0]
                    curr_acc_per_class = acc_per_class_meter.value()
                    matching = None
                    if config['model']['name'] == 'kmeans':
                        matching = conf_matrix.purity_assignment()
                        curr_acc = conf_matrix.get_acc()
                        curr_acc_per_class = conf_matrix.get_acc_per_class()
                        conf_matrix.reset()
                    print('   Iter {}: Loss {:.3f}, Acc {:.2f}%, Mean Acc {:.2f}%, ARI {:.4f}'.format(n_iter,
                                                                                          loss_meter.value()[0],
                                                                                          curr_acc,
                                                                                          np.mean(curr_acc_per_class),
                                                                                          ari_meter.value()[0])
                          )
                    print('      Loss per class {}'.format(' '.join(['{:.2f}'.format(curr_acc)
                                                                     for curr_acc in curr_acc_per_class])))
                    train_metrics = {'loss': loss_meter.value()[0],
                                     'acc': curr_acc,
                                     'mean_acc': np.mean(curr_acc_per_class),
                                     'ari': ari_meter.value()[0],
                                     'acc_per_class': curr_acc_per_class,
                                     'lr': optimizer.param_groups[0]['lr']}
                    if config['model']['name'] == 'kmeans':
                        train_metrics['proportions'] = curr_prop
                    if config['training']['loss'] in ['logreg', 'mixed', 'ce']:
                        train_metrics['lambda'] = F.softplus(model.module.beta).item()
                    if config['model']['learn_weights']:
                        tmp = F.softplus(model.module.weights)/F.softplus(model.module.weights).sum()
                        train_metrics['weights'] = [[tmp[i, j].item() for i in range(config['model']['num_steps'])]
                                                    for j in range(config['model']['input_dim'])]
                    update_viz(viz, train_metrics, 'train', n_iter, config['model']['name'])
                    train_logger.update(train_metrics, n_iter)
                    loss_meter.reset()
                    acc_per_class_meter.reset()
                    acc_meter.reset()
                if n_iter % config['training']['valid_step'] == 0:
                    print('Validation...')
                    model.eval()
                    val_metrics = validate(val_loader, model, optimizer, config, matching=matching)
                    update_viz(viz, val_metrics, 'val', n_iter, config['model']['name'])
                    if config['model']['name'] == 'kmeans':
                        pre_lr = optimizer.param_groups[0]['lr']
                        scheduler.step(val_metrics['loss'])
                        post_lr = optimizer.param_groups[0]['lr']
                        if pre_lr > post_lr:
                            scheduler._reset()
                    else:
                        pre_lr = optimizer.param_groups[0]['lr']
                        scheduler.step(val_metrics['acc'])
                        post_lr = optimizer.param_groups[0]['lr']
                        if pre_lr > post_lr:
                            scheduler._reset()
                    print('... done!')
                    val_logger.update(val_metrics, n_iter)
                    model.train()
                    if val_metrics['acc'] > best_acc:
                        best_acc = val_metrics['acc']
                        torch.save(
                            {
                                "iter": n_iter,
                                "state_dict": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                            },
                            os.path.join(
                                res_dir, "model_acc.pth.tar"
                            ))
                        if matching is not None:
                            with open(os.path.join(res_dir, "matching_acc.json"), "w") as file:
                                file.write(json.dumps({'matching': matching.astype(int).tolist()}, indent=4))
                    if val_metrics['loss'] < best_loss:
                        best_loss = val_metrics['loss']
                        torch.save(
                            {
                                "iter": n_iter,
                                "state_dict": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                            },
                            os.path.join(
                                res_dir, "model_loss.pth.tar"
                            ))
                        if matching is not None:
                            with open(os.path.join(res_dir, "matching_loss.json"), "w") as file:
                                file.write(json.dumps({'matching': matching.astype(int).tolist()}, indent=4))
                    torch.save(
                        {
                            "iter": n_iter,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        os.path.join(
                            res_dir, "model_last.pth.tar"
                        ))
                    if matching is not None:
                        with open(os.path.join(res_dir, "matching_last.json"), "w") as file:
                            file.write(json.dumps({'matching': matching.astype(int).tolist()}, indent=4))
                if optimizer.param_groups[0]['lr'] < 0.00001:
                    break

        t_end = time.time()
        print('Epoch time: {:.2f}s'.format(t_end - t_start))

        if optimizer.param_groups[0]['lr'] < 0.00001:
            break

    test_dataset = get_test_dataset(config["dataset"])
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    if config['model']['name'] == 'means':
        model.module.compute_mean()
        torch.save(
            {
                "iter": n_iter,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(
                res_dir, "model.pth.tar"
            ))
        model.load_state_dict(
            torch.load(
                os.path.join(
                    res_dir, "model.pth.tar"
                )
            )["state_dict"]
        )
        model.eval()

        print("Evaluate on val...")
        val_metrics = validate(val_loader, model, optimizer, config, device=device)
        print(val_metrics)
        with open(os.path.join(res_dir, "val_metrics.json"), "w") as file:
            file.write(json.dumps(val_metrics, indent=4))

        print("Evaluate on test...")
        test_metrics = validate(test_loader, model, optimizer, config, device=device)
        print(test_metrics)
        with open(os.path.join(res_dir, "test_metrics.json"), "w") as file:
            file.write(json.dumps(test_metrics, indent=4))

        print("Evaluate lda on val...")
        val_metrics = validate(val_loader, model, optimizer, config, device=device, use_variance=True)
        print(val_metrics)
        with open(os.path.join(res_dir, "val_metrics_lda.json"), "w") as file:
            file.write(json.dumps(val_metrics, indent=4))

        print("Evaluate lda on test...")
        test_metrics = validate(test_loader, model, optimizer, config, device=device, use_variance=True)
        print(test_metrics)
        with open(os.path.join(res_dir, "test_metrics_lda.json"), "w") as file:
            file.write(json.dumps(test_metrics, indent=4))

    else:
        model.load_state_dict(
            torch.load(
                os.path.join(
                    res_dir, "model_acc.pth.tar"
                )
            )["state_dict"]
        )
        model.eval()
        print("Testing best model acc...")
        matching = None
        if config['model']['name'] == 'kmeans':
            with open(os.path.join(res_dir, "matching_acc.json"), "r") as file:
                matching = json.load(file)['matching']
        test_metrics = validate(test_loader, model, optimizer, config, device=device, matching=matching)
        print(test_metrics)
        with open(os.path.join(res_dir, "test_metrics_acc.json"), "w") as file:
            file.write(json.dumps(test_metrics, indent=4))

        model.load_state_dict(
            torch.load(
                os.path.join(
                    res_dir, "model_loss.pth.tar"
                )
            )["state_dict"]
        )
        model.eval()
        print("Testing best model loss...")
        matching = None
        if config['model']['name'] == 'kmeans':
            with open(os.path.join(res_dir, "matching_loss.json"), "r") as file:
                matching = json.load(file)['matching']

        test_metrics = validate(test_loader, model, optimizer, config, device=device, matching=matching)
        print(test_metrics)
        with open(os.path.join(res_dir, "test_metrics_loss.json"), "w") as file:
            file.write(json.dumps(test_metrics, indent=4))

        model.load_state_dict(
            torch.load(
                os.path.join(
                    res_dir, "model_last.pth.tar"
                )
            )["state_dict"]
        )
        model.eval()
        print("Testing last model...")
        matching = None
        if config['model']['name'] == 'kmeans':
            with open(os.path.join(res_dir, "matching_last.json"), "r") as file:
                matching = json.load(file)['matching']
        test_metrics = validate(test_loader, model, optimizer, config, device=device, matching=matching)
        print(test_metrics)
        with open(os.path.join(res_dir, "test_metrics_last.json"), "w") as file:
            file.write(json.dumps(test_metrics, indent=4))


def predict_by_crop(config, res_dir):
    if torch.cuda.is_available():
        type_device = "cuda"
        nb_device = torch.cuda.device_count()
    else:
        type_device = "cpu"
        nb_device = None

    print("Using {} device, nb_device is {}".format(type_device, nb_device))

    np.random.seed(config['training'].get("seed", 621))
    torch.manual_seed(config['training'].get("seed", 621))
    device = torch.device(config['training'].get("device", 'cuda'))

    model = get_model(config["model"])
    model = torch.nn.DataParallel(model.to(device), device_ids=[0, 1, 2, 3])
    config["N_params"] = get_ntrainparams(model)
    print(f"Model has {config['N_params']} parameters.")

    optimizer = torch.optim.Adam(model.parameters(), **config['training']['optimizer'])

    test_dataset = get_test_dataset(config["dataset"])
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )


    model.load_state_dict(
        torch.load(
            os.path.join(
                res_dir, "model_last.pth.tar"
            )
        )["state_dict"]
    )
    model.eval()
    print("Testing best model acc...")
    matching = None
    if config['model']['name'] == 'kmeans':
        with open(os.path.join(res_dir, "matching_acc.json"), "r") as file:
            matching = json.load(file)['matching']
    test_metrics = validate_per_cropid(test_loader, model, optimizer, config, device=device, matching=matching)
    print(test_metrics)
    with open(os.path.join(res_dir, "test_metrics_acc.json"), "w") as file:
        file.write(json.dumps(test_metrics, indent=4))

    model.load_state_dict(
        torch.load(
            os.path.join(
                res_dir, "model_loss.pth.tar"
            )
        )["state_dict"]
    )
    model.eval()
    print("Testing best model loss...")
    matching = None
    if config['model']['name'] == 'kmeans':
        with open(os.path.join(res_dir, "matching_loss.json"), "r") as file:
            matching = json.load(file)['matching']

    test_metrics = validate_per_cropid(test_loader, model, optimizer, config, device=device, matching=matching)
    print(test_metrics)
    with open(os.path.join(res_dir, "test_metrics_loss.json"), "w") as file:
        file.write(json.dumps(test_metrics, indent=4))

    model.load_state_dict(
        torch.load(
            os.path.join(
                res_dir, "model_last.pth.tar"
            )
        )["state_dict"]
    )
    model.eval()
    print("Testing last model...")
    matching = None
    if config['model']['name'] == 'kmeans':
        with open(os.path.join(res_dir, "matching_last.json"), "r") as file:
            matching = json.load(file)['matching']
    test_metrics = validate_per_cropid(test_loader, model, optimizer, config, device=device, matching=matching)
    print(test_metrics)
    with open(os.path.join(res_dir, "test_metrics_last.json"), "w") as file:
        file.write(json.dumps(test_metrics, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline to train a NN model specified by a YML config")
    parser.add_argument("-t", "--tag", nargs="?", type=str, required=True, help="Run tag of the experiment")
    parser.add_argument("-c", "--config", nargs="?", type=str, required=True, help="Config file name")
    args = parser.parse_args()

    assert args.tag is not None and args.config is not None
    config = coerce_to_path_and_check_exist(CONFIGS_PATH / args.config)
    with open(config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    seed = cfg["training"].get("seed", 621)
    dataset = cfg["dataset"]["name"]
    # config = parser.parse_args()
    exp_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_id = "{}".format(exp_time)
    if dataset == 'pastis':
        folds = [[[1, 2, 3], [4], [5]],
                 [[2, 3, 4], [5], [1]],
                 [[3, 4, 5], [1], [2]],
                 [[4, 5, 1], [2], [3]],
                 [[5, 1, 2], [3], [4]],
                 ]
        for i, (train, val, test) in enumerate(folds):
            res_dir = RESULTS_PATH / dataset / args.tag / f'Fold_{i+1}'
            cfg["dataset"]["train_fold"] = train
            cfg["dataset"]["val_fold"] = val
            cfg["dataset"]["test_fold"] = test
            viz = VisdomLinePlotter(exp_id)
            main(cfg, res_dir, exp_id, viz)

    else:
        res_dir = RESULTS_PATH / dataset / args.tag
        viz = VisdomLinePlotter(exp_id)
        # predict_by_crop(cfg, res_dir)
        main(cfg, res_dir, exp_id, viz)
import argparse
import joblib
import math
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import sys
import time
import torch.nn.functional
import torchnet as tnt
import yaml

from src.utils.metrics import ConfusionMatrix, CustomMeter
from src.utils.paths import CONFIGS_PATH
from src.utils.utils import *
from src.utils.constants import *


def iterate(input_seq, label, mask, config, model, optimizer, mode='train', n_iter=0):
    if config['model']['name'] == 'dtits':
        return iterate_dtits(input_seq, label, mask, config, model, optimizer, mode=mode, n_iter=n_iter)
    elif config['model']['name'] == 'ltae':
        return iterate_ltae(input_seq, label, mask, config, model, optimizer, mode=mode)
    elif config['model']['name'] == 'tapnet':
        return iterate_tapnet(input_seq, label, config, model, mode=mode)
    else:
        return iterate_ncc(input_seq, label, mask, config, model)


def iterate_ltae(input_seq, label, mask, config, model, optimizer, mode='train'):
    logits, label, mask = model.forward(input_seq, label, mask)
    _, pred_indices = torch.max(logits, dim=1)
    training_loss = torch.nn.functional.cross_entropy(logits, label)
    class_count = [(label == k).float().sum() for k in range(config['model']['num_classes'])]
    class_count = [1. if class_count[k] > 0 else 0. for k in range(config['model']['num_classes'])]
    if mode == 'train':
        training_loss.backward()
        optimizer.step()
    return training_loss, class_count, torch.ones((19, 1)), label, pred_indices, None, torch.zeros((1,1)), torch.zeros((1,1))


def iterate_tapnet(input_seq, label, config, model, mode='train'):
    input_seq = input_seq.permute(0, 2, 1)
    output = model(input_seq, label, mode=mode)
    pred_indices = torch.argmax(output)
    training_loss = torch.nn.functional.cross_entropy(output, torch.squeeze(label))
    class_count = [(label == k).float().sum() for k in range(config['model']['num_classes'])]
    class_count = [1. if class_count[k] > 0 else 0. for k in range(config['model']['num_classes'])]
    return training_loss, class_count,  torch.ones((19, 1)), label, pred_indices, None, None, None


def iterate_ncc(input_seq, label, mask, config, model):
    output_seq, input_seq, loss, indices, label, mask = model.forward(input_seq, label, mask)
    logits = - loss
    proto_count = torch.stack([(indices == k).float().sum() for k in range(config['model']['num_prototypes'])])
    recons_loss, pred_indices = torch.max(logits, dim=1)
    recons_loss = (-recons_loss / config['model']['input_dim'] / torch.where(mask.sum(1) == 0, torch.ones_like(mask.sum(1)), mask.sum(1))).mean()
    training_loss = recons_loss
    class_count = [(label == k).float().sum() for k in range(config['model']['num_classes'])]
    class_count = [1. if class_count[k] > 0 else 0. for k in range(config['model']['num_classes'])]
    return training_loss, class_count, proto_count, label, pred_indices, output_seq, None, None


def iterate_dtits(input_seq, label, mask, config, model, optimizer, mode='train', n_iter=0):
    if optimizer is not None:
        optimizer.zero_grad()
    if mode == 'train' or mode == 'val':
        trans_activ, offset_activ = False, False
        if config['training']['trans_activ'] and n_iter >= config['training']['curriculum'][0]:
            trans_activ = True
        if config['training']['offset_activ'] and n_iter >= config['training']['curriculum'][1]:
            offset_activ = True
    else:
        trans_activ = True
        offset_activ = True
    output_seq, input_seq, loss, indices, label, mask = model.forward(input_seq, label, mask,
                                                                      trans_activ=trans_activ,
                                                                      offset_activ=offset_activ)
    tv_h = ((model.module.prototypes[:, 1:, :] - model.module.prototypes[:, :-1, :])**2).mean()
    logits = - loss
    proto_count = torch.stack([(indices == k).float().sum() for k in range(config['model']['num_prototypes'] * config['model'].get('num_proto_per_class', 1))])
    training_loss = ((mask * ((output_seq - input_seq) ** 2).mean(2)).sum(1) / torch.where(mask.sum(1) == 0, torch.ones_like(mask.sum(1)), mask.sum(1))).mean()
    training_loss_backward = training_loss + LAMBDA * tv_h
    l_ce = torch.nn.functional.cross_entropy(logits, label, reduction='none').mean()
    if config['model'].get('supervised', False) and config['training']['ce_activ'] and n_iter >= config['training']['curriculum'][2] and mode == 'train':
        training_loss_backward = training_loss_backward + MU * l_ce
    if mode == 'train':
        training_loss_backward.backward()
        optimizer.step()
    recons_loss, pred_indices = torch.max(logits, dim=1)
    if mode != 'train':
        recons_loss = (-recons_loss / config['model']['input_dim'] / torch.where(mask.sum(1) == 0, torch.ones_like(mask.sum(1)), mask.sum(1))).mean()
        training_loss = recons_loss
    class_count = [(label == k).float().sum() for k in range(config['model']['num_classes'])]
    class_count = [1. if class_count[k] > 0 else 0. for k in range(config['model']['num_classes'])]
    return training_loss, class_count, proto_count, label, pred_indices, output_seq, tv_h, l_ce


def validate_ltae(val_loader, model, optimizer, config, mode='test', matching=None, n_iter=0):
    loss_meter = tnt.meter.AverageValueMeter()
    conf_matrix = ConfusionMatrix(config['model']['num_classes'], config['model']['num_classes'])

    for i, batch in enumerate(val_loader):
        input_seq, mask, y = batch
        input_seq = input_seq.view(-1, config['model']['num_steps'], config['model']['input_dim']).to(torch.float32)
        label = y.view(-1).long()
        mask = mask.view(-1, config['model']['num_steps']).int()
        input_seq = input_seq.to(torch.float32)
        with torch.no_grad():
            loss, _, _, label, pred_indices, _, _, _ = iterate(input_seq, label, mask,
                                                            config, model, optimizer, mode=mode, n_iter=n_iter)
        loss_meter.add(loss.item())
        conf_matrix.add(pred_indices, label)
    conf_matrix.purity_assignment(matching=matching)
    curr_acc = conf_matrix.get_acc()
    curr_acc_per_class = conf_matrix.get_acc_per_class()
    val_metrics = {'loss': loss_meter.value()[0],
                   'acc': curr_acc,
                   'mean_acc': np.mean(curr_acc_per_class),
                   'acc_per_class': curr_acc_per_class,
                   'conf_matrix': conf_matrix.purity_conf.tolist(),
                   }
    print('   Val Loss: {:.3f}, Val Acc: {:.2f}, Val mAcc: {:.2f}'.format(val_metrics['loss'],
                                                                          val_metrics['acc'], val_metrics['mean_acc']))
    return val_metrics


def validate(val_loader, model, optimizer, config, mode='test', matching=None, n_iter=0):
    loss_meter = tnt.meter.AverageValueMeter()
    tvh_meter = tnt.meter.AverageValueMeter()
    lce_meter = tnt.meter.AverageValueMeter()
    conf_matrix = ConfusionMatrix(config['model']['num_classes'], config['model']['num_prototypes'])

    for i, batch in enumerate(val_loader):
        input_seq, mask, y = batch
        input_seq = input_seq.view(-1, config['model']['num_steps'], config['model']['input_dim']).to(torch.float32)
        label = y.view(-1).long()
        mask = mask.view(-1, config['model']['num_steps']).int()
        input_seq = input_seq.to(torch.float32)
        with torch.no_grad():
            loss, class_count, proto_count, label, pred_indices, _, tv_h, l_ce = iterate(input_seq, label, mask,
                                                            config, model, optimizer, mode=mode, n_iter=n_iter)
        loss_meter.add(loss.item())
        tvh_meter.add(tv_h.item())
        lce_meter.add(l_ce.item())
        conf_matrix.add(pred_indices, label)
    conf_matrix.purity_assignment(matching=matching)
    curr_acc = conf_matrix.get_acc()
    curr_acc_per_class = conf_matrix.get_acc_per_class()
    val_metrics = {'loss': loss_meter.value()[0],
                   'acc': curr_acc,
                   'mean_acc': np.mean(curr_acc_per_class),
                   'tvh': tvh_meter.value()[0],
                   'lce': lce_meter.value()[0],
                   'acc_per_class': curr_acc_per_class,
                   'conf_matrix': conf_matrix.purity_conf.tolist(),
                   }
    print('   Val Loss: {:.3f}, Val Acc: {:.2f}, Val mAcc: {:.2f}'.format(val_metrics['loss'],
                                                                          val_metrics['acc'], val_metrics['mean_acc']))
    return val_metrics


def get_matching(config, res_dir, model, device, ckpt=''):
    print('Getting the prototype assignment...')
    path = Path(os.path.join(res_dir, f'matching{ckpt}.json'))
    if path.exists():
        matching = np.array(json.load(open(path, 'r'))['matching'])
        print('... done!')
        return matching
    elif config['model'].get('supervised', True):
        print('... done!')
        return np.array(list(range(config['model']['num_classes'])))
    else:
        train_dataset = get_dataset(config["dataset"], 'train')
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['training'].get("batch_size"),
            num_workers=config['training']['n_workers'],
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        model.eval()
        with torch.no_grad():
            conf_mat = np.zeros((config['model']['num_classes'], config['model']['num_prototypes'])).flatten()
            for i, batch in enumerate(train_loader):
                input_seq, mask, y = batch
                input_seq = input_seq.view(-1, config['model']['num_steps'], config['model']['input_dim']).to(
                    torch.float32).to(device)
                label = y.view(-1).long().to(device)
                mask = mask.view(-1, config['model']['num_steps']).int().to(device)
                loss, _, _, label, pred_indices, _, _, _ = iterate(input_seq, label, mask, config, model, None, n_iter=0,
                                                                mode='test')
                np.add.at(conf_mat, config['model'][
                    'num_prototypes'] * label.detach().cpu().numpy() + pred_indices.detach().cpu().numpy(), 1)
        conf_mat = np.reshape(conf_mat, (config['model']['num_classes'], config['model']['num_prototypes']))
        matching = np.argmax(conf_mat, axis=0)
        with open(os.path.join(res_dir, f"matching{ckpt}.json"), 'w') as file:
            file.write(json.dumps({'matching': matching.tolist(),
                                   'conf_mat': conf_mat.tolist()}, indent=4))
        print('... done!')
        return matching


def train(config, res_dir):
    coerce_to_path_and_create_dir(res_dir)

    if torch.cuda.is_available():
        type_device = "cuda"
        nb_device = torch.cuda.device_count()
    else:
        type_device = "cpu"
        nb_device = None
    print("Using {} device, nb_device is {}".format(type_device, nb_device))
    device = torch.device(config['training'].get("device", 'cuda'))

    seed = config['training'].get("rdm_seed", 621)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_split = 'train' if config['model']['supervised'] else 'all'
    train_dataset = get_dataset(config["dataset"], train_split)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training'].get("batch_size"),
        num_workers=config['training']['n_workers'],
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )

    val_dataset = get_dataset(config["dataset"], 'val')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training'].get("batch_size"),
        num_workers=config['training']['n_workers'],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    proto_init = initialize_prototypes(config, train_loader, device)
    config["model"]["sample"] = proto_init
    model = get_model(config["model"])
    model = torch.nn.DataParallel(model.to(device), device_ids=range(nb_device))
    config["N_params"] = get_ntrainparams(model)
    print(f"Model has {config['N_params']} parameters.")
    with open(os.path.join(res_dir, "conf.json"), "w") as file:
        config["model"]["sample"] = None
        file.write(json.dumps(config, indent=4))

    optimizer = torch.optim.Adam(model.parameters(), **config['training']['optimizer'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', **config['training']['scheduler'])

    train_log_path = os.path.join(res_dir, "train_logs.json")
    val_log_path = os.path.join(res_dir, "val_logs.json")
    train_logger = Logger(train_log_path)
    val_logger = Logger(val_log_path)

    n_iter = 0
    n_iter_since_new = 0
    n_iter_since_complete = 0
    best_val_loss = np.inf
    loss_meter = tnt.meter.AverageValueMeter()
    conf_matrix = ConfusionMatrix(config['model']['num_classes'], config['model']['num_prototypes'])
    proto_count_meter = CustomMeter(num_classes=config['model']['num_prototypes'] * config['model'].get('num_proto_per_class', 1))
    tvh_meter = tnt.meter.AverageValueMeter()
    lce_meter = tnt.meter.AverageValueMeter()
    curriculum_scheduler = TransfoScheduler(config)
    for epoch in range(1, config['training']['n_epochs'] + 1):
        print('Epoch {}/{}:'.format(epoch, config['training']['n_epochs']))
        for i, batch in enumerate(train_loader):
            input_seq, mask, y = batch
            input_seq = input_seq.view(-1, config['model']['num_steps'], config['model']['input_dim']).to(torch.float32)
            label = y.view(-1).long()
            mask = mask.view(-1, config['model']['num_steps']).int()
            loss, class_count, proto_count, label, pred_indices, output_seq, tv_h, l_ce = iterate(input_seq, label, mask, config, model, optimizer, n_iter=n_iter)
            n_iter += 1
            n_iter_since_new += 1
            if curriculum_scheduler.is_complete:
                n_iter_since_complete += 1

            loss_meter.add(loss.item())
            tvh_meter.add(tv_h.item())
            lce_meter.add(l_ce.item())
            conf_matrix.add(pred_indices, label)
            proto_count_meter.add(proto_count, [0 for _ in range(config['model']['num_prototypes'] * config['model'].get('num_proto_per_class', 1))])

            if n_iter % config['training']['check_cluster_step'] == 0:
                curr_prop = proto_count_meter.value(mode='density')
                if not config['model']['supervised'] or config['model'].get('num_proto_per_class', 1) > 1:
                    model.module.reassign_empty_clusters(curr_prop)
                proto_count_meter.reset()

            if n_iter % config['training']['print_step'] == 0:
                matching = conf_matrix.purity_assignment()
                curr_acc = conf_matrix.get_acc()
                curr_acc_per_class = conf_matrix.get_acc_per_class()
                conf_matrix.reset()
                print('   Iter {}: Loss {:.3f}, Acc {:.2f}%, Mean Acc {:.2f}%, TV {:.3f}, LCE {:.3f}'.format(n_iter,
                                                                                      loss_meter.value()[0],
                                                                                      curr_acc,
                                                                                      np.mean(curr_acc_per_class),
                                                                                      tvh_meter.value()[0],
                                                                                      lce_meter.value()[0]))
                print('      Loss per class {}'.format(' '.join(['{:.2f}'.format(curr_acc)
                                                                 for curr_acc in curr_acc_per_class])))
                train_metrics = {'loss': loss_meter.value()[0],
                                 'acc': curr_acc,
                                 'mean_acc': np.mean(curr_acc_per_class),
                                 'tvh': tvh_meter.value()[0],
                                 'lce': lce_meter.value()[0],
                                 'acc_per_class': curr_acc_per_class,
                                 'lr': optimizer.param_groups[0]['lr'],
                                 'proportions': curr_prop}
                train_logger.update(train_metrics, n_iter)
                loss_meter.reset()
                tvh_meter.reset()
                lce_meter.reset()
                proto_count_meter.reset()

            if n_iter % config['training']['valid_step'] == 0:
                if config['model']['supervised']:
                    matching = np.array(list(range(config['model']['num_classes'])))
                print('Validation...')
                model.eval()
                val_metrics = validate(val_loader, model, optimizer, config, mode='val', matching=matching, n_iter=n_iter)
                val_loss_to_consider = -val_metrics['mean_acc'] if config['model']['supervised'] else val_metrics[
                    'loss']
                if n_iter_since_new >= N_WARM_UP_ITER:
                    if not curriculum_scheduler.is_complete:
                        prev_transf = curriculum_scheduler.curr_transfo
                        curriculum_scheduler.update(val_loss_to_consider, n_iter)
                        next_transf = curriculum_scheduler.curr_transfo
                        if prev_transf != next_transf:
                            n_iter_since_new = 0
                            with open(os.path.join(res_dir, f"conf_transf_{next_transf}.json"), "w") as file:
                                file.write(json.dumps(curriculum_scheduler.config, indent=4))
                            model.load_state_dict(
                                torch.load(
                                    os.path.join(
                                        res_dir, f"model.pth.tar"
                                    )
                                )["state_dict"]
                            )
                            optimizer.load_state_dict(
                                torch.load(
                                    os.path.join(
                                        res_dir, f"model.pth.tar"
                                    )
                                )["optimizer"]
                            )
                            torch.save(
                                {
                                    "iter": n_iter,
                                    "state_dict": model.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                },
                                os.path.join(
                                    res_dir, f"model_best_transf_{next_transf}.pth.tar"
                                ))
                        config = curriculum_scheduler.config
                pre_lr = optimizer.param_groups[0]['lr']
                if curriculum_scheduler.is_complete and n_iter_since_complete > N_WARM_UP_ITER:
                    scheduler.step(val_loss_to_consider)
                post_lr = optimizer.param_groups[0]['lr']
                if pre_lr > post_lr:
                    scheduler._reset()
                print('... done!')
                val_logger.update(val_metrics, n_iter)
                model.train()
                if val_loss_to_consider < best_val_loss:
                    best_val_loss = val_loss_to_consider
                    torch.save(
                        {
                            "iter": n_iter,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        os.path.join(
                            res_dir, "model.pth.tar"
                        ))
                    with open(os.path.join(res_dir, "val_metrics.json"), "w") as file:
                        file.write(json.dumps(val_metrics, indent=4))

            if optimizer.param_groups[0]['lr'] < MIN_LR:
                break

        if optimizer.param_groups[0]['lr'] < MIN_LR:
            break

    print('Training over!')
    print('Evaluate on test...')
    test_dataset = get_dataset(config['dataset'], 'test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['n_workers'],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    model.load_state_dict(torch.load(os.path.join(res_dir, f"model.pth.tar"))["state_dict"])
    matching = get_matching(config, res_dir, model, device)
    model.eval()
    with torch.no_grad():
        loss_tot, n_iter = 0, 0
        predict_list = np.array([])
        labs_list = np.array([])
        for i, batch in enumerate(test_loader):
            input_seq, mask, y = batch
            input_seq = input_seq.view(-1, config['model']['num_steps'], config['model']['input_dim']).to(torch.float32).to(device)
            label = y.view(-1).long().to(device)
            mask = mask.view(-1, config['model']['num_steps']).int().to(device)
            loss, _, _, label, pred_indices, _, _, _ = iterate(input_seq, label, mask, config, model, optimizer,
                                                            n_iter=n_iter, mode='test')
            loss_tot += loss.item()
            n_iter += 1
            labs_list = np.concatenate((labs_list, label.detach().cpu().numpy()), axis=0)
            predict_list = np.concatenate((predict_list, matching[pred_indices.detach().cpu().numpy()]), axis=0)
    report = classification_report(labs_list, predict_list, digits=4, zero_division=0)
    conf_mat = confusion_matrix(labs_list, predict_list)
    with open(os.path.join(res_dir, f'test_recons_loss_best_acc.txt'), 'w') as f:
        f.write(f'recons_loss: {loss_tot / n_iter}')
    with open(os.path.join(res_dir, f'test_metrics_raw_best_acc.txt'), 'w') as f:
        f.write(report)
    with open(os.path.join(res_dir, f'test_conf_mat_best_acc.json'), 'w') as f:
        f.write(json.dumps({'conf_mat': conf_mat.tolist()}))
    print('... done!')


def train_oscnn(config, res_dir, tag):
    coerce_to_path_and_create_dir(res_dir)

    if torch.cuda.is_available():
        type_device = "cuda"
        nb_device = torch.cuda.device_count()
    else:
        type_device = "cpu"
        nb_device = None
    print("Using {} device, nb_device is {}".format(type_device, nb_device))

    seed = config['training'].get("rdm_seed", 621)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset = get_dataset(config["dataset"], 'train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training'].get("batch_size"),
        num_workers=config['training']['n_workers'],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    config['model']['dataset_name'] = config['dataset']['name']
    model = get_model(config["model"], exp_tag=tag)
    model.initt()
    config["N_params"] = get_ntrainparams(model.OS_CNN)
    print(f"Model has {config['N_params']} parameters.")
    with open(os.path.join(res_dir, "conf.json"), "w") as file:
        file.write(json.dumps(config, indent=4))

    model.fit(train_loader, nb_device)

    print('Training over!')
    print('Evaluate on test...')
    test_dataset = get_dataset(config["dataset"], 'test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['n_workers'],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    model.OS_CNN.load_state_dict(torch.load(os.path.join(res_dir, f"model.pth.tar")))
    predict_list, labs_list = model.predict(test_loader)

    report = classification_report(labs_list, predict_list, digits=4, zero_division=0)
    conf_mat = confusion_matrix(labs_list, predict_list)
    with open(os.path.join(res_dir, f'test_metrics_raw_acc.txt'), 'w') as f:
        f.write(report)
    with open(os.path.join(res_dir, f'test_conf_mat_acc.json'), 'w') as f:
        f.write(json.dumps({'conf_mat': conf_mat.tolist()}))
    print('... done!')


def train_svm(config, res_dir):
    clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-4)
    coerce_to_path_and_create_dir(res_dir)

    seed = config['training'].get("rdm_seed", 621)
    np.random.seed(seed)
    torch.manual_seed(seed)

    with open(os.path.join(res_dir, "conf.json"), "w") as file:
        file.write(json.dumps(config, indent=4))

    train_dataset = get_dataset(config["dataset"], 'train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training'].get("batch_size"),
        num_workers=config['training']['n_workers'],
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )

    val_dataset = get_dataset(config["dataset"], 'val')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training'].get("batch_size"),
        num_workers=config['training']['n_workers'],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    train_log_path = os.path.join(res_dir, "train_logs.json")
    train_logger = Logger(train_log_path)

    n_iter = 0
    best_val_loss = np.inf
    conf_matrix = ConfusionMatrix(config['model']['num_classes'], config['model']['num_classes'])
    conf_matrix_val = ConfusionMatrix(config['model']['num_classes'], config['model']['num_classes'])
    conf_matrix_test = ConfusionMatrix(config['model']['num_classes'], config['model']['num_classes'])
    for epoch in range(1, config['training']['n_epochs'] + 1):
        for i, batch in enumerate(train_loader):
            input_seq, mask, y = batch
            X = np.array(input_seq.view(-1,
                                        config['model']['num_steps'],
                                        config['model']['input_dim']
                                        ).flatten(1).to(torch.float32))
            Y = y.flatten()
            clf.partial_fit(X, Y, classes=range(config['model']['num_classes']))
            Y_pred = clf.predict(X)
            conf_matrix.add(Y_pred, Y)
            n_iter += 1

            if n_iter % config['training']['print_step'] == 0:
                conf_matrix.purity_assignment()
                curr_acc = conf_matrix.get_acc()
                curr_acc_per_class = conf_matrix.get_acc_per_class()
                conf_matrix.reset()
                print('   Iter {}: Acc {:.2f}%, Mean Acc {:.2f}%'.format(n_iter,
                                                                                      curr_acc,
                                                                                      np.mean(curr_acc_per_class)))
                print('      Loss per class {}'.format(' '.join(['{:.2f}'.format(curr_acc)
                                                                 for curr_acc in curr_acc_per_class])))
                train_metrics = {'acc': curr_acc,
                                 'mean_acc': np.mean(curr_acc_per_class),
                                 'acc_per_class': curr_acc_per_class}
                train_logger.update(train_metrics, n_iter)

            if n_iter % config['training']['valid_step'] == 0:
                print('Validation...')
                conf_matrix_val.reset()
                for i, batch in enumerate(val_loader):
                    input_seq, mask, y = batch
                    X = np.array(input_seq.view(-1,
                                                config['model']['num_steps'],
                                                config['model']['input_dim']
                                                ).flatten(1).to(torch.float32))
                    Y = y.flatten()
                    Y_pred = clf.predict(X)
                    conf_matrix_val.add(Y_pred, Y)
                conf_matrix_val.purity_assignment()
                curr_acc = conf_matrix_val.get_acc()
                curr_acc_per_class = conf_matrix_val.get_acc_per_class()
                print('   Val Iter {}: Acc {:.2f}%, Mean Acc {:.2f}%'.format(n_iter,
                                                                         curr_acc,
                                                                         np.mean(curr_acc_per_class)))
                val_loss_to_consider = -np.mean(curr_acc_per_class)

                if val_loss_to_consider < best_val_loss:
                    best_val_loss = val_loss_to_consider
                    joblib.dump(clf, os.path.join(
                            res_dir, "model.pkl"
                        ), compress=9)

    print('Training over!')
    print('Evaluate on test...')
    test_dataset = get_dataset(config["dataset"], 'test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['n_workers'],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    clf = joblib.load(os.path.join(
                            res_dir, "model.pkl"
                        ))
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            input_seq, mask, y = batch
            X = np.array(input_seq.view(-1,
                                        config['model']['num_steps'],
                                        config['model']['input_dim']
                                        ).flatten(1).to(torch.float32))
            Y = y.flatten()
            Y_pred = clf.predict(X)
            conf_matrix_test.add(Y_pred, Y)
    conf_matrix_test.purity_assignment()
    curr_acc = conf_matrix_test.get_acc()
    curr_acc_per_class = conf_matrix_test.get_acc_per_class()
    print('Test: Acc {:.2f}%, Mean Acc {:.2f}%'.format(curr_acc, np.mean(curr_acc_per_class)))
    with open(os.path.join(res_dir, f'test_scores.txt'), 'w') as f:
        f.write(f'mean_acc: {np.mean(curr_acc_per_class)}, acc:{curr_acc}')
    with open(os.path.join(res_dir, f'test_conf_mat.json'), 'w') as f:
        f.write(json.dumps({'conf_mat': conf_matrix_test.conf.tolist()}))
    print('... done!')


def train_rf(config, res_dir):
    coerce_to_path_and_create_dir(res_dir)

    seed = config['training'].get("rdm_seed", 621)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset = get_dataset(config["dataset"], 'train')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training'].get("batch_size"),
        num_workers=config['training']['n_workers'],
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )
    rf_model = RandomForestClassifier(warm_start=True, n_estimators=1, random_state=seed)
    conf_matrix_test = ConfusionMatrix(config['model']['num_classes'], config['model']['num_classes'])

    for i, batch in enumerate(train_loader):
        input_seq, mask, y = batch
        X = np.array(input_seq.view(-1,
                                    config['model']['num_steps'],
                                    config['model']['input_dim']
                                    ).flatten(1).to(torch.float32))
        Y = y.flatten()
        if torch.unique(Y).shape[0] == config['model']['num_classes']:
            rf_model.fit(X, Y)
            rf_model.n_estimators += 1
        if i == 99:
            break
    joblib.dump(rf_model, os.path.join(
        res_dir, "model.pkl"
    ), compress=9)
    rf_model = joblib.load(os.path.join(
        res_dir, "model.pkl"
    ))
    print(f'N_params: {np.sum([estimator.tree_.max_depth for estimator in rf_model.estimators_])*2}')
    print('Training over!')
    print('Evaluate on test...')
    test_dataset = get_dataset(config["dataset"], 'test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        num_workers=config['training']['n_workers'],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            input_seq, mask, y = batch
            X = np.array(input_seq.view(-1,
                                        config['model']['num_steps'],
                                        config['model']['input_dim']
                                        ).flatten(1).to(torch.float32))
            Y = y.flatten()
            Y_pred = rf_model.predict(X)
            conf_matrix_test.add(Y_pred, Y)
    conf_matrix_test.purity_assignment()
    curr_acc = conf_matrix_test.get_acc()
    curr_acc_per_class = conf_matrix_test.get_acc_per_class()
    print('Test: Acc {:.2f}%, Mean Acc {:.2f}%'.format(curr_acc, np.mean(curr_acc_per_class)))
    with open(os.path.join(res_dir, f'test_scores.txt'), 'w') as f:
        f.write(f'mean_acc: {np.mean(curr_acc_per_class)}, acc:{curr_acc}')
    with open(os.path.join(res_dir, f'test_conf_mat.json'), 'w') as f:
        f.write(json.dumps({'conf_mat': conf_matrix_test.conf.tolist()}))
    print('... done!')


def train_ltae(config, res_dir):
    coerce_to_path_and_create_dir(res_dir)

    if torch.cuda.is_available():
        type_device = "cuda"
        nb_device = torch.cuda.device_count()
    else:
        type_device = "cpu"
        nb_device = None
    print("Using {} device, nb_device is {}".format(type_device, nb_device))
    device = torch.device(config['training'].get("device", 'cuda'))

    seed = config['training'].get("rdm_seed", 621)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset = get_dataset(config["dataset"], 'train')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training'].get("batch_size"),
        num_workers=config['training']['n_workers'],
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )

    val_dataset = get_dataset(config["dataset"], 'val')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training'].get("batch_size"),
        num_workers=config['training']['n_workers'],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    model = get_model(config["model"])
    model = torch.nn.DataParallel(model.to(device), device_ids=range(nb_device))
    config["N_params"] = get_ntrainparams(model)
    print(f"Model has {config['N_params']} parameters.")
    with open(os.path.join(res_dir, "conf.json"), "w") as file:
        file.write(json.dumps(config, indent=4))

    optimizer = torch.optim.Adam(model.parameters(), **config['training']['optimizer'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **config['training']['scheduler'])

    train_log_path = os.path.join(res_dir, "train_logs.json")
    val_log_path = os.path.join(res_dir, "val_logs.json")
    train_logger = Logger(train_log_path)
    val_logger = Logger(val_log_path)

    n_iter = 0
    best_val_loss = np.inf
    loss_meter = tnt.meter.AverageValueMeter()
    conf_matrix = ConfusionMatrix(config['model']['num_classes'], config['model']['num_classes'])
    for epoch in range(1, config['training']['n_epochs'] + 1):
        for i, batch in enumerate(train_loader):
            input_seq, mask, y = batch
            input_seq = input_seq.view(-1, config['model']['num_steps'], config['model']['input_dim']).to(torch.float32)
            label = y.view(-1).long()
            mask = mask.view(-1, config['model']['num_steps']).int()
            loss, class_count, _, label, pred_indices, output_seq, _, _ = iterate(input_seq, label, mask, config, model, optimizer, n_iter=n_iter)
            n_iter += 1
            scheduler.step()
            loss_meter.add(loss.item())
            conf_matrix.add(pred_indices, label)

            if n_iter % config['training']['print_step'] == 0:
                conf_matrix.purity_assignment()
                curr_acc = conf_matrix.get_acc()
                curr_acc_per_class = conf_matrix.get_acc_per_class()
                conf_matrix.reset()
                print('   Iter {}: Loss {:.3f}, Acc {:.2f}%, Mean Acc {:.2f}%'.format(n_iter,
                                                                                      loss_meter.value()[0],
                                                                                      curr_acc,
                                                                                      np.mean(curr_acc_per_class)))
                print('      Loss per class {}'.format(' '.join(['{:.2f}'.format(curr_acc)
                                                                 for curr_acc in curr_acc_per_class])))
                train_metrics = {'loss': loss_meter.value()[0],
                                 'acc': curr_acc,
                                 'mean_acc': np.mean(curr_acc_per_class),
                                 'acc_per_class': curr_acc_per_class,
                                 'lr': optimizer.param_groups[0]['lr']}
                train_logger.update(train_metrics, n_iter)
                loss_meter.reset()

            if n_iter % config['training']['valid_step'] == 0:
                matching = np.array(list(range(config['model']['num_classes'])))
                print('Validation...')
                model.eval()
                val_metrics = validate_ltae(val_loader, model, optimizer, config, mode='val', matching=matching, n_iter=n_iter)
                val_loss_to_consider = -val_metrics['mean_acc']
                val_logger.update(val_metrics, n_iter)
                model.train()
                if val_loss_to_consider < best_val_loss:
                    best_val_loss = val_loss_to_consider
                    torch.save(
                        {
                            "iter": n_iter,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        os.path.join(
                            res_dir, "model.pth.tar"
                        ))
                    with open(os.path.join(res_dir, "val_metrics.json"), "w") as file:
                        file.write(json.dumps(val_metrics, indent=4))

    print('Training over!')
    print('Evaluate on test...')
    test_dataset = get_dataset(config["dataset"], 'test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['n_workers'],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    model.load_state_dict(torch.load(os.path.join(res_dir, f"model.pth.tar"))["state_dict"])
    matching = get_matching(config, res_dir, model, device)
    model.eval()
    with torch.no_grad():
        loss_tot, n_iter = 0, 0
        predict_list = np.array([])
        labs_list = np.array([])
        for i, batch in enumerate(test_loader):
            input_seq, mask, y = batch
            input_seq = input_seq.view(-1, config['model']['num_steps'], config['model']['input_dim']).to(torch.float32).to(device)
            label = y.view(-1).long().to(device)
            mask = mask.view(-1, config['model']['num_steps']).int().to(device)
            loss, _, _, label, pred_indices, _, _, _ = iterate(input_seq, label, mask, config, model, optimizer,
                                                            n_iter=n_iter, mode='test')
            loss_tot += loss.item()
            n_iter += 1
            labs_list = np.concatenate((labs_list, label.detach().cpu().numpy()), axis=0)
            predict_list = np.concatenate((predict_list, matching[pred_indices.detach().cpu().numpy()]), axis=0)
    report = classification_report(labs_list, predict_list, digits=4, zero_division=0)
    conf_mat = confusion_matrix(labs_list, predict_list)
    with open(os.path.join(res_dir, f'test_recons_loss_acc.txt'), 'w') as f:
        f.write(f'recons_loss: {loss_tot / n_iter}')
    with open(os.path.join(res_dir, f'test_metrics_raw_acc.txt'), 'w') as f:
        f.write(report)
    with open(os.path.join(res_dir, f'test_conf_mat_acc.json'), 'w') as f:
        f.write(json.dumps({'conf_mat': conf_mat.tolist()}))
    print('... done!')


def train_tapnet(config, res_dir):
    coerce_to_path_and_create_dir(res_dir)

    if torch.cuda.is_available():
        type_device = "cuda"
        nb_device = torch.cuda.device_count()
    else:
        type_device = "cpu"
        nb_device = None
    print("Using {} device, nb_device is {}".format(type_device, nb_device))
    device = torch.device(config['training'].get("device", 'cuda'))

    seed = config['training'].get("rdm_seed", 621)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset = get_dataset(config["dataset"], 'train')
    train_loader_unshuffle = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=config['training']['n_workers'],
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training'].get("batch_size"),
        num_workers=config['training']['n_workers'],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    val_dataset = get_dataset(config["dataset"], 'val')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training'].get("batch_size"),
        num_workers=config['training']['n_workers'],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    if config['model']['rp_params'][0] < 0:
        dim = config['model']['input_dim']
        config['model']['rp_params'] = [3, math.floor(dim / (3 / 2))]
    else:
        dim = config['model']['input_dim']
        config['model']['rp_params'][1] = math.floor(dim / config['model']['rp_params'][1])

    config['model']['rp_params'] = [int(l) for l in config['model']['rp_params']]

    # update dilation parameter
    if config['model']['dilation'] == -1:
        config['model']['rp_params'] = math.floor(config['model']['num_steps'] / 64)


    model = get_model(config["model"])
    model = torch.nn.DataParallel(model.to(device), device_ids=range(nb_device))

    config["N_params"] = get_ntrainparams(model)
    print(f"Model has {config['N_params']} parameters.")
    with open(os.path.join(res_dir, "conf.json"), "w") as file:
        file.write(json.dumps(config, indent=4))

    optimizer = torch.optim.Adam(model.parameters(), **config['training']['optimizer'])

    n_iter = 0
    curr_loss, curr_acc = 0, 0
    loss_list = [sys.maxsize]
    best_acc = 0.
    t = time.time()
    for epoch in range(config['training']['num_epochs']):
        for i, batch in enumerate(train_loader):
            input_seq, mask, y = batch
            input_seq = input_seq.view(-1, config['model']['num_steps'], config['model']['input_dim']).to(torch.float32).permute(0, 2, 1)
            label = y.view(-1).long().to(device)
            model.train()
            optimizer.zero_grad()
            output = model(input_seq, label, mode='train')
            loss_train = torch.nn.functional.cross_entropy(output, torch.squeeze(label))
            loss_list.append(loss_train.item())

            acc_train = model.module.accuracy(output, label)
            loss_train.backward()
            optimizer.step()
            n_iter += 1
            curr_loss += loss_train.item()
            curr_acc += acc_train.item()
            if n_iter % config['training']['train_iter'] == 0:
                print('Iter: {:06d}'.format(n_iter),
                      'loss_train: {:.6f}'.format(curr_loss / config['training']['train_iter']),
                      'acc_train: {:.4f}'.format(curr_acc / config['training']['train_iter']),
                      'time: {:.4f}s'.format(time.time() - t))
                t = time.time()
                curr_loss, curr_acc = 0, 0
            if n_iter % config['training']['val_iter'] == 0:
                print('Validation...')
                val_acc, val_loss, n_iter_val = 0, 0, 0
                with torch.no_grad():
                    model.module.get_prototypes(train_loader_unshuffle, device)
                    for j, batch in enumerate(val_loader):
                        input_seq, mask, y = batch
                        input_seq = input_seq.view(-1, config['model']['num_steps'], config['model']['input_dim']).to(torch.float32).permute(0, 2, 1)
                        label = y.view(-1).long().to(device)
                        model.eval()
                        output = model(input_seq, label, mode='val')
                        loss_train = torch.nn.functional.cross_entropy(output, torch.squeeze(label))
                        loss_list.append(loss_train.item())
                        n_iter_val += 1
                        acc_train = model.module.accuracy(output, label)
                        val_loss += loss_train.item()
                        val_acc += acc_train.item()
                val_loss /= n_iter_val
                val_acc /= n_iter_val
                if val_acc > best_acc:
                    torch.save(model.module.x_proto, os.path.join(res_dir, 'x_proto.pt'))
                    best_acc = val_acc
                    torch.save(
                        {
                            "iter": n_iter,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        os.path.join(
                            res_dir, "model.pth.tar"
                        ))

                print('loss_val: {:.8f}'.format(val_loss),
                      'acc_val: {:.4f}'.format(val_acc))

            if n_iter >= config['training']['max_iter']:
                break
        if n_iter >= config['training']['max_iter']:
            break

    print('Training over!')
    print('Evaluate on test...')
    test_dataset = get_dataset(config["dataset"], 'test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['n_workers'],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    model.load_state_dict(torch.load(os.path.join(res_dir, f"model.pth.tar"))["state_dict"])
    model.eval()
    with torch.no_grad():
        model.module.get_prototypes(train_loader_unshuffle, device)
    with torch.no_grad():
        predict_list = np.array([])
        labs_list = np.array([])
        for i, batch in enumerate(test_loader):
            input_seq, mask, y = batch
            input_seq = input_seq.view(-1, config['model']['num_steps'], config['model']['input_dim']).to(torch.float32).permute(0, 2, 1)
            label = y.view(-1).long().to(device)
            optimizer.zero_grad()
            output = model(input_seq, label, mode='test')
            labs_list = np.concatenate((labs_list, label.detach().cpu().numpy()), axis=0)
            predict_list = np.concatenate((predict_list, np.argmax(output.detach().cpu().numpy(), axis=1)), axis=0)
            if i % 500 == 0:
                print(i)
    report = classification_report(labs_list, predict_list, digits=4, zero_division=0)
    conf_mat = confusion_matrix(labs_list, predict_list)
    with open(os.path.join(res_dir, f'test_metrics_raw_acc.txt'), 'w') as f:
        f.write(report)
    with open(os.path.join(res_dir, f'test_conf_mat_acc.json'), 'w') as f:
        f.write(json.dumps({'conf_mat': conf_mat.tolist()}))
    print('... done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline to train a NN model specified by a YML config")
    parser.add_argument("-t", "--tag", nargs="?", type=str, required=True, help="Run tag of the experiment")
    parser.add_argument("-c", "--config", nargs="?", type=str, required=True, help="Config file name")
    args = parser.parse_args()

    print(f'Experiment tag is {args.tag}.')
    print(f'Configuration file is {args.config}.')

    assert args.tag is not None and args.config is not None
    config = coerce_to_path_and_check_exist(CONFIGS_PATH / args.config)
    with open(config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    dataset = cfg["dataset"]["name"]
    tag = args.tag
    res_dir = RESULTS_PATH / dataset / tag
    if 'pastis' in dataset:
        res_dir = res_dir / f'Fold_{cfg["dataset"]["fold"]}'
    if cfg["model"]["name"] == 'dtits':
        train(cfg, res_dir)
    elif cfg["model"]["name"] == 'ltae':
        train_ltae(cfg, res_dir)
    elif cfg["model"]["name"] == 'oscnn':
        train_oscnn(cfg, res_dir, tag)
    elif cfg["model"]["name"] == 'tapnet':
        train_tapnet(cfg, res_dir)
    elif cfg["model"]["name"] == 'svm':
        train_svm(cfg, res_dir)
    elif cfg["model"]["name"] == 'rf':
        train_rf(cfg, res_dir)

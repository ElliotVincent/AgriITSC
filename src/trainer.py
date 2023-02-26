import argparse

from sklearn.metrics import classification_report, confusion_matrix
import torchnet as tnt
import yaml

from src.utils.metrics import ConfusionMatrix, CustomMeter
from src.utils.paths import CONFIGS_PATH
from src.utils.utils import *
from src.utils.constants import *


def iterate(input_seq, label, mask, config, model, optimizer, mode='train', n_iter=0):
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
    tv_h = torch.pow(model.module.prototypes[:, 1:, :] - model.module.prototypes[:, :-1, :], 2).mean()
    logits = - loss
    proto_count = torch.stack([(indices == k).float().sum() for k in range(config['model']['num_prototypes'])])
    training_loss = ((mask * ((output_seq - input_seq) ** 2).mean(2)).sum(1) / torch.where(mask.sum(1) == 0, torch.ones_like(mask.sum(1)), mask.sum(1))).mean()
    training_loss_backward = training_loss + LAMBDA * tv_h
    if config['model'].get('supervised', False) and config['training']['ce_activ'] and n_iter >= config['training']['curriculum'][2] and mode == 'train':
        training_loss = training_loss + MU * torch.nn.functional.cross_entropy(logits, label, reduction='none').mean()
    if mode == 'train':
        training_loss_backward.backward()
        optimizer.step()
    recons_loss, pred_indices = torch.max(logits, dim=1)
    if mode != 'train':
        recons_loss = (-recons_loss / config['model']['input_dim'] / torch.where(mask.sum(1) == 0, torch.ones_like(mask.sum(1)), mask.sum(1))).mean()
        training_loss = recons_loss
    class_count = [(label == k).float().sum() for k in range(config['model']['num_classes'])]
    class_count = [1. if class_count[k] > 0 else 0. for k in range(config['model']['num_classes'])]
    return training_loss, class_count, proto_count, label, pred_indices, output_seq, tv_h


def validate(val_loader, model, optimizer, config, mode='test', matching=None, n_iter=0):
    loss_meter = tnt.meter.AverageValueMeter()
    tvh_meter = tnt.meter.AverageValueMeter()
    conf_matrix = ConfusionMatrix(config['model']['num_classes'], config['model']['num_prototypes'])

    for i, batch in enumerate(val_loader):
        input_seq, mask, y = batch
        input_seq = input_seq.view(-1, config['model']['num_steps'], config['model']['input_dim']).to(torch.float32)
        label = y.view(-1).long()
        mask = mask.view(-1, config['model']['num_steps']).int()
        input_seq = input_seq.to(torch.float32)
        with torch.no_grad():
            loss, class_count, proto_count, label, pred_indices, _, tv_h = iterate(input_seq, label, mask,
                                                            config, model, optimizer, mode=mode, n_iter=n_iter)
        loss_meter.add(loss.item())
        tvh_meter.add(tv_h.item())
        conf_matrix.add(pred_indices, label)
    conf_matrix.purity_assignment(matching=matching)
    curr_acc = conf_matrix.get_acc()
    curr_acc_per_class = conf_matrix.get_acc_per_class()
    val_metrics = {'loss': loss_meter.value()[0],
                   'acc': curr_acc,
                   'mean_acc': np.mean(curr_acc_per_class),
                   'tvh': tvh_meter.value()[0],
                   'acc_per_class': curr_acc_per_class,
                   'conf_matrix': conf_matrix.purity_conf.tolist(),
                   }
    print('   Val Loss: {:.3f}, Val Acc: {:.2f}, Val mAcc: {:.2f}'.format(val_metrics['loss'],
                                                                          val_metrics['acc'], val_metrics['mean_acc']))
    return val_metrics


def get_matching(config, res_dir, model, device):
    print('Getting the prototype assignment...')
    path = Path(os.path.join(res_dir, 'matching.json'))
    if path.exists():
        matching = np.array(json.load(open(path, 'r'))['matching'])
        print('... done!')
        return matching
    elif config['model']['supervised']:
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
                loss, _, _, label, pred_indices, _, _ = iterate(input_seq, label, mask, config, model, None, n_iter=0,
                                                                mode='test')
                np.add.at(conf_mat, config['model'][
                    'num_prototypes'] * label.detach().cpu().numpy() + pred_indices.detach().cpu().numpy(), 1)
        conf_mat = np.reshape(conf_mat, (config['model']['num_classes'], config['model']['num_prototypes']))
        matching = np.argmax(conf_mat, axis=0)
        with open(os.path.join(res_dir, f"matching.json"), 'w') as file:
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

    proto_init = initialize_prototypes(config, train_loader, device)
    config["model"]["sample"] = proto_init
    model = get_model(config["model"])
    model = torch.nn.DataParallel(model.to(device), device_ids=[0, 1, 2, 3])
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
    proto_count_meter = CustomMeter(num_classes=config['model']['num_prototypes'])
    tvh_meter = tnt.meter.AverageValueMeter()
    curriculum_scheduler = TransfoScheduler(config)

    for epoch in range(1, config['training']['n_epochs'] + 1):
        print('Epoch {}/{}:'.format(epoch, config['training']['n_epochs']))
        for i, batch in enumerate(train_loader):
            input_seq, mask, y = batch
            input_seq = input_seq.view(-1, config['model']['num_steps'], config['model']['input_dim']).to(torch.float32)
            label = y.view(-1).long()
            mask = mask.view(-1, config['model']['num_steps']).int()
            loss, class_count, proto_count, label, pred_indices, output_seq, tv_h = iterate(input_seq, label, mask, config, model, optimizer, n_iter=n_iter)
            n_iter += 1
            n_iter_since_new += 1
            if curriculum_scheduler.is_complete:
                n_iter_since_complete += 1

            loss_meter.add(loss.item())
            tvh_meter.add(tv_h.item())
            conf_matrix.add(pred_indices, label)
            proto_count_meter.add(proto_count, [0 for _ in range(config['model']['num_prototypes'])])

            if n_iter % config['training']['check_cluster_step'] == 0:
                curr_prop = proto_count_meter.value(mode='density')
                if not config['model']['supervised']:
                    model.module.reassign_empty_clusters(curr_prop)
                proto_count_meter.reset()

            if n_iter % config['training']['print_step'] == 0:
                matching = conf_matrix.purity_assignment()
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
                                 'tvh': tvh_meter.value()[0],
                                 'acc_per_class': curr_acc_per_class,
                                 'lr': optimizer.param_groups[0]['lr'],
                                 'proportions': curr_prop}
                train_logger.update(train_metrics, n_iter)
                loss_meter.reset()
                tvh_meter.reset()
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
        loss_tot = 0
        predict_list = np.array([])
        labs_list = np.array([])
        for i, batch in enumerate(test_loader):
            input_seq, mask, y = batch
            input_seq = input_seq.view(-1, config['model']['num_steps'], config['model']['input_dim']).to(torch.float32).to(device)
            label = y.view(-1).long().to(device)
            mask = mask.view(-1, config['model']['num_steps']).int().to(device)
            loss, _, _, label, pred_indices, _, _ = iterate(input_seq, label, mask, config, model, optimizer,
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
    res_dir = RESULTS_PATH / dataset / args.tag
    train(cfg, res_dir)

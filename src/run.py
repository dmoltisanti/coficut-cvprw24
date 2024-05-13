import argparse
import sys
from pathlib import Path

from dice_loss import DiceLoss
from unet import Unet

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

from dataset import get_global_info, VostAugDataset, basic_transforms, basic_mask_transforms, CoficutDataset


def create_parser():
    parser = argparse.ArgumentParser(add_help=True, description='Code to run the model presented in the CVPRW 24 '  
                                                                'paper "Coarse or Fine? Recognising Action End '
                                                                'States without Labels"')
    parser.add_argument('dataset_folder', type=Path, help='Path to the dataset folder containing'
                                                          'the VOST csv files')
    parser.add_argument('original_data_path', type=Path, help='Path to the original images of the VOST'
                                                              'dataset')
    parser.add_argument('augmented_data_path', type=Path, help='Path to the images augment from VOST')
    parser.add_argument('output_path', type=Path, help='Where you want to save logs, checkpoints and '
                                                       'results')
    parser.add_argument('--train_batch', default=32, type=int, help='Training batch size')
    parser.add_argument('--train_workers', default=12, type=int, help='Number of workers for the training '
                                                                      'data loader')
    parser.add_argument('--test_batch', default=32, type=int, help='Testing batch size')
    parser.add_argument('--test_workers', default=12, type=int, help='Number of workers for the testing '
                                                                     'data loader')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout for the model')
    parser.add_argument('--weight_decay', default=5e-5, type=float, help='Weight decay for the optimiser')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of training epochs')
    parser.add_argument('--run_tags', default=['lr', 'train_batch', 'dropout', 'hidden_units',
                                               'loss'], action='append',
                        help='What arguments should be used to create a run id, i.e. an output folder name')
    parser.add_argument('--tag', default=None, type=str, help='Any additional tag to be added to the run id')
    parser.add_argument('--test_frequency', default=20, type=int, help='How often the model should be '
                                                                       'evaluated on the test set')
    parser.add_argument('--hidden_units', default=',', type=str, help='Number of layers and hidden units '
                                                                      'for the model''s MLP, to be '
                                                                      'specified as a string of comma-'
                                                                      'separated integers. For example, '
                                                                      '512,512,512 will create 3 hidden '
                                                                      'layers, each with 512 hidden units')

    parser.add_argument('--split_vost_by', default='change_ratio', type=str,
                        choices=['change_ratio', 'n_bits'], help='How to assign pseudo labels to the VOST training'
                                                                 ' dataset ')
    parser.add_argument('--test_n_aug_samples', default=5, help='How many samples to use for testing')
    parser.add_argument('--loss', default='dice', type=str, choices=['bce', 'reg', 'dice'],
                        help='Loss function')
    parser.add_argument('--test_only', action='store_true', help='Only test the model')
    parser.add_argument('--checkpoint_path', type=Path, default=None,
                        help='Path to the model checkpoint, useful to test a pre-trained model')
    parser.add_argument('--coficut_path', type=Path, default=None, help='Path to the folder'
                                                                        'containing the COFICUT dataset images')

    return parser


def setup_data(args):
    train_df = pd.read_csv(args.dataset_folder / 'train_df.csv')
    test_df = pd.read_csv(args.dataset_folder / 'test_df.csv')
    global_info = get_global_info(args.dataset_folder / 'global_info.csv')
    train_trans = basic_transforms(random_crop=True)
    test_trans = basic_transforms()
    mask_trans = basic_mask_transforms()
    extra_val_loaders = {}
    test_n_aug_samples = args.test_n_aug_samples
    test_n_aug_samples = test_n_aug_samples if test_n_aug_samples == 'all' else int(test_n_aug_samples)
    train_dataset = VostAugDataset(train_df, global_info, args.original_data_path,
                                   args.augmented_data_path, train_trans, training=True,
                                   split_by=args.split_vost_by, n_aug_samples='all', mask_transforms=mask_trans)

    seen_objects = train_dataset.object_names

    # for consistent testing we always split the VOST test set based on the change_ratio as in the paper
    test_dataset = VostAugDataset(test_df, global_info, args.original_data_path,
                                  args.augmented_data_path, test_trans, training=False, split_by='change_ratio',
                                  mask_transforms=mask_trans, n_aug_samples=test_n_aug_samples, crop_to_mask=False,
                                  seen_objects=seen_objects)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True,
                              pin_memory=True, num_workers=args.train_workers)

    test_loader = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False, pin_memory=True,
                             num_workers=args.test_workers)

    if args.coficut_path is not None:
        coficut_dataset = CoficutDataset(args.coficut_path, test_trans, seen_objects)
        coficut_loader = DataLoader(coficut_dataset, batch_size=args.test_batch, shuffle=False, pin_memory=True,
                                    num_workers=args.test_workers)
        extra_val_loaders['coficut'] = coficut_loader

    return train_loader, test_loader, extra_val_loaders


def setup_model(args, cuda=True):
    model = Unet(backbone='resnet50', in_channels=3, num_classes=1, pretrained=True, hidden_units=args.hidden_units,
                 backbone_kwargs={'drop_rate': args.dropout})

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path)['model_state']
        msg = model.load_state_dict(checkpoint)
        print(f'Loaded checkpoint from {args.checkpoint_path}.\nLoading message: {msg}')

    if torch.cuda.is_available() and cuda:
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    return model


def setup_optimiser(model, args):
    optim_params = model.parameters()
    return torch.optim.Adam(optim_params, lr=args.lr, weight_decay=args.weight_decay)


def setup_criterion(args):
    if args.loss == 'dice':
        dice_loss = DiceLoss(from_logits=False, reduction='mean')

        def dice_wrapper(x, labels):
            unet_labels = labels['unet_target'].squeeze(1)
            change_ratio = labels['change_ratio']
            x, r = x[0], x[2]  # x is a tuple
            x = x.squeeze(1)
            dice = dice_loss(x, unet_labels)

            if change_ratio.shape != r.shape:
                change_ratio = change_ratio.unsqueeze(1)

            ratio_loss = F.l1_loss(r, change_ratio)

            return dict(dice_loss=dice, ratio_loss=ratio_loss)

        return dice_wrapper
    elif args.loss == 'reg':
        def l1_wrapper(x, labels):
            change_ratio = labels['change_ratio']
            r = x[2]  # x is a tuple

            if change_ratio.shape != r.shape:
                change_ratio = change_ratio.unsqueeze(1)

            ratio_loss = F.l1_loss(r, change_ratio)
            return ratio_loss

        return l1_wrapper
    elif args.loss == 'bce':
        def bce_wrapper(x, labels):
            pseudo_labels = labels['pseudo_label']
            r = x[2]  # x is a tuple

            if pseudo_labels.shape != r.shape:
                pseudo_labels = pseudo_labels.unsqueeze(1)

            bce_loss = F.binary_cross_entropy_with_logits(r, pseudo_labels)
            return bce_loss

        return bce_wrapper
    else:
        raise ValueError(f'Loss {args.loss} not implemented yet')


def to_cuda(data_tuple):
    if torch.cuda.is_available():
        cuda_tuple = []

        for x in data_tuple:
            if isinstance(x, torch.Tensor):
                xc = x.cuda()
            elif isinstance(x, dict):
                xc = {k: v.cuda() if isinstance(v, torch.Tensor) and ('no_cuda' not in k) else v for k, v in x.items()}
            elif isinstance(x, list):
                xc = [item.cuda() if isinstance(item, torch.Tensor) else item for item in x]
            else:
                raise RuntimeError(f'Review this for type {type(x)}')

            cuda_tuple.append(xc)

        return tuple(cuda_tuple)
    else:
        return data_tuple


def train(args, loader, model, optimiser, criterion):
    model.train()
    train_loss, train_metadata = run_through_loader(args, model, loader, criterion, optimiser, tag='Training')

    return train_loss, train_metadata


def run_through_loader(args, model, loader, criterion, optimiser, tag, optimise=True):
    bar = tqdm(desc=tag, file=sys.stdout, total=len(loader))
    loss_batches = []
    all_metadata = {}

    for x_tuple in loader:
        batch_output, loss, batch_labels, batch_metadata = process_batch(x_tuple, model, criterion, optimise, args)

        if optimise:
            if isinstance(loss, dict):
                if len(loss) == 1:
                    loss_sum = list(loss.values())[0]
                else:
                    loss_sum = sum(loss.values())

                loss_sum.backward()
            else:
                loss.backward()

            optimiser.step()
            optimiser.zero_grad()

        if loss is not None:
            if isinstance(loss, dict):
                l_str = ', '.join([f'{k}={v.detach().item():0.4f}' for k, v in loss.items()])

                if len(loss) > 1:
                    l_str = f'Loss sum={loss_sum.detach().item():0.4f}, {l_str}'

                bar.set_description(f'{tag} {l_str}')
                loss_batches.append({k: v.detach().item() for k, v in loss.items()})

                if len(loss) > 1:
                    loss_batches.append({'loss_sum': loss_sum.detach().item()})
            else:
                bar.set_description(f'{tag} loss: {loss.detach().item():0.4f}')
                loss_batches.append(loss.detach().item())

        bar.update()

        if all_metadata is None and batch_metadata is not None:
            all_metadata = {k: [] for k in batch_metadata[0].keys()}

        for k in all_metadata.keys():
            for bi in batch_metadata:
                bv = bi[k]

                if isinstance(bv, list):
                    all_metadata[k].extend(bv)
                elif isinstance(bv, torch.Tensor):
                    all_metadata[k].extend(bv.tolist())
                else:
                    raise RuntimeError(f'Define how to combine metadata for type {type(bv)}')

    if isinstance(loss_batches[0], dict):
        avg_loss = {}

        for lb in loss_batches:
            for k, v in lb.items():
                if k in avg_loss:
                    avg_loss[k].append(v)
                else:
                    avg_loss[k] = [v]

        avg_loss = {k: torch.Tensor(v).mean().item() for k, v in avg_loss.items()}
    else:
        avg_loss = torch.Tensor(loss_batches).mean().item()

    return avg_loss, all_metadata


def process_batch(x_tuple, model, criterion, training, args):
    data, labels, metadata = x_tuple
    data, labels = to_cuda((data, labels))

    if args.loss == 'dice':
        output = model(data)
    else:
        output = model(data, encoder_only=True)

    loss = criterion(output, labels) if training else None

    return output, loss, labels, metadata


def is_best_metric(metric, summary_dict, best_metric, min_max, set_):
    metric_d = metric.replace(f'{set_}_', '')

    if min_max[metric_d] == 'max' and summary_dict[metric] > best_metric:
        return True
    if min_max[metric_d] == 'min' and summary_dict[metric] < best_metric:
        return True

    assert all(v in ('min', 'max') for v in min_max.values())
    return False


def evaluate(model, test_loader, extra_val_loaders):
    model.eval()

    with torch.no_grad():
        res_dict, loss, res_df, res_min_max, res_file = test_model(model, test_loader)

        if 'coficut' in extra_val_loaders:
            coficut_loader = extra_val_loaders['coficut']
            coficut_res_dict, _, _, coficut_min_max, _ = test_model(model, coficut_loader, is_coficut=True)
            res_dict.update(coficut_res_dict)
            res_min_max.update(coficut_min_max)

        return res_dict, loss, res_df, res_min_max, res_file


def test_model(model, test_loader, is_coficut=False):
    all_labels = []
    all_predictions = []
    all_objects = []
    all_original_mask = []
    seen_mask = []

    for x_tuple in tqdm(test_loader, file=sys.stdout, desc=f'Testing (is_coficut={is_coficut})'):
        data, labels, batch_metadata = x_tuple
        data, labels = to_cuda((data, labels))
        batch_output = model(data, encoder_only=True)
        c = batch_output[2]
        all_predictions.append(c.cpu().numpy())

        if is_coficut:
            all_labels.append(labels['label_num'].cpu().numpy())
            all_objects.append(batch_metadata['object_idx'].cpu().numpy())
        else:
            l = labels['pseudo_label']
            all_labels.append(l.cpu().numpy().astype(int))
            all_original_mask.extend(batch_metadata['is_original'])

        seen = batch_metadata['seen_in_training']
        seen_mask.append(seen)

    res_dict = {}
    seen_mask = np.concatenate(seen_mask)
    all_original_mask = np.array(all_original_mask)
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)

    if is_coficut:
        map_m = coficut_map(all_labels, all_predictions)

        if seen_mask.any():
            map_m_seen = coficut_map(all_labels[seen_mask], all_predictions[seen_mask])
            res_dict['map_m_seen'] = map_m_seen

        if (~seen_mask).any():
            map_m_unseen = coficut_map(all_labels[~seen_mask], all_predictions[~seen_mask])
            res_dict['map_m_unseen'] = map_m_unseen

        res_dict['map_m'] = map_m
    else:
        map_m = vost_map(all_labels, all_predictions, all_original_mask)
        res_dict['map_m'] = map_m

        if seen_mask.any():
            map_m_seen = vost_map(all_labels[seen_mask], all_predictions[seen_mask], all_original_mask[seen_mask])
            res_dict['map_m_seen'] = map_m_seen

        if (~seen_mask).any():
            map_m_unseen = vost_map(all_labels[~seen_mask], all_predictions[~seen_mask],
                                    all_original_mask[~seen_mask])
            res_dict['map_m_unseen'] = map_m_unseen

    loss = None
    res_df = None
    res_file = None

    res_min_max = {k: 'min' if k == 'ks' else 'max' for k in res_dict.keys()}

    if is_coficut:
        res_dict = {f'coficut_{k}': v for k, v in res_dict.items()}
        res_min_max = {f'coficut_{k}': v for k, v in res_min_max.items()}

    return res_dict, loss, res_df, res_min_max, res_file


def coficut_map(labels, predictions):
    map_m = average_precision_score(labels, predictions, average='macro')
    return map_m


def vost_map(pseudo_labels, predictions, original_mask):
    y = pseudo_labels[~original_mask]
    p = predictions[~original_mask]
    map_m = average_precision_score(y, p, average='macro')
    return map_m


def run(model, train_loader, test_loader, extra_val_loaders, optimiser, criterion, args, output_dict):

    log_writer = SummaryWriter(log_dir=output_dict['logs'])
    summary_output_path = output_dict['root'] / 'summary.csv'
    metrics_meters = {}

    for epoch in range(1, args.epochs + 1):
        print('=' * 120)
        print(f'EPOCH {epoch}')
        print('=' * 120)

        if not args.test_only:
            train_loss, train_metadata = train(args, train_loader, model, optimiser, criterion)
            train_metrics = {}  # simplify code and log only the loss during training
            log_run(epoch, log_writer, train_metrics, train_loss, 'Train')
        else:
            train_loss = None
            train_metrics = {}

        if epoch == 1 or epoch % args.test_frequency == 0:
            test_metrics, test_loss, res_df, metric_best_min_max, res_file = evaluate(model, test_loader,
                                                                                      extra_val_loaders)
            log_run(epoch, log_writer, test_metrics, test_loss, 'Test')
            e_res_path = output_dict['results'] / f'epoch_{epoch}.pth'
            torch.save(res_file, e_res_path)
        else:
            test_loss = None
            test_metrics = {}
            metric_best_min_max = {}
            res_file = {}

        summary_dict = dict(epoch=epoch)

        for set_, set_loss in zip(('train', 'test'), (train_loss, test_loss)):
            if isinstance(set_loss, dict):
                for k, v in set_loss.items():
                    if isinstance(v, torch.Tensor):
                        v = v.item()

                    summary_dict[f'{set_}_loss_{k}'] = v
            else:
                summary_dict[f'{set_}_loss'] = set_loss.item() if isinstance(set_loss, torch.Tensor) else set_loss

        for set_, metric_dict in zip(('train', 'test'), (train_metrics, test_metrics)):
            metric_dict_to_df_row(metric_dict, metrics_meters, set_, summary_dict, metric_best_min_max)

        summary_df = pd.DataFrame([summary_dict])

        if summary_output_path.exists():
            old_summary_df = pd.read_csv(summary_output_path)
            summary_df = pd.concat([old_summary_df, summary_df], ignore_index=True)

        summary_df.to_csv(summary_output_path, index=False)

        for metric, best_metric in metrics_meters.items():
            if best_metric is None or summary_dict.get(metric, None) is None:
                continue

            if is_best_metric(metric, summary_dict, best_metric, metric_best_min_max, 'test'):
                metrics_meters[metric] = summary_dict[metric]
                print(f'Best {metric} so far, dumping model state and results')
                best_state_path = output_dict['model_state'] / f'best_{metric}.pth'
                save_model_state(best_state_path, epoch, model, optimiser)
                best_res_path = output_dict['results'] / f'best_{metric}.pth'
                torch.save(res_file, best_res_path)

        last_state_path = output_dict['model_state'] / f'last.pth'
        save_model_state(last_state_path, epoch, model, optimiser)

        if args.test_only:
            break

    log_writer.close()


def save_model_state(best_state_path, epoch, model, optimiser):
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) \
        else model.state_dict()
    state = dict(model_state=state_dict, optimiser=optimiser.state_dict(), epoch=epoch)
    torch.save(state, best_state_path)


def metric_dict_to_df_row(metric_dict, metrics_meters, set_, summary_dict, min_max_metric, agg='sum'):
    for k, v in metric_dict.items():
        if v is None:
            continue

        k = k.replace('/', '_')
        kd = f'{set_}_{k}'

        if isinstance(v, torch.Tensor):
            if agg == 'sum':
                vv = v.sum().item()
            elif agg == 'mean':
                vv = v.mean().item()
            else:
                raise ValueError(f'Aggregation {agg} not implemented')
        else:
            vv = v

        summary_dict[kd] = vv

        if kd not in metrics_meters:
            metrics_meters[kd] = 0 if min_max_metric[k] == 'max' else torch.inf


def log_run(epoch, log_writer, metrics, loss, tag):
    if isinstance(loss, dict):
        for k, v in loss.items():
            if v is None:
                continue

            print(f'Average {tag} {k}: {v:0.4f}')
            log_writer.add_scalar(f'{tag}/{k}', v, epoch)
    else:
        if loss is not None:
            print(f'Average {tag} loss: {loss:0.4f}')
            log_writer.add_scalar(f'{tag}/loss', loss, epoch)

    for k, v in metrics.items():
        k = k.split('/')

        if len(k) == 1:
            k = k[0]
            k_tag = tag
        else:
            kt = k[:-1]
            k = k[-1]
            k_tag = tag + '_' + '_'.join(kt)

        v = v.mean() if isinstance(v, torch.Tensor) else v
        log_writer.add_scalar(f'{k_tag}/{k}', v, epoch)
        print(f'{k_tag} {k}: {v:0.4f}')


def setup_run_output(args):
    sub_paths = ('logs', 'results', 'model_state')
    run_tags = tuple(args.run_tags)

    if args.tag is not None:
        run_tags += ('tag',)

    run_id = ';'.join([f'{attr}={getattr(args, attr)}' for attr in run_tags])

    output_path = args.output_path / run_id
    count = 0

    while output_path.exists():
        output_path = output_path.parent / f'{run_id}.{count}'
        count += 1

    output_dict = {}

    for p in sub_paths:
        sp = output_path / p
        sp.mkdir(parents=True, exist_ok=True)
        output_dict[p] = sp

    output_dict['root'] = output_path

    return output_dict, run_id


def main():
    parser = create_parser()
    args = parser.parse_args()
    output_dict, run_id = setup_run_output(args)
    train_loader, test_loader, extra_val_loaders = setup_data(args)
    model = setup_model(args)
    criterion = setup_criterion(args)
    optimiser = setup_optimiser(model, args)
    run(model, train_loader, test_loader, extra_val_loaders, optimiser, criterion, args, output_dict)


if __name__ == '__main__':
    main()

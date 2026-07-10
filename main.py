from utils.data import *
from utils.metric import *
from argparse import ArgumentParser, ArgumentTypeError
import torch
import torch.nn as nn
import torch.utils.data as Data
from model.MSHNet import *
from model.loss import *
from model.full_dea_mshnet import FullDEAMSHNet
from model.full_dea_loss import (
    full_dea_aux_loss_v2,
    full_dea_aux_loss_v3,
    full_dea_aux_loss_v4,
)
from torch.optim import Adagrad
from tqdm import tqdm
import os.path as osp
import os
import time
import glob
import random
import numpy as np

PROJECT_DIR = osp.dirname(osp.abspath(__file__))
DEFAULT_DATASET_DIR = osp.join(PROJECT_DIR, 'datasets', 'IRSTD-1K')
DEFAULT_WEIGHT_DIR = osp.join(PROJECT_DIR, 'weight')

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('yes', 'true', 't', '1', 'y'):
        return True
    if value in ('no', 'false', 'f', '0', 'n'):
        return False
    raise ArgumentTypeError('Boolean value expected.')

def load_torch_file(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)

def get_dea_ramp(epoch, warm_epoch, ramp_epochs):
    if ramp_epochs <= 0:
        return 1.0
    if epoch <= warm_epoch:
        return 0.0
    return min(1.0, float(epoch - warm_epoch) / float(ramp_epochs))

def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def validate_args(args):
    if args.model_type == "full_dea":
        if args.if_checkpoint and args.init_from_baseline:
            raise ValueError("--if-checkpoint and --init-from-baseline are separate paths.")
        if not args.init_from_baseline and not args.if_checkpoint:
            print("warning: Full DEA is running without --init-from-baseline.")
        lite_lambdas = (
            args.dea_lambda_single,
            args.dea_lambda_dec,
            args.dea_lambda_empty,
        )
        if any(float(value) != 0.0 for value in lite_lambdas):
            raise ValueError(
                "Full DEA and DEA-lite losses must not be enabled together."
            )
        if args.full_dea_safe_kernel <= 0 or args.full_dea_safe_kernel % 2 == 0:
            raise ValueError("--full-dea-safe-kernel must be a positive odd integer.")
        for name in (
            "full_dea_topk_ratio",
            "full_dea_max_hard_bg_ratio",
        ):
            value = float(getattr(args, name))
            if value < 0.0 or value > 1.0:
                raise ValueError("--%s must be in [0, 1]." % name.replace("_", "-"))
        if float(args.full_dea_topk_min_score) < 0.0:
            raise ValueError("--full-dea-topk-min-score must be non-negative.")
        if args.full_dea_version not in ("v2", "v3", "v4", "v5"):
            raise ValueError("--full-dea-version must be v2, v3, v4, or v5.")
        if args.full_dea_protect_kernel <= 0 or args.full_dea_protect_kernel % 2 == 0:
            raise ValueError("--full-dea-protect-kernel must be a positive odd integer.")
        if args.full_dea_hard_min_area < 1:
            raise ValueError("--full-dea-hard-min-area must be >= 1.")
        if (
            args.full_dea_hard_max_area > 0
            and args.full_dea_hard_max_area < args.full_dea_hard_min_area
        ):
            raise ValueError(
                "--full-dea-hard-max-area must be 0 or >= --full-dea-hard-min-area."
            )
        if args.init_from_baseline and not osp.isfile(args.init_from_baseline):
            raise FileNotFoundError(args.init_from_baseline)
    return args

def get_method_name(args):
    if args.model_type == "full_dea":
        version = getattr(args, "full_dea_version", "v3")
        if version == "v2":
            return "FullDEA-v2"
        if version == "v4":
            return "FullDEA-v4-CRR"
        if version == "v5":
            return "FullDEA-v5-CRR-HT"
        return "FullDEA-v3-TPS"
    if (
        args.dea_lambda_single > 0
        or args.dea_lambda_dec > 0
        or args.dea_lambda_empty > 0
    ):
        return "DEA-lite"
    return "MSHNet"

def get_run_folder_name(args, timestamp=None):
    if timestamp is None:
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    safe_method = get_method_name(args).replace('/', '_')
    return '%s-%s' % (safe_method, timestamp)

def get_method_metadata(args):
    return {
        "method": get_method_name(args),
        "model_type": args.model_type,
        "full_dea_version": getattr(args, "full_dea_version", ""),
        "init_from_baseline": args.init_from_baseline,
        "full_dea_lambda": float(args.full_dea_lambda),
        "full_dea_ramp_epochs": int(args.full_dea_ramp_epochs),
        "full_dea_start_epoch": int(args.full_dea_start_epoch),
        "full_dea_freeze_backbone_epochs": int(args.full_dea_freeze_backbone_epochs),
        "full_dea_tau_base": float(args.full_dea_tau_base),
        "full_dea_tau_target": float(args.full_dea_tau_target),
        "full_dea_tau_scale": float(args.full_dea_tau_scale),
        "full_dea_topk_ratio": float(args.full_dea_topk_ratio),
        "full_dea_topk_min_score": float(args.full_dea_topk_min_score),
        "full_dea_max_hard_bg_ratio": float(args.full_dea_max_hard_bg_ratio),
        "full_dea_safe_kernel": int(args.full_dea_safe_kernel),
        "full_dea_protect_kernel": int(args.full_dea_protect_kernel),
        "full_dea_hard_min_area": int(args.full_dea_hard_min_area),
        "full_dea_hard_max_area": int(args.full_dea_hard_max_area),
        "dea_lambda_single": float(args.dea_lambda_single),
        "dea_lambda_dec": float(args.dea_lambda_dec),
        "dea_lambda_empty": float(args.dea_lambda_empty),
        "dataset_dir": args.dataset_dir,
        "seed": int(args.seed),
        "deterministic": bool(args.deterministic),
    }

def parse_args():

    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of model')

    parser.add_argument('--dataset-dir', type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--warm-epoch', type=int, default=5)

    parser.add_argument('--base-size', type=int, default=256)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--multi-gpus', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--gpu-ids', type=str, default='')
    parser.add_argument('--pin-memory', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--if-checkpoint', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--reset-optimizer', type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--weight-path', type=str, default=osp.join(DEFAULT_WEIGHT_DIR, 'IRSTD-1k_weight.tar'))
    parser.add_argument('--checkpoint-dir', type=str, default='')
    parser.add_argument(
        '--model-type',
        type=str,
        default='mshnet',
        choices=['mshnet', 'full_dea'],
    )
    parser.add_argument('--init-from-baseline', type=str, default='')
    parser.add_argument('--dea-lambda-single', type=float, default=0.0)
    parser.add_argument('--dea-lambda-dec', type=float, default=0.0)
    parser.add_argument('--dea-lambda-empty', type=float, default=0.0)
    parser.add_argument('--dea-tau', type=float, default=0.5)
    parser.add_argument('--dea-ramp-epochs', type=int, default=0)
    parser.add_argument('--save-dea-debug', action='store_true')
    parser.add_argument('--dea-debug-interval', type=int, default=50)
    parser.add_argument('--dea-debug-max-batches', type=int, default=1)
    parser.add_argument('--dea-detach-evidence', action='store_true')
    parser.add_argument('--seed', type=int, default=20260706)
    parser.add_argument('--deterministic', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--pd-fa-min-pd', type=float, default=0.93)
    parser.add_argument('--pd-fa-min-iou', type=float, default=0.655)
    parser.add_argument('--paired-baseline-iou', type=float, default=0.0)
    parser.add_argument('--pd-fa-iou-margin', type=float, default=0.005)
    parser.add_argument('--full-dea-lambda', type=float, default=1.0)
    parser.add_argument(
        '--full-dea-version',
        type=str,
        default='v3',
        choices=['v2', 'v3', 'v4', 'v5'],
    )
    parser.add_argument('--full-dea-ramp-epochs', type=int, default=30)
    parser.add_argument('--full-dea-start-epoch', type=int, default=0)
    parser.add_argument('--full-dea-freeze-backbone-epochs', type=int, default=0)
    parser.add_argument('--full-dea-tau-base', type=float, default=0.45)
    parser.add_argument('--full-dea-tau-target', type=float, default=0.45)
    parser.add_argument('--full-dea-tau-scale', type=float, default=0.45)
    parser.add_argument('--full-dea-topk-ratio', type=float, default=0.001)
    parser.add_argument('--full-dea-topk-min-score', type=float, default=0.45)
    parser.add_argument('--full-dea-max-hard-bg-ratio', type=float, default=0.003)
    parser.add_argument('--full-dea-safe-kernel', type=int, default=15)
    parser.add_argument('--full-dea-protect-kernel', type=int, default=9)
    parser.add_argument('--full-dea-hard-min-area', type=int, default=1)
    parser.add_argument('--full-dea-hard-max-area', type=int, default=256)
    parser.add_argument('--full-dea-debug', action='store_true')

    args = parser.parse_args()
    return validate_args(args)

class Trainer(object):
    def __init__(self, args):
        assert args.mode == 'train' or args.mode == 'test'

        self.args = args
        self.start_epoch = 0   
        self.mode = args.mode

        trainset = IRSTD_Dataset(args, mode='train')
        valset = IRSTD_Dataset(args, mode='val')

        data_generator = torch.Generator()
        data_generator.manual_seed(args.seed)

        loader_kwargs = {
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
            "persistent_workers": args.num_workers > 0,
            "worker_init_fn": seed_worker,
            "generator": data_generator,
        }
        if args.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2

        self.train_loader = Data.DataLoader(
            trainset,
            args.batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )
        self.val_loader = Data.DataLoader(
            valset,
            1,
            drop_last=False,
            **loader_kwargs,
        )

        device = torch.device('cuda')
        self.device = device
        torch.backends.cudnn.benchmark = not args.deterministic

        if args.model_type == "full_dea":
            model = FullDEAMSHNet(3, full_dea_version=args.full_dea_version)
        else:
            model = MSHNet(3)

        if args.multi_gpus and torch.cuda.device_count() > 1:
            device_ids = self.parse_gpu_ids(args.gpu_ids)
            print('use %d gpus: %s' % (len(device_ids), device_ids))
            model = nn.DataParallel(model, device_ids=device_ids)
        model.to(device)
        self.model = model

        if args.mode == 'train' and args.init_from_baseline and not args.if_checkpoint:
            baseline = load_torch_file(args.init_from_baseline)
            state_dict = self.extract_state_dict(baseline)
            self.load_model_state_partial(
                state_dict,
                allowed_missing_prefixes=("full_dea_head.",),
            )

        self.optimizer = Adagrad(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)

        self.down = nn.MaxPool2d(2, 2)
        self.loss_fun = SLSIoULoss()
        self.PD_FA = PD_FA(1, 10, args.base_size)
        self.mIoU = mIoU(1)
        self.ROC  = ROCMetric(1, 10)
        self.best_iou = 0.0
        self.best_pd_fa = float('inf')
        self.best_pd_fa_iou = 0.0
        self.best_pd_fa_pd = 0.0
        self.best_pd_fa_epoch = -1
        self.warm_epoch = args.warm_epoch

        if args.mode=='train':
            if args.if_checkpoint:
                check_folder = args.checkpoint_dir or self.find_latest_checkpoint_folder()
                checkpoint = load_torch_file(osp.join(check_folder, 'checkpoint.pkl'))
                self.load_model_state(checkpoint['net'])
                if args.reset_optimizer:
                    print('reset optimizer state')
                else:
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer'])
                    except (ValueError, RuntimeError) as exc:
                        print('skip optimizer state: %s' % exc)
                self.set_optimizer_lr(args.lr)
                self.start_epoch = checkpoint.get('epoch', -1) + 1
                self.best_iou = float(checkpoint.get('best_iou', checkpoint.get('iou', 0.0)))
                self.best_pd_fa = float(checkpoint.get('best_pd_fa', float('inf')))
                self.best_pd_fa_iou = float(checkpoint.get('best_pd_fa_iou', 0.0))
                self.best_pd_fa_pd = float(checkpoint.get('best_pd_fa_pd', 0.0))
                self.best_pd_fa_epoch = int(checkpoint.get('best_pd_fa_epoch', -1))
                self.save_folder = check_folder
            else:
                self.save_folder = osp.join(
                    DEFAULT_WEIGHT_DIR,
                    get_run_folder_name(args),
                )
                os.makedirs(self.save_folder, exist_ok=True)
        if args.mode=='test':
          
            weight = load_torch_file(args.weight_path)
            state_dict = self.extract_state_dict(weight)
            self.load_model_state(state_dict)
            '''
                # iou_67.87_weight
                weight = torch.load(args.weight_path)
                self.model.load_state_dict(weight)
            '''
            self.warm_epoch = -1

    def parse_gpu_ids(self, gpu_ids):
        if gpu_ids:
            device_ids = [int(item) for item in gpu_ids.split(',') if item.strip()]
        else:
            device_ids = list(range(torch.cuda.device_count()))
        if not device_ids:
            raise ValueError('No GPU ids selected.')
        return device_ids

    def extract_state_dict(self, weight_obj):
        if isinstance(weight_obj, dict):
            if 'state_dict' in weight_obj:
                return weight_obj['state_dict']
            if 'net' in weight_obj:
                return weight_obj['net']

            looks_like_state_dict = all(
                torch.is_tensor(value) for value in weight_obj.values()
            )
            if looks_like_state_dict:
                return weight_obj

        raise RuntimeError(
            'Unsupported weight format. Expected raw state_dict, '
            'dict with state_dict, or dict with net.'
        )

    def load_model_state(self, state_dict):
        try:
            self.model.load_state_dict(state_dict)
            return
        except RuntimeError:
            pass

        if isinstance(self.model, nn.DataParallel):
            try:
                self.model.module.load_state_dict(state_dict)
                return
            except RuntimeError:
                pass

        has_module_prefix = all(key.startswith('module.') for key in state_dict.keys())
        if has_module_prefix:
            state_dict = {key[len('module.'):]: value for key, value in state_dict.items()}
            target_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            target_model.load_state_dict(state_dict)
            return

        raise RuntimeError('Failed to load model state_dict.')

    def load_model_state_partial(self, state_dict, allowed_missing_prefixes=()):
        target_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if state_dict and all(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key[len('module.'):]: value for key, value in state_dict.items()}

        missing, unexpected = target_model.load_state_dict(state_dict, strict=False)
        bad_missing = [
            key
            for key in missing
            if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
        ]
        if bad_missing or unexpected:
            raise RuntimeError(
                'Partial baseline load failed. bad_missing=%s unexpected=%s'
                % (bad_missing, unexpected)
            )
        print(
            'loaded baseline with partial state: missing=%d unexpected=%d'
            % (len(missing), len(unexpected))
        )

    def set_optimizer_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print('set optimizer lr: %.6f' % lr)

    def find_latest_checkpoint_folder(self):
        run_glob = '%s-*' % get_method_name(self.args)
        checkpoint_paths = sorted(
            glob.glob(osp.join(DEFAULT_WEIGHT_DIR, run_glob, 'checkpoint.pkl')),
            key=osp.getmtime,
        )
        if not checkpoint_paths:
            raise FileNotFoundError(
                'No %s checkpoint found under %s. Pass --checkpoint-dir inside the project weight directory.'
                % (get_method_name(self.args), DEFAULT_WEIGHT_DIR)
            )
        return osp.dirname(checkpoint_paths[-1])
        
    def use_dea(self, epoch):
        return (
            self.args.model_type != "full_dea"
            and
            epoch > self.warm_epoch
            and (
                self.args.dea_lambda_single > 0
                or self.args.dea_lambda_dec > 0
                or self.args.dea_lambda_empty > 0
            )
        )

    def get_forward_tag(self, epoch):
        if self.args.model_type == "full_dea":
            return epoch >= self.args.full_dea_start_epoch
        return epoch > self.warm_epoch

    def get_full_dea_ramp(self, epoch):
        if epoch < self.args.full_dea_start_epoch:
            return 0.0
        return get_dea_ramp(
            epoch,
            self.args.full_dea_start_epoch - 1,
            self.args.full_dea_ramp_epochs,
        )

    def configure_full_dea_trainable(self, epoch):
        if self.args.model_type != "full_dea":
            return

        freeze = epoch < self.args.full_dea_freeze_backbone_epochs
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        for name, param in model.named_parameters():
            if freeze:
                param.requires_grad = name.startswith("full_dea_head")
            else:
                param.requires_grad = True

        if freeze:
            for name, module in model.named_modules():
                if name.startswith("full_dea_head"):
                    continue
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
            model.full_dea_head.train()

    def format_log_dict(self, log_dict):
        msg = []
        for key, value in log_dict.items():
            try:
                scalar = float(value.detach().mean()) if torch.is_tensor(value) else float(value)
                msg.append('%s=%.6f' % (key, scalar))
            except (TypeError, ValueError):
                pass
        return msg

    def save_dea_debug(self, epoch, iteration, data, labels, pred, dea_out):
        if not self.args.save_dea_debug:
            return
        if iteration >= self.args.dea_debug_max_batches:
            return
        if self.args.dea_debug_interval > 0 and epoch % self.args.dea_debug_interval != 0:
            return

        debug_root = self.save_folder if self.save_folder else PROJECT_DIR
        debug_dir = osp.join(debug_root, 'dea_debug')
        os.makedirs(debug_dir, exist_ok=True)

        sample = {
            "image": data[:1].detach().cpu(),
            "label": labels[:1].detach().cpu(),
            "z_full": pred[:1].detach().cpu(),
            "p_full": torch.sigmoid(pred[:1]).detach().cpu(),
            "scale_logits": dea_out["scale_logits"][:1].detach().cpu(),
            "z_only": dea_out["z_only"][:1].detach().cpu(),
            "p_only": torch.sigmoid(dea_out["z_only"][:1]).detach().cpu(),
            "z_only_max": dea_out["z_only_max"][:1].detach().cpu(),
            "p_only_max": torch.sigmoid(dea_out["z_only_max"][:1]).detach().cpu(),
            "z_empty": dea_out["z_empty"][:1].detach().cpu(),
            "p_empty": torch.sigmoid(dea_out["z_empty"][:1]).detach().cpu(),
            "d_logit": dea_out["decidability_logit"][:1].detach().cpu(),
            "d_prob": torch.sigmoid(dea_out["decidability_logit"][:1]).detach().cpu(),
        }

        torch.save(sample, osp.join(debug_dir, 'epoch_%04d_iter_%04d.pt' % (epoch, iteration)))

    def train(self, epoch):
        self.model.train()
        self.configure_full_dea_trainable(epoch)
        tbar = tqdm(self.train_loader)
        losses = AverageMeter()
        for i, (data, mask) in enumerate(tbar):
  
            data = data.to(self.device, non_blocking=True)
            labels = mask.to(self.device, non_blocking=True)

            tag = self.get_forward_tag(epoch)
            use_dea = self.use_dea(epoch)

            full_dea_out = None
            if self.args.model_type == "full_dea":
                out = self.model(data, tag, return_dict=True)
                masks = out["masks"]
                pred = out["pred"]
                full_dea_out = out["full_dea"]
                dea_out = None
            elif use_dea:
                masks, pred, dea_out = self.model(
                    data,
                    tag,
                    return_dea=True,
                    dea_detach_evidence=self.args.dea_detach_evidence,
                )
            else:
                masks, pred = self.model(data, tag)
                dea_out = None

            loss = 0

            loss = loss + self.loss_fun(pred, labels, self.warm_epoch, epoch)
            labels_for_scale = labels
            for j in range(len(masks)):
                if j>0:
                    labels_for_scale = self.down(labels_for_scale)
                loss = loss + self.loss_fun(masks[j], labels_for_scale, self.warm_epoch, epoch)
                
            loss = loss / (len(masks)+1)
            loss_seg_for_debug = loss.detach()

            if self.args.model_type == "full_dea" and full_dea_out is not None:
                ramp = self.get_full_dea_ramp(epoch)
                if self.args.full_dea_version == "v2":
                    loss_full_dea, full_dea_log = full_dea_aux_loss_v2(
                        full_dea_out=full_dea_out,
                        target=labels,
                        epoch=epoch,
                        warm_epoch=self.warm_epoch,
                        seg_criterion=self.loss_fun,
                        tau_base=self.args.full_dea_tau_base,
                        tau_target=self.args.full_dea_tau_target,
                        tau_scale=self.args.full_dea_tau_scale,
                        safe_kernel=self.args.full_dea_safe_kernel,
                        topk_ratio=self.args.full_dea_topk_ratio,
                        topk_min_score=self.args.full_dea_topk_min_score,
                        max_hard_bg_ratio=self.args.full_dea_max_hard_bg_ratio,
                    )
                elif self.args.full_dea_version == "v3":
                    loss_full_dea, full_dea_log = full_dea_aux_loss_v3(
                        full_dea_out=full_dea_out,
                        target=labels,
                        epoch=epoch,
                        warm_epoch=self.warm_epoch,
                        seg_criterion=self.loss_fun,
                        tau_base=self.args.full_dea_tau_base,
                        tau_target=self.args.full_dea_tau_target,
                        tau_scale=self.args.full_dea_tau_scale,
                        protect_kernel=self.args.full_dea_protect_kernel,
                        safe_kernel=self.args.full_dea_safe_kernel,
                        min_component_area=self.args.full_dea_hard_min_area,
                        max_component_area=self.args.full_dea_hard_max_area,
                        max_hard_bg_ratio=self.args.full_dea_max_hard_bg_ratio,
                    )
                else:
                    loss_full_dea, full_dea_log = full_dea_aux_loss_v4(
                        full_dea_out=full_dea_out,
                        target=labels,
                        epoch=epoch,
                        warm_epoch=self.warm_epoch,
                        seg_criterion=self.loss_fun,
                        tau_base=self.args.full_dea_tau_base,
                        tau_target=self.args.full_dea_tau_target,
                        tau_scale=self.args.full_dea_tau_scale,
                        protect_kernel=self.args.full_dea_protect_kernel,
                        safe_kernel=self.args.full_dea_safe_kernel,
                        min_component_area=self.args.full_dea_hard_min_area,
                        max_component_area=self.args.full_dea_hard_max_area,
                        max_hard_bg_ratio=self.args.full_dea_max_hard_bg_ratio,
                    )
                loss = loss + self.args.full_dea_lambda * ramp * loss_full_dea

                if self.args.full_dea_debug and i % max(1, self.args.dea_debug_interval) == 0:
                    msg = [
                        'full_dea_ramp=%.6f' % ramp,
                        'full_dea_loss=%.6f' % float(loss_full_dea.detach()),
                        'full_dea_weighted=%.6f'
                        % float((self.args.full_dea_lambda * ramp * loss_full_dea).detach()),
                    ]
                    msg.extend(self.format_log_dict(full_dea_log))
                    print('[FULL DEA DEBUG] ' + ' | '.join(msg))
            elif use_dea:
                ramp = get_dea_ramp(epoch, self.warm_epoch, self.args.dea_ramp_epochs)
                cur_lambda_single = self.args.dea_lambda_single * ramp
                cur_lambda_dec = self.args.dea_lambda_dec * ramp
                cur_lambda_empty = self.args.dea_lambda_empty * ramp

                loss_dea, dea_log = dea_lite_loss(
                    dea_out=dea_out,
                    z_full=pred,
                    gt=labels,
                    lambda_single=cur_lambda_single,
                    lambda_dec=cur_lambda_dec,
                    lambda_empty=cur_lambda_empty,
                    tau=self.args.dea_tau,
                )
                loss = loss + loss_dea
                self.save_dea_debug(epoch, i, data, labels, pred, dea_out)

                if self.args.save_dea_debug and self.args.dea_debug_interval > 0 and i % self.args.dea_debug_interval == 0:
                    dea_ratio = (loss_dea.detach() / (loss_seg_for_debug + 1e-6)).item()
                    msg = [
                        'dea_ratio=%.4f' % dea_ratio,
                        'lambda_single=%.6f' % cur_lambda_single,
                        'lambda_empty=%.6f' % cur_lambda_empty,
                        'lambda_dec=%.6f' % cur_lambda_dec,
                    ]
                    msg.extend(self.format_log_dict(dea_log))
                    print('[DEA DEBUG] ' + ' | '.join(msg))
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
       
            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, loss %.4f' % (epoch, losses.avg))
    
    def test(self, epoch):
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        self.ROC.reset()
        tbar = tqdm(self.val_loader)
        tag = False
        with torch.no_grad():
            for i, (data, mask) in enumerate(tbar):
    
                data = data.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)

                if self.args.model_type == "full_dea":
                    tag = True
                elif epoch>self.warm_epoch:
                    tag = True

                loss = 0
                if self.args.model_type == "full_dea":
                    out = self.model(data, tag, return_dict=True)
                    pred = out["pred"]
                else:
                    _, pred = self.model(data, tag)
                # loss += self.loss_fun(pred, mask,self.warm_epoch, epoch)

                self.mIoU.update(pred, mask)
                self.PD_FA.update(pred, mask)
                self.ROC.update(pred, mask)
                _, mean_IoU = self.mIoU.get()

                tbar.set_description('Epoch %d, IoU %.4f' % (epoch, mean_IoU))
            FA, PD = self.PD_FA.get(len(self.val_loader))
            _, mean_IoU = self.mIoU.get()
            ture_positive_rate, false_positive_rate, _, _ = self.ROC.get()

            
            if self.mode == 'train':
                current_pd = PD[0]
                current_fa = FA[0] * 1000000
                if self.args.paired_baseline_iou > 0:
                    pd_fa_iou_threshold = max(
                        self.args.pd_fa_min_iou,
                        self.args.paired_baseline_iou - self.args.pd_fa_iou_margin,
                    )
                else:
                    pd_fa_iou_threshold = self.args.pd_fa_min_iou

                is_pd_fa_candidate = (
                    current_pd >= self.args.pd_fa_min_pd
                    and mean_IoU >= pd_fa_iou_threshold
                    and current_fa < self.best_pd_fa
                )
                metric_line = '{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n'.format(
                    time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())),
                    epoch,
                    mean_IoU,
                    current_pd,
                    current_fa,
                )
                print(metric_line.strip())
                with open(osp.join(self.save_folder, 'epoch_metric.log'), 'a') as f:
                    f.write(metric_line)

                if mean_IoU > self.best_iou:
                    self.best_iou = mean_IoU
                
                    torch.save(
                        self.model.state_dict(),
                        osp.join(self.save_folder, 'weight.pkl'),
                    )

                    best_iou_states = {
                        "net": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "iou": mean_IoU,
                        "pd": current_pd,
                        "fa": current_fa,
                        "best_iou": self.best_iou,
                        "best_pd_fa": self.best_pd_fa,
                        "best_pd_fa_iou": self.best_pd_fa_iou,
                        "best_pd_fa_pd": self.best_pd_fa_pd,
                        "best_pd_fa_epoch": self.best_pd_fa_epoch,
                        "method_meta": get_method_metadata(self.args),
                    }
                    torch.save(
                        best_iou_states,
                        osp.join(self.save_folder, 'checkpoint_best_iou.pkl'),
                    )

                    with open(osp.join(self.save_folder, 'metric.log'), 'a') as f:
                        f.write('{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n' .
                            format(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())), 
                                epoch, self.best_iou, current_pd, current_fa))

                if is_pd_fa_candidate:
                    self.best_pd_fa = current_fa
                    self.best_pd_fa_iou = mean_IoU
                    self.best_pd_fa_pd = current_pd
                    self.best_pd_fa_epoch = epoch

                    torch.save(
                        self.model.state_dict(),
                        osp.join(self.save_folder, 'weight_pd_fa_best.pkl'),
                    )

                    pd_fa_states = {
                        "net": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "iou": mean_IoU,
                        "pd": current_pd,
                        "fa": current_fa,
                        "best_iou": self.best_iou,
                        "best_pd_fa": self.best_pd_fa,
                        "best_pd_fa_iou": self.best_pd_fa_iou,
                        "best_pd_fa_pd": self.best_pd_fa_pd,
                        "best_pd_fa_epoch": self.best_pd_fa_epoch,
                        "method_meta": get_method_metadata(self.args),
                    }
                    torch.save(
                        pd_fa_states,
                        osp.join(self.save_folder, 'checkpoint_pd_fa_best.pkl'),
                    )

                    with open(osp.join(self.save_folder, 'metric_pd_fa_best.log'), 'a') as f:
                        f.write('{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n' .
                            format(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())), 
                                epoch, mean_IoU, current_pd, current_fa))
                        
                latest_states = {
                    "net": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "iou": mean_IoU,
                    "pd": current_pd,
                    "fa": current_fa,
                    "best_iou": self.best_iou,
                    "best_pd_fa": self.best_pd_fa,
                    "best_pd_fa_iou": self.best_pd_fa_iou,
                    "best_pd_fa_pd": self.best_pd_fa_pd,
                    "best_pd_fa_epoch": self.best_pd_fa_epoch,
                    "method_meta": get_method_metadata(self.args),
                }
                torch.save(latest_states, osp.join(self.save_folder, 'checkpoint.pkl'))
            elif self.mode == 'test':
                print('mIoU: '+str(mean_IoU)+'\n')
                print('Pd: '+str(PD[0])+'\n')
                print('Fa: '+str(FA[0]*1000000)+'\n')


         
if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed, args.deterministic)

    trainer = Trainer(args)
    
    if trainer.mode=='train':
        for epoch in range(trainer.start_epoch, args.epochs):
            trainer.train(epoch)
            trainer.test(epoch)
    else:
        trainer.test(1)
 

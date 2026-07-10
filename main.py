from utils.data import *
from utils.metric import *
from argparse import ArgumentParser, ArgumentTypeError
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from model.MSHNet import *
from model.loss import *
from model.full_dea_mshnet import FullDEAMSHNet
from model.dea_integrated_mshnet import DEAIntegratedMSHNet
from model.dea_mshnet import DEAMSHNet
from model.dea_integrated_loss import residual_aligned_route_loss
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
import json

PROJECT_DIR = osp.dirname(osp.abspath(__file__))
DEFAULT_DATASET_DIR = osp.join(PROJECT_DIR, 'datasets', 'IRSTD-1K')
DEFAULT_WEIGHT_DIR = osp.join(PROJECT_DIR, 'weight')
DEA_MODEL_TYPES = ('dea', 'predictive_correction')


def is_dea_main_model(model_type):
    return model_type in DEA_MODEL_TYPES

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
    if getattr(args, "mode", "train") not in ("train", "test"):
        raise ValueError("--mode must be train or test.")

    if args.model_type == "dea_integrated":
        if args.if_checkpoint and args.init_from_baseline:
            raise ValueError("--if-checkpoint and --init-from-baseline are separate paths.")
        if not args.init_from_baseline and not args.if_checkpoint and getattr(args, "mode", "train") == "train":
            print("warning: Integrated DEA is running without --init-from-baseline.")
        lite_lambdas = (
            args.dea_lambda_single,
            args.dea_lambda_dec,
            args.dea_lambda_empty,
        )
        if any(float(value) != 0.0 for value in lite_lambdas):
            raise ValueError(
                "Integrated DEA and DEA-lite losses must not be enabled together."
            )
        if int(args.integrated_route_channels) < 1:
            raise ValueError("--integrated-route-channels must be >= 1.")
        if float(args.integrated_route_temperature) <= 0.0:
            raise ValueError("--integrated-route-temperature must be > 0.")
        if float(args.integrated_update_limit) <= 0.0:
            raise ValueError("--integrated-update-limit must be > 0.")
        if float(args.integrated_route_loss_weight) < 0.0:
            raise ValueError("--integrated-route-loss-weight must be non-negative.")
        if int(args.integrated_route_ramp_epochs) < 0:
            raise ValueError("--integrated-route-ramp-epochs must be non-negative.")
        if (
            args.integrated_routing_mode == "attention"
            and float(args.integrated_route_loss_weight) > 0.0
        ):
            raise ValueError(
                "Residual action supervision is not defined for the attention "
                "control; set --integrated-route-loss-weight 0."
            )
        if float(args.integrated_uncertain_margin) <= 0.1:
            raise ValueError(
                "--integrated-uncertain-margin must be > 0.1 to guarantee "
                "the initial all-uncertain route."
            )
        if (
            args.integrated_routing_mode == "dea"
            and
            args.integrated_scale_routing
            and args.integrated_route_upsample_mode not in ("nearest", "nearest-exact")
        ):
            raise ValueError(
                "Hard scale routing requires nearest/nearest-exact upsampling; "
                "continuous interpolation destroys target/clutter exclusivity."
            )
        if args.init_from_baseline and not osp.isfile(args.init_from_baseline):
            raise FileNotFoundError(args.init_from_baseline)

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

    if is_dea_main_model(args.model_type):
        if int(args.predictive_state_channels) < 4:
            raise ValueError("--predictive-state-channels must be >= 4.")
        if not 0.0 < float(args.predictive_step_size) <= 1.0:
            raise ValueError("--predictive-step-size must be in (0, 1].")
        if float(args.predictive_delta_min) <= 0.0:
            raise ValueError("--predictive-delta-min must be > 0.")
        if float(args.predictive_delta_init) <= float(args.predictive_delta_min):
            raise ValueError(
                "--predictive-delta-init must be greater than "
                "--predictive-delta-min."
            )
        if int(args.predictive_log_interval) < 0:
            raise ValueError("--predictive-log-interval must be non-negative.")
        if args.init_from_baseline and not osp.isfile(args.init_from_baseline):
            raise FileNotFoundError(args.init_from_baseline)
        lite_lambdas = (
            args.dea_lambda_single,
            args.dea_lambda_dec,
            args.dea_lambda_empty,
        )
        if any(float(value) != 0.0 for value in lite_lambdas):
            raise ValueError(
                "DEA main-model training and DEA-lite losses must not be mixed."
            )

    if getattr(args, "mode", "train") == "train" and not getattr(args, "val_split_file", ""):
        val_fraction = float(getattr(args, "val_fraction", 0.2))
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("--val-fraction must be strictly between 0 and 1.")
    return args

def get_method_name(args):
    if is_dea_main_model(args.model_type):
        eta = ("%g" % float(
            getattr(args, "predictive_step_size", 1.0)
        )).replace("-", "m").replace(".", "p")
        prefix = "DEA-v0" if args.model_type == "dea" else "PredictiveCorrection"
        name = "%s-C%d-Eta%s" % (
            prefix,
            int(getattr(args, "predictive_state_channels", 32)),
            eta,
        )
        if bool(getattr(args, "predictive_legacy_numerics", False)):
            name += "-LegacyNum"
        return name
    if args.model_type == "dea_integrated":
        routing_mode = getattr(args, "integrated_routing_mode", "dea")
        decoder_routing = bool(getattr(args, "integrated_decoder_routing", True))
        scale_routing = bool(getattr(args, "integrated_scale_routing", True))
        residual_aligned = (
            float(getattr(args, "integrated_route_loss_weight", 0.0)) > 0.0
        )
        suffix = "-ResidualAligned" if residual_aligned else "-UnsupervisedRoute"
        if routing_mode == "soft_tri":
            return "DEAIntegrated-SoftTri" + suffix
        if routing_mode == "attention":
            return "DEAIntegrated-Attention"
        if decoder_routing and scale_routing:
            return "DEAIntegrated" + suffix
        if decoder_routing:
            return "DEAIntegrated-DecoderOnly" + suffix
        if scale_routing:
            return "DEAIntegrated-ScaleOnly" + suffix
        return "DEAIntegrated-Identity" + suffix
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
    if getattr(args, "init_from_baseline", ""):
        return "MSHNet-Continued"
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
        "init_from_baseline": getattr(
            args, "origin_baseline_checkpoint", args.init_from_baseline
        ),
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
        "integrated_route_channels": int(getattr(args, "integrated_route_channels", 16)),
        "integrated_route_temperature": float(getattr(args, "integrated_route_temperature", 1.0)),
        "integrated_routing_mode": getattr(args, "integrated_routing_mode", ""),
        "integrated_decoder_routing": bool(getattr(args, "integrated_decoder_routing", False)),
        "integrated_scale_routing": bool(getattr(args, "integrated_scale_routing", False)),
        "integrated_route_upsample_mode": getattr(args, "integrated_route_upsample_mode", ""),
        "integrated_update_limit": float(getattr(args, "integrated_update_limit", 0.25)),
        "integrated_uncertain_margin": float(getattr(args, "integrated_uncertain_margin", 1.0)),
        "integrated_route_loss_weight": float(getattr(args, "integrated_route_loss_weight", 0.0)),
        "integrated_route_ramp_epochs": int(getattr(args, "integrated_route_ramp_epochs", 0)),
        "integrated_isolate_route_gradients": bool(
            getattr(args, "integrated_isolate_route_gradients", True)
        ),
        "predictive_state_channels": int(
            getattr(args, "predictive_state_channels", 32)
        ),
        "predictive_step_size": float(
            getattr(args, "predictive_step_size", 1.0)
        ),
        "predictive_delta_init": float(
            getattr(args, "predictive_delta_init", 1.0)
        ),
        "predictive_delta_min": float(
            getattr(args, "predictive_delta_min", 0.05)
        ),
        "predictive_legacy_numerics": bool(
            getattr(args, "predictive_legacy_numerics", False)
        ),
        "dea_version": "v0_adjoint_predictive_correction",
        "dea_state_channels": int(
            getattr(args, "predictive_state_channels", 32)
        ),
        "dea_step_size": float(
            getattr(args, "predictive_step_size", 1.0)
        ),
        "dea_delta_init": float(
            getattr(args, "predictive_delta_init", 1.0)
        ),
        "dea_delta_min": float(
            getattr(args, "predictive_delta_min", 0.05)
        ),
        "dea_legacy_numerics": bool(
            getattr(args, "predictive_legacy_numerics", False)
        ),
        "dataset_dir": args.dataset_dir,
        "train_split_file": getattr(args, "train_split_file", ""),
        "val_split_file": getattr(args, "val_split_file", ""),
        "test_split_file": getattr(args, "test_split_file", ""),
        "val_fraction": float(getattr(args, "val_fraction", 0.2)),
        "split_seed": int(getattr(args, "split_seed", getattr(args, "seed", 0))),
        "train_split_sha256": getattr(args, "train_split_sha256", ""),
        "val_split_sha256": getattr(args, "val_split_sha256", ""),
        "test_split_sha256": getattr(args, "test_split_sha256", ""),
        "seed": int(args.seed),
        "deterministic": bool(args.deterministic),
    }

def parse_args():

    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of model')

    parser.add_argument('--dataset-dir', type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument('--train-split-file', type=str, default='')
    parser.add_argument('--val-split-file', type=str, default='')
    parser.add_argument('--test-split-file', type=str, default='')
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--split-seed', type=int, default=20260706)
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

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--weight-path', type=str, default=osp.join(DEFAULT_WEIGHT_DIR, 'IRSTD-1k_weight.tar'))
    parser.add_argument('--checkpoint-dir', type=str, default='')
    parser.add_argument(
        '--model-type',
        type=str,
        default='mshnet',
        choices=['mshnet', 'dea', 'full_dea', 'dea_integrated', 'predictive_correction'],
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
    parser.add_argument('--integrated-route-channels', type=int, default=16)
    parser.add_argument('--integrated-route-temperature', type=float, default=1.0)
    parser.add_argument(
        '--integrated-routing-mode',
        type=str,
        default='dea',
        choices=['dea', 'soft_tri', 'attention'],
    )
    parser.add_argument(
        '--integrated-decoder-routing',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
    )
    parser.add_argument(
        '--integrated-scale-routing',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
    )
    parser.add_argument(
        '--integrated-route-upsample-mode',
        type=str,
        default='nearest-exact',
        choices=['nearest', 'nearest-exact', 'bilinear', 'bicubic'],
    )
    parser.add_argument('--integrated-update-limit', type=float, default=0.25)
    parser.add_argument('--integrated-uncertain-margin', type=float, default=1.0)
    # Experimental identifiability control.  It is disabled by default because
    # the first real-data smoke drove every hard route to keep/abstain; it must
    # not be presented as a validated formal objective without the prescribed
    # controls in the experiment protocol.
    parser.add_argument('--integrated-route-loss-weight', type=float, default=0.0)
    parser.add_argument('--integrated-route-ramp-epochs', type=int, default=3)
    parser.add_argument(
        '--dea-state-channels', '--predictive-state-channels',
        dest='predictive_state_channels', type=int, default=32,
    )
    parser.add_argument(
        '--dea-step-size', '--predictive-step-size',
        dest='predictive_step_size', type=float, default=1.0,
    )
    parser.add_argument(
        '--dea-delta-init', '--predictive-delta-init',
        dest='predictive_delta_init', type=float, default=1.0,
    )
    parser.add_argument(
        '--dea-delta-min', '--predictive-delta-min',
        dest='predictive_delta_min', type=float, default=0.05,
    )
    parser.add_argument(
        '--dea-legacy-numerics', '--predictive-legacy-numerics',
        dest='predictive_legacy_numerics',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
    )
    parser.add_argument('--predictive-log-interval', type=int, default=50)
    parser.add_argument(
        '--integrated-isolate-route-gradients',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
    )
    parser.add_argument('--integrated-log-interval', type=int, default=50)

    args = parser.parse_args()
    return validate_args(args)

class Trainer(object):
    def __init__(self, args):
        assert args.mode == 'train' or args.mode == 'test'

        self.args = args
        setattr(args, 'origin_baseline_checkpoint', args.init_from_baseline)
        self.start_epoch = 0   
        self.mode = args.mode

        self.train_dataset = None
        if args.mode == 'train':
            trainset = IRSTD_Dataset(args, mode='train')
            valset = IRSTD_Dataset(args, mode='val')
            test_reference = IRSTD_Dataset(args, mode='test')
            self.assert_disjoint_splits(trainset, valset, test_reference)
            self.train_dataset = trainset
            setattr(args, 'train_split_sha256', trainset.split_sha256)
            setattr(args, 'val_split_sha256', valset.split_sha256)
            setattr(args, 'test_split_sha256', test_reference.split_sha256)
        else:
            valset = IRSTD_Dataset(args, mode='test')
            setattr(args, 'test_split_sha256', valset.split_sha256)
        self.val_dataset = valset

        def loader_kwargs(generator_seed):
            data_generator = torch.Generator()
            data_generator.manual_seed(generator_seed)
            kwargs = {
                "num_workers": args.num_workers,
                "pin_memory": args.pin_memory,
                "persistent_workers": args.num_workers > 0,
                "worker_init_fn": seed_worker,
                "generator": data_generator,
            }
            if args.num_workers > 0:
                kwargs["prefetch_factor"] = 2
            return kwargs

        self.train_loader = None
        if self.train_dataset is not None:
            self.train_loader = Data.DataLoader(
                self.train_dataset,
                args.batch_size,
                shuffle=True,
                drop_last=True,
                **loader_kwargs(args.seed),
            )
        self.val_loader = Data.DataLoader(
            valset,
            1,
            drop_last=False,
            **loader_kwargs(args.seed + 1),
        )
        self.print_split_summary()

        device = torch.device('cuda')
        self.device = device
        torch.backends.cudnn.benchmark = not args.deterministic

        if args.model_type == "full_dea":
            model = FullDEAMSHNet(3, full_dea_version=args.full_dea_version)
        elif args.model_type == "dea_integrated":
            model = DEAIntegratedMSHNet(
                3,
                route_channels=args.integrated_route_channels,
                route_temperature=args.integrated_route_temperature,
                routing_mode=args.integrated_routing_mode,
                decoder_routing=args.integrated_decoder_routing,
                scale_routing=args.integrated_scale_routing,
                route_upsample_mode=args.integrated_route_upsample_mode,
                update_limit=args.integrated_update_limit,
                uncertain_margin=args.integrated_uncertain_margin,
                isolate_route_gradients=args.integrated_isolate_route_gradients,
            )
        elif is_dea_main_model(args.model_type):
            model = DEAMSHNet(
                3,
                state_channels=args.predictive_state_channels,
                step_size=args.predictive_step_size,
                delta_init=args.predictive_delta_init,
                delta_min=args.predictive_delta_min,
                legacy_influence_numerics=args.predictive_legacy_numerics,
            )
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
            if args.model_type == "dea_integrated":
                allowed_missing = DEAIntegratedMSHNet.BASELINE_MISSING_PREFIXES
                allowed_unexpected = DEAIntegratedMSHNet.BASELINE_UNEXPECTED_PREFIXES
            elif is_dea_main_model(args.model_type):
                allowed_missing = DEAMSHNet.BASELINE_MISSING_PREFIXES
                allowed_unexpected = DEAMSHNet.BASELINE_UNEXPECTED_PREFIXES
            elif args.model_type == "full_dea":
                allowed_missing = ("full_dea_head.", "decidability_head.")
                allowed_unexpected = ()
            else:
                # The pristine public MSHNet checkpoint predates the optional
                # DEA-lite head present in this repository.
                allowed_missing = ("decidability_head.",)
                allowed_unexpected = ()
            self.load_model_state_partial(
                state_dict,
                allowed_missing_prefixes=allowed_missing,
                allowed_unexpected_prefixes=allowed_unexpected,
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
                self.validate_integrated_checkpoint_metadata(
                    checkpoint,
                    check_split_hashes=True,
                )
                if args.model_type == 'dea_integrated':
                    setattr(
                        args,
                        'origin_baseline_checkpoint',
                        checkpoint.get('method_meta', {}).get(
                            'init_from_baseline', ''
                        ),
                    )
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
            self.persist_split_manifests()
        if args.mode=='test':
          
            weight = load_torch_file(args.weight_path)
            self.validate_integrated_checkpoint_metadata(
                weight,
                check_split_hashes=False,
            )
            state_dict = self.extract_state_dict(weight)
            self.load_model_state(state_dict)
            '''
                # iou_67.87_weight
                weight = torch.load(args.weight_path)
                self.model.load_state_dict(weight)
            '''
            self.warm_epoch = -1

    @staticmethod
    def assert_disjoint_splits(trainset, valset, testset):
        named_sets = {
            'train': set(trainset.names),
            'val': set(valset.names),
            'test': set(testset.names),
        }
        for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            overlap = sorted(named_sets[left].intersection(named_sets[right]))
            if overlap:
                raise RuntimeError(
                    '%s/%s split leakage detected (%d samples), e.g. %s'
                    % (left, right, len(overlap), overlap[:5])
                )

    def print_split_summary(self):
        if self.train_dataset is not None:
            print(
                'split train: n=%d sha256=%s source=%s'
                % (
                    len(self.train_dataset),
                    self.train_dataset.split_sha256[:12],
                    self.train_dataset.split_source,
                )
            )
            print(
                'split val:   n=%d sha256=%s source=%s'
                % (
                    len(self.val_dataset),
                    self.val_dataset.split_sha256[:12],
                    self.val_dataset.split_source,
                )
            )
        else:
            print(
                'split test:  n=%d sha256=%s source=%s'
                % (
                    len(self.val_dataset),
                    self.val_dataset.split_sha256[:12],
                    self.val_dataset.split_source,
                )
            )

    def persist_split_manifests(self):
        if self.mode != 'train':
            return
        for split_name, dataset in (
            ('train', self.train_dataset),
            ('val', self.val_dataset),
        ):
            manifest_path = osp.join(self.save_folder, 'split_%s.txt' % split_name)
            with open(manifest_path, 'w') as f:
                for name in dataset.names:
                    f.write(name + '\n')

    def validate_integrated_checkpoint_metadata(
        self,
        checkpoint,
        check_split_hashes,
    ):
        if self.args.model_type != 'dea_integrated' and not is_dea_main_model(
            self.args.model_type
        ):
            return
        if not isinstance(checkpoint, dict) or 'method_meta' not in checkpoint:
            raise RuntimeError(
                '%s resume/test requires a checkpoint containing '
                'method_meta; use checkpoint.pkl/checkpoint_best_iou.pkl rather '
                'than a raw weight.pkl file.'
                % self.args.model_type
            )

        metadata = checkpoint['method_meta']
        expected = get_method_metadata(self.args)
        if self.args.model_type == 'dea_integrated':
            semantic_keys = (
                'model_type',
                'integrated_route_channels',
                'integrated_route_temperature',
                'integrated_routing_mode',
                'integrated_decoder_routing',
                'integrated_scale_routing',
                'integrated_route_upsample_mode',
                'integrated_update_limit',
                'integrated_uncertain_margin',
                'integrated_route_loss_weight',
                'integrated_route_ramp_epochs',
                'integrated_isolate_route_gradients',
                'test_split_sha256',
            )
        elif self.args.model_type == 'predictive_correction':
            semantic_keys = (
                'model_type',
                'predictive_state_channels',
                'predictive_step_size',
                'predictive_delta_init',
                'predictive_delta_min',
                'predictive_legacy_numerics',
                'test_split_sha256',
            )
        else:
            semantic_keys = (
                'model_type',
                'dea_version',
                'dea_state_channels',
                'dea_step_size',
                'dea_delta_init',
                'dea_delta_min',
                'dea_legacy_numerics',
                'test_split_sha256',
            )
        if check_split_hashes:
            semantic_keys = semantic_keys + (
                'train_split_sha256',
                'val_split_sha256',
            )

        mismatches = []
        for key in semantic_keys:
            if key not in metadata:
                mismatches.append('%s=<missing>' % key)
            elif metadata[key] != expected[key]:
                mismatches.append(
                    '%s checkpoint=%r cli=%r'
                    % (key, metadata[key], expected[key])
                )
        if mismatches:
            raise RuntimeError(
                '%s checkpoint semantics mismatch: %s'
                % (self.args.model_type, '; '.join(mismatches))
            )

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

    def load_model_state_partial(
        self,
        state_dict,
        allowed_missing_prefixes=(),
        allowed_unexpected_prefixes=(),
    ):
        target_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if state_dict and all(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key[len('module.'):]: value for key, value in state_dict.items()}

        missing, unexpected = target_model.load_state_dict(state_dict, strict=False)
        bad_missing = [
            key
            for key in missing
            if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
        ]
        bad_unexpected = [
            key
            for key in unexpected
            if not any(key.startswith(prefix) for prefix in allowed_unexpected_prefixes)
        ]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                'Partial baseline load failed. bad_missing=%s bad_unexpected=%s'
                % (bad_missing, bad_unexpected)
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
            self.args.model_type == "mshnet"
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
        if self.args.model_type == "dea_integrated":
            # Integrated DEA is a continued-training method over a complete
            # MSHNet checkpoint; terminal scale routing must be exercised from
            # the first optimization step.
            return True
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

    @staticmethod
    def new_integrated_route_audit():
        return {
            "route_counts": [torch.zeros(3, dtype=torch.long) for _ in range(4)],
            "gt_route_counts": [torch.zeros(3, dtype=torch.long) for _ in range(4)],
            "bg_route_counts": [torch.zeros(3, dtype=torch.long) for _ in range(4)],
            "entropy_sum": [0.0] * 4,
            "entropy_count": [0] * 4,
            "transitions": [torch.zeros((3, 3), dtype=torch.long) for _ in range(3)],
            "delta_abs_sum": [0.0] * 4,
            "delta_count": [0] * 4,
            "target_prob_fn_sum": [0.0] * 4,
            "target_prob_fn_count": [0] * 4,
            "clutter_prob_fp_sum": [0.0] * 4,
            "clutter_prob_fp_count": [0] * 4,
            "keep_prob_correct_sum": [0.0] * 4,
            "keep_prob_correct_count": [0] * 4,
            "hard_action_condition_hits": [torch.zeros(3, dtype=torch.long) for _ in range(4)],
            "hard_action_condition_totals": [torch.zeros(3, dtype=torch.long) for _ in range(4)],
            "delta_residual_aligned": [0] * 4,
            "delta_residual_active": [0] * 4,
        }

    @staticmethod
    def update_integrated_route_audit(audit, output, target):
        routes = output["routes"]
        preclosure_probability = torch.sigmoid(
            output["scale_fusion"]["z_base"].detach()
        )
        binary_target_full = target > 0.5
        false_negative = binary_target_full & (preclosure_probability < 0.5)
        false_positive = (~binary_target_full) & (preclosure_probability >= 0.5)
        correct = ~(false_negative | false_positive)
        for scale, route in enumerate(routes):
            winner = route["winner"].detach()
            probabilities = route["probabilities"].detach().clamp_min(1e-12)
            entropy = -(probabilities * probabilities.log()).sum(dim=1)
            audit["route_counts"][scale] += torch.bincount(
                winner.reshape(-1).cpu(), minlength=3
            )
            audit["entropy_sum"][scale] += float(entropy.sum().item())
            audit["entropy_count"][scale] += int(entropy.numel())

            target_at_scale = F.adaptive_max_pool2d(
                target.float(), output_size=winner.shape[-2:]
            )[:, 0] > 0.5
            gt_winner = winner[target_at_scale]
            bg_winner = winner[~target_at_scale]
            if gt_winner.numel():
                audit["gt_route_counts"][scale] += torch.bincount(
                    gt_winner.reshape(-1).cpu(), minlength=3
                )
            if bg_winner.numel():
                audit["bg_route_counts"][scale] += torch.bincount(
                    bg_winner.reshape(-1).cpu(), minlength=3
                )

            probabilities_full = F.interpolate(
                probabilities,
                size=target.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
            winner_full = F.interpolate(
                winner.unsqueeze(1).float(),
                size=target.shape[-2:],
                mode='nearest',
            ).long()
            conditions = (
                (false_negative, 0, "target_prob_fn"),
                (false_positive, 1, "clutter_prob_fp"),
                (correct, 2, "keep_prob_correct"),
            )
            for condition, action, prefix in conditions:
                count = int(condition.sum().item())
                audit[prefix + "_count"][scale] += count
                if count:
                    audit[prefix + "_sum"][scale] += float(
                        probabilities_full[:, action:action + 1][condition].sum().item()
                    )
                    audit["hard_action_condition_hits"][scale][action] += int(
                        (winner_full[condition] == action).sum().item()
                    )
                audit["hard_action_condition_totals"][scale][action] += count

        # routes are fine-to-coarse.  Each matrix row is the coarse state and
        # each column is the corresponding next-finer state.
        for fine_scale in range(3):
            fine = routes[fine_scale]["winner"].detach()
            coarse = routes[fine_scale + 1]["winner"].detach()
            coarse_up = F.interpolate(
                coarse.unsqueeze(1).float(),
                size=fine.shape[-2:],
                mode='nearest',
            ).squeeze(1).long()
            pair_index = (coarse_up * 3 + fine).reshape(-1).cpu()
            audit["transitions"][fine_scale] += torch.bincount(
                pair_index, minlength=9
            ).reshape(3, 3)

        deltas = output["scale_fusion"]["deltas"].detach()
        for scale in range(4):
            scale_delta = deltas[:, scale]
            audit["delta_abs_sum"][scale] += float(scale_delta.abs().sum().item())
            audit["delta_count"][scale] += int(scale_delta.numel())
            desired_direction = (
                binary_target_full.to(scale_delta.dtype)
                - preclosure_probability
            )[:, 0]
            active = scale_delta.abs() > 0
            audit["delta_residual_active"][scale] += int(active.sum().item())
            audit["delta_residual_aligned"][scale] += int(
                ((scale_delta * desired_direction > 0) & active).sum().item()
            )

    def finalize_integrated_route_audit(self, audit, epoch):
        def normalized(counts):
            total = int(counts.sum().item())
            if total == 0:
                return [0.0, 0.0, 0.0]
            return [float(value) / float(total) for value in counts.tolist()]

        report = {
            "epoch": int(epoch),
            "mode": self.mode,
            "evaluation_split_sha256": self.val_dataset.split_sha256,
            "state_order": [
                "increase/target",
                "decrease/clutter",
                "keep/abstain",
            ],
            "route_occupancy": [normalized(item) for item in audit["route_counts"]],
            "gt_route_occupancy": [normalized(item) for item in audit["gt_route_counts"]],
            "bg_route_occupancy": [normalized(item) for item in audit["bg_route_counts"]],
            "route_entropy": [
                audit["entropy_sum"][scale]
                / max(1, audit["entropy_count"][scale])
                for scale in range(4)
            ],
            "coarse_to_fine_transitions": [
                matrix.tolist() for matrix in audit["transitions"]
            ],
            "scale_delta_abs_mean": [
                audit["delta_abs_sum"][scale]
                / max(1, audit["delta_count"][scale])
                for scale in range(4)
            ],
            "target_probability_on_false_negative": [
                audit["target_prob_fn_sum"][scale]
                / max(1, audit["target_prob_fn_count"][scale])
                for scale in range(4)
            ],
            "clutter_probability_on_false_positive": [
                audit["clutter_prob_fp_sum"][scale]
                / max(1, audit["clutter_prob_fp_count"][scale])
                for scale in range(4)
            ],
            "keep_probability_on_correct": [
                audit["keep_prob_correct_sum"][scale]
                / max(1, audit["keep_prob_correct_count"][scale])
                for scale in range(4)
            ],
            "hard_action_condition_accuracy": [
                [
                    float(audit["hard_action_condition_hits"][scale][action])
                    / max(
                        1,
                        int(audit["hard_action_condition_totals"][scale][action]),
                    )
                    for action in range(3)
                ]
                for scale in range(4)
            ],
            "delta_residual_sign_alignment": [
                float(audit["delta_residual_aligned"][scale])
                / max(1, audit["delta_residual_active"][scale])
                for scale in range(4)
            ],
        }
        serialized = json.dumps(report, sort_keys=True)
        print('[INTEGRATED DEA VAL] ' + serialized)
        if self.mode == 'train':
            with open(osp.join(self.save_folder, 'route_metric.jsonl'), 'a') as f:
                f.write(serialized + '\n')

        if all(item[2] > 0.999999 for item in report["route_occupancy"]):
            print(
                'warning: Integrated DEA remains all keep/abstain on the entire '
                'evaluation split; the current forward mapping is still the baseline.'
            )
        return report

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

    def dea_main_loss(self, state_logits, labels, epoch):
        """Match MSHNet's effective resolution weights without duplicate heads.

        MSHNet averages final, full-resolution side, half, quarter, and eighth
        losses.  The first two together give the full-resolution prediction a
        weight of 0.4.  The predictive decoder has only one shared readout, so
        it applies 0.4 once to the final state and 0.2 to the preceding three
        states.  The coarsest 1/16 state is kept as an unsupervised prefix.
        """
        if len(state_logits) != 5:
            raise ValueError(
                "predictive decoder must return five coarse-to-fine logits"
            )
        final_loss = self.loss_fun(
            state_logits[4], labels, self.warm_epoch, epoch
        )
        if epoch <= self.warm_epoch:
            return final_loss

        labels_half = self.down(labels)
        labels_quarter = self.down(labels_half)
        labels_eighth = self.down(labels_quarter)
        loss_half = self.loss_fun(
            state_logits[3], labels_half, self.warm_epoch, epoch
        )
        loss_quarter = self.loss_fun(
            state_logits[2], labels_quarter, self.warm_epoch, epoch
        )
        loss_eighth = self.loss_fun(
            state_logits[1], labels_eighth, self.warm_epoch, epoch
        )
        return (
            0.4 * final_loss
            + 0.2 * loss_half
            + 0.2 * loss_quarter
            + 0.2 * loss_eighth
        )

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
            dea_main_out = None
            if self.args.model_type == "full_dea":
                out = self.model(data, tag, return_dict=True)
                masks = out["masks"]
                pred = out["pred"]
                full_dea_out = out["full_dea"]
                dea_out = None
            elif self.args.model_type == "dea_integrated":
                out = self.model(data, tag, return_dict=True)
                masks = out["masks"]
                pred = out["pred"]
                dea_out = None
            elif is_dea_main_model(self.args.model_type):
                return_details = (
                    self.args.predictive_log_interval > 0
                    and i % self.args.predictive_log_interval == 0
                )
                dea_main_out = self.model(
                    data,
                    tag,
                    return_dict=True,
                    return_details=return_details,
                )
                masks = []
                pred = dea_main_out["pred"]
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

            if dea_main_out is not None:
                loss = self.dea_main_loss(
                    dea_main_out["state_logits"], labels, epoch
                )
            else:
                loss = self.loss_fun(pred, labels, self.warm_epoch, epoch)
                labels_for_scale = labels
                for j in range(len(masks)):
                    if j>0:
                        labels_for_scale = self.down(labels_for_scale)
                    loss = loss + self.loss_fun(
                        masks[j], labels_for_scale, self.warm_epoch, epoch
                    )
                loss = loss / (len(masks)+1)
            loss_seg_for_debug = loss.detach()
            integrated_route_loss = None
            integrated_route_log = {}
            integrated_route_ramp = 0.0
            if (
                self.args.model_type == "dea_integrated"
                and self.args.integrated_route_loss_weight > 0.0
            ):
                integrated_route_loss, integrated_route_log = (
                    residual_aligned_route_loss(out, labels)
                )
                if self.args.integrated_route_ramp_epochs > 0:
                    integrated_route_ramp = min(
                        1.0,
                        float(epoch + 1)
                        / float(self.args.integrated_route_ramp_epochs),
                    )
                else:
                    integrated_route_ramp = 1.0
                loss = loss + (
                    self.args.integrated_route_loss_weight
                    * integrated_route_ramp
                    * integrated_route_loss
                )

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

            if (
                self.args.model_type == "dea_integrated"
                and self.args.integrated_log_interval > 0
                and i % self.args.integrated_log_interval == 0
            ):
                core_model = (
                    self.model.module
                    if isinstance(self.model, nn.DataParallel)
                    else self.model
                )
                route_stats = core_model.route_statistics(out["routes"])
                msg = self.format_log_dict(route_stats)
                if "scale_fusion" in out:
                    deltas = out["scale_fusion"]["deltas"]
                    msg.append('scale_delta_abs=%.6f' % float(deltas.detach().abs().mean()))
                if integrated_route_loss is not None:
                    weighted_route_loss = (
                        self.args.integrated_route_loss_weight
                        * integrated_route_ramp
                        * integrated_route_loss.detach()
                    )
                    msg.extend([
                        'route_ramp=%.4f' % integrated_route_ramp,
                        'route_loss_raw=%.6f' % float(integrated_route_loss.detach()),
                        'route_loss_weighted=%.6f' % float(weighted_route_loss),
                        'route_to_seg=%.6f'
                        % float(weighted_route_loss / (loss_seg_for_debug + 1e-6)),
                    ])
                    msg.extend(self.format_log_dict(integrated_route_log))
                print('[INTEGRATED DEA] ' + ' | '.join(msg))

            if (
                dea_main_out is not None
                and "corrections" in dea_main_out
            ):
                core_model = (
                    self.model.module
                    if isinstance(self.model, nn.DataParallel)
                    else self.model
                )
                stats = core_model.state_statistics(dea_main_out)
                print('[DEA MAIN] ' + ' | '.join(
                    self.format_log_dict(stats)
                ))
        
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
        # A saved model must always be evaluated through its complete
        # multi-scale inference graph.  Tying this to a training epoch silently
        # bypassed final fusion in the old test entry point.
        tag = True
        route_audit = (
            self.new_integrated_route_audit()
            if self.args.model_type == "dea_integrated"
            else None
        )
        with torch.no_grad():
            for i, (data, mask) in enumerate(tbar):
    
                data = data.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)

                loss = 0
                if self.args.model_type in (
                    "full_dea",
                    "dea_integrated",
                ) or is_dea_main_model(self.args.model_type):
                    out = self.model(data, tag, return_dict=True)
                    pred = out["pred"]
                    if route_audit is not None:
                        self.update_integrated_route_audit(route_audit, out, mask)
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
            if route_audit is not None:
                self.finalize_integrated_route_audit(route_audit, epoch)

            
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
 

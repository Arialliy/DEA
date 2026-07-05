from utils.data import *
from utils.metric import *
from argparse import ArgumentParser, ArgumentTypeError
import torch
import torch.utils.data as Data
from model.MSHNet import *
from model.loss import *
from torch.optim import Adagrad
from tqdm import tqdm
import os.path as osp
import os
import time
import glob

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
    parser.add_argument('--dea-lambda-single', type=float, default=0.10)
    parser.add_argument('--dea-lambda-dec', type=float, default=0.05)
    parser.add_argument('--dea-lambda-empty', type=float, default=0.01)
    parser.add_argument('--dea-tau', type=float, default=0.3)
    parser.add_argument('--dea-ramp-epochs', type=int, default=20)
    parser.add_argument('--save-dea-debug', action='store_true')
    parser.add_argument('--dea-debug-interval', type=int, default=20)
    parser.add_argument('--dea-debug-max-batches', type=int, default=1)

    args = parser.parse_args()
    return args

class Trainer(object):
    def __init__(self, args):
        assert args.mode == 'train' or args.mode == 'test'

        self.args = args
        self.start_epoch = 0   
        self.mode = args.mode

        trainset = IRSTD_Dataset(args, mode='train')
        valset = IRSTD_Dataset(args, mode='val')

        loader_kwargs = {
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
            "persistent_workers": args.num_workers > 0,
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
        torch.backends.cudnn.benchmark = False

        model = MSHNet(3)

        if args.multi_gpus and torch.cuda.device_count() > 1:
            device_ids = self.parse_gpu_ids(args.gpu_ids)
            print('use %d gpus: %s' % (len(device_ids), device_ids))
            model = nn.DataParallel(model, device_ids=device_ids)
        model.to(device)
        self.model = model

        self.optimizer = Adagrad(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)

        self.down = nn.MaxPool2d(2, 2)
        self.loss_fun = SLSIoULoss()
        self.PD_FA = PD_FA(1, 10, args.base_size)
        self.mIoU = mIoU(1)
        self.ROC  = ROCMetric(1, 10)
        self.best_iou = 0
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
                self.start_epoch = checkpoint['epoch']+1
                self.best_iou = checkpoint['iou']
                self.save_folder = check_folder
            else:
                self.save_folder = osp.join(
                    DEFAULT_WEIGHT_DIR,
                    'MSHNet-%s' % (time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))),
                )
                os.makedirs(self.save_folder, exist_ok=True)
        if args.mode=='test':
          
            weight = load_torch_file(args.weight_path)
            self.load_model_state(weight['state_dict'])
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

    def set_optimizer_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print('set optimizer lr: %.6f' % lr)

    def find_latest_checkpoint_folder(self):
        checkpoint_paths = sorted(
            glob.glob(osp.join(DEFAULT_WEIGHT_DIR, 'MSHNet-*', 'checkpoint.pkl')),
            key=osp.getmtime,
        )
        if not checkpoint_paths:
            raise FileNotFoundError(
                'No checkpoint found under %s. Pass --checkpoint-dir inside the project weight directory.' % DEFAULT_WEIGHT_DIR
            )
        return osp.dirname(checkpoint_paths[-1])
        
    def get_dea_loss_weights(self, epoch):
        if self.args.dea_ramp_epochs <= 0:
            ratio = 1.0
        else:
            ratio = (epoch - self.warm_epoch) / float(self.args.dea_ramp_epochs)
            ratio = min(1.0, max(0.0, ratio))

        return {
            "lambda_single": self.args.dea_lambda_single * ratio,
            "lambda_dec": self.args.dea_lambda_dec * ratio,
            "lambda_empty": self.args.dea_lambda_empty * ratio,
        }

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
        tbar = tqdm(self.train_loader)
        losses = AverageMeter()
        for i, (data, mask) in enumerate(tbar):
  
            data = data.to(self.device, non_blocking=True)
            labels = mask.to(self.device, non_blocking=True)

            tag = epoch > self.warm_epoch

            if tag:
                masks, pred, dea_out = self.model(data, tag, return_dea=True)
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

            if tag:
                dea_weights = self.get_dea_loss_weights(epoch)
                loss_dea, _ = dea_lite_loss(
                    dea_out,
                    pred,
                    labels,
                    lambda_single=dea_weights["lambda_single"],
                    lambda_dec=dea_weights["lambda_dec"],
                    lambda_empty=dea_weights["lambda_empty"],
                    tau=self.args.dea_tau,
                )
                loss = loss + loss_dea
                self.save_dea_debug(epoch, i, data, labels, pred, dea_out)
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
       
            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, loss %.4f' % (epoch, losses.avg))
    
    def test(self, epoch):
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        tbar = tqdm(self.val_loader)
        tag = False
        with torch.no_grad():
            for i, (data, mask) in enumerate(tbar):
    
                data = data.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)

                if epoch>self.warm_epoch:
                    tag = True

                loss = 0
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
                
                    torch.save(self.model.state_dict(), osp.join(self.save_folder, 'weight.pkl'))
                    with open(osp.join(self.save_folder, 'metric.log'), 'a') as f:
                        f.write('{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n' .
                            format(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())), 
                                epoch, self.best_iou, current_pd, current_fa))
                        
                all_states = {"net":self.model.state_dict(), "optimizer":self.optimizer.state_dict(), "epoch": epoch, "iou":self.best_iou}
                torch.save(all_states, osp.join(self.save_folder, 'checkpoint.pkl'))
            elif self.mode == 'test':
                print('mIoU: '+str(mean_IoU)+'\n')
                print('Pd: '+str(PD[0])+'\n')
                print('Fa: '+str(FA[0]*1000000)+'\n')


         
if __name__ == '__main__':
    args = parse_args()

    trainer = Trainer(args)
    
    if trainer.mode=='train':
        for epoch in range(trainer.start_epoch, args.epochs):
            trainer.train(epoch)
            trainer.test(epoch)
    else:
        trainer.test(1)
 

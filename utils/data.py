import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms

import os
from PIL import Image, ImageOps, ImageFilter
import os.path as osp
import sys
import random
import shutil
import glob
import hashlib
from skimage import measure


class IRSTD_Dataset(Data.Dataset):
    def __init__(self, args, mode='train'):
        if mode not in ('train', 'val', 'test'):
            raise ValueError("Unknown dataset mode: %s" % mode)

        dataset_dir = args.dataset_dir

        if mode in ('train', 'val'):
            txtfile = 'trainval.txt'
            split_prefix = 'train'
        else:
            txtfile = 'test.txt'
            split_prefix = 'test'

        split_override = (
            getattr(args, 'train_split_file', '')
            if mode in ('train', 'val')
            else getattr(args, 'test_split_file', '')
        )
        self.list_dir = self._resolve_split_file(
            dataset_dir, txtfile, split_prefix, split_override
        )
        self.imgs_dir = osp.join(dataset_dir, 'images')
        self.label_dir = osp.join(dataset_dir, 'masks')

        source_names = self._read_names(self.list_dir)
        self.split_source = self.list_dir
        if mode in ('train', 'val'):
            train_names, val_names = self._split_train_validation(
                source_names,
                mode=mode,
                val_fraction=float(getattr(args, 'val_fraction', 0.2)),
                split_seed=int(getattr(args, 'split_seed', getattr(args, 'seed', 0))),
                explicit_val_file=getattr(args, 'val_split_file', ''),
                dataset_dir=dataset_dir,
            )
            self.names = train_names if mode == 'train' else val_names
        else:
            self.names = source_names

        self.mode = mode
        self.return_instance_labels = bool(
            mode == 'train'
            and getattr(args, 'return_instance_labels', False)
        )
        self.crop_size = args.crop_size
        self.base_size = args.base_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        self.split_sha256 = hashlib.sha256(
            ('\n'.join(self.names) + '\n').encode('utf-8')
        ).hexdigest()

    @staticmethod
    def _read_names(path):
        with open(path, 'r') as f:
            names = [line.strip() for line in f if line.strip()]
        if not names:
            raise ValueError('Empty split file: %s' % path)
        if len(names) != len(set(names)):
            raise ValueError('Duplicate sample names in split file: %s' % path)
        return names

    def _split_train_validation(
        self,
        source_names,
        mode,
        val_fraction,
        split_seed,
        explicit_val_file,
        dataset_dir,
    ):
        if len(source_names) < 2:
            raise ValueError('At least two training samples are required for a holdout split.')

        if explicit_val_file:
            val_path = explicit_val_file
            if not osp.isabs(val_path):
                val_path = osp.join(dataset_dir, val_path)
            val_names = self._read_names(val_path)
            self.split_source = '%s + val=%s' % (self.list_dir, val_path)
            # With explicit manifests, the train file is already the fit set;
            # Trainer performs the fail-closed overlap audit across all three
            # manifests.  This also supports an official validation set that
            # is not a subset of the source training list.
            return source_names, val_names

        if not 0.0 < val_fraction < 1.0:
            raise ValueError('val_fraction must be strictly between 0 and 1.')

        ranked_names = sorted(
            source_names,
            key=lambda name: hashlib.sha256(
                ('%d\0%s' % (split_seed, name)).encode('utf-8')
            ).digest(),
        )
        num_val = max(1, min(len(source_names) - 1, int(round(len(source_names) * val_fraction))))
        val_set = set(ranked_names[:num_val])
        # Preserve the source-file order in both subsets so evaluation is
        # stable and the train loader remains the only shuffled component.
        train_names = [name for name in source_names if name not in val_set]
        val_names = [name for name in source_names if name in val_set]
        return train_names, val_names

    def _resolve_split_file(self, dataset_dir, txtfile, split_prefix, split_override=''):
        if split_override:
            override_path = split_override
            if not osp.isabs(override_path):
                override_path = osp.join(dataset_dir, override_path)
            if not osp.isfile(override_path):
                raise FileNotFoundError(override_path)
            return override_path

        candidates = [osp.join(dataset_dir, txtfile)]
        dataset_name = osp.basename(osp.normpath(dataset_dir))
        candidates.append(osp.join(dataset_dir, 'img_idx', '%s_%s.txt' % (split_prefix, dataset_name)))
        candidates.extend(sorted(glob.glob(osp.join(dataset_dir, 'img_idx', '%s_*.txt' % split_prefix))))

        for candidate in candidates:
            if osp.exists(candidate):
                return candidate

        raise FileNotFoundError(
            'Could not find split file. Tried: %s' % ', '.join(candidates)
        )

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name+'.png')
        label_path = osp.join(self.label_dir, name+'.png')

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode in ('val', 'test'):
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unknown self.mode")

        
        img, mask = self.transform(img), transforms.ToTensor()(mask)
        if self.return_instance_labels:
            instance_labels = measure.label(
                mask[0].numpy() > 0.5,
                connectivity=2,
            )
            instance_labels = torch.from_numpy(instance_labels).long().unsqueeze(0)
            return img, mask, instance_labels
        return img, mask

    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img, mask


    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        return img, mask

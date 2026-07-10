from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest

from main import Trainer
from utils.data import IRSTD_Dataset


def make_args(dataset_dir, **kwargs):
    args = Namespace(
        dataset_dir=str(dataset_dir),
        crop_size=16,
        base_size=16,
        seed=17,
        split_seed=23,
        val_fraction=0.25,
        train_split_file='',
        val_split_file='',
        test_split_file='',
    )
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


def write_split(path, names):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(names) + '\n', encoding='utf-8')


def test_implicit_holdout_is_deterministic_disjoint_and_never_uses_test(tmp_path):
    train_names = ['train_%02d' % index for index in range(20)]
    test_names = ['test_%02d' % index for index in range(7)]
    write_split(tmp_path / 'img_idx' / ('train_%s.txt' % tmp_path.name), train_names)
    write_split(tmp_path / 'img_idx' / ('test_%s.txt' % tmp_path.name), test_names)
    args = make_args(tmp_path)

    train_a = IRSTD_Dataset(args, mode='train')
    train_b = IRSTD_Dataset(args, mode='train')
    val = IRSTD_Dataset(args, mode='val')
    test = IRSTD_Dataset(args, mode='test')

    assert train_a.names == train_b.names
    assert set(train_a.names).isdisjoint(val.names)
    assert set(train_a.names).union(val.names) == set(train_names)
    assert test.names == test_names
    assert set(val.names).isdisjoint(test.names)
    assert len(val) == 5
    assert train_a.split_sha256 == train_b.split_sha256


def test_explicit_fit_val_test_manifests_are_supported(tmp_path):
    fit = ['fit_a', 'fit_b']
    val = ['val_a']
    test = ['test_a']
    write_split(tmp_path / 'fit.txt', fit)
    write_split(tmp_path / 'val.txt', val)
    write_split(tmp_path / 'test.txt', test)
    args = make_args(
        tmp_path,
        train_split_file='fit.txt',
        val_split_file='val.txt',
        test_split_file='test.txt',
    )

    trainset = IRSTD_Dataset(args, mode='train')
    valset = IRSTD_Dataset(args, mode='val')
    testset = IRSTD_Dataset(args, mode='test')

    assert trainset.names == fit
    assert valset.names == val
    assert testset.names == test
    Trainer.assert_disjoint_splits(trainset, valset, testset)


def test_overlap_audit_fails_closed():
    trainset = SimpleNamespace(names=['a', 'b'])
    valset = SimpleNamespace(names=['b', 'c'])
    testset = SimpleNamespace(names=['d'])
    with pytest.raises(RuntimeError, match='train/val split leakage'):
        Trainer.assert_disjoint_splits(trainset, valset, testset)


def test_duplicate_manifest_names_are_rejected(tmp_path):
    write_split(tmp_path / 'train.txt', ['a', 'a'])
    write_split(tmp_path / 'test.txt', ['t'])
    args = make_args(
        tmp_path,
        train_split_file='train.txt',
        test_split_file='test.txt',
    )
    with pytest.raises(ValueError, match='Duplicate sample names'):
        IRSTD_Dataset(args, mode='train')

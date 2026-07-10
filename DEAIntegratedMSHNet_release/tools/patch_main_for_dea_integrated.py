#!/usr/bin/env python3
"""Idempotently add the ``dea_integrated`` CLI entry to the current DEA main.py.

Run from the repository root:

    python /path/to/patch_main_for_dea_integrated.py --main main.py

A ``main.py.pre_dea_integrated`` backup is created before the first write.  The
script fails loudly when the expected current-repository anchors are absent;
it never silently emits a partially patched training entry.
"""

import argparse
import os
import re
import shutil


IMPORT_LINE = "from model.dea_integrated_mshnet import DEAIntegratedMSHNet"


def replace_once(text, old, new, description):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            "%s: expected exactly one anchor, found %d" % (description, count)
        )
    return text.replace(old, new, 1)


def regex_replace_once(text, pattern, replacement, description, flags=0):
    text_new, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(
            "%s: expected exactly one regex match, found %d" % (description, count)
        )
    return text_new


def patch(text):
    if IMPORT_LINE in text:
        required_markers = [
            "choices=['mshnet', 'full_dea', 'dea_integrated']",
            'elif args.model_type == "dea_integrated":',
            "allowed_unexpected_prefixes",
        ]
        if (
            "--integrated-dea-routing-mode" not in text
            and "--integrated-routing-mode" not in text
        ):
            required_markers.append("integrated routing CLI")
        missing_markers = [marker for marker in required_markers if marker not in text]
        if missing_markers:
            raise RuntimeError(
                "main.py contains the Integrated DEA import but is only partially patched; "
                "missing markers=%s" % missing_markers
            )
        return text, False

    text = replace_once(
        text,
        "from model.full_dea_mshnet import FullDEAMSHNet\n",
        "from model.full_dea_mshnet import FullDEAMSHNet\n"
        + IMPORT_LINE + "\n",
        "model import",
    )

    text = replace_once(
        text,
        "choices=['mshnet', 'full_dea'],",
        "choices=['mshnet', 'full_dea', 'dea_integrated'],",
        "--model-type choices",
    )

    argument_anchor = "    parser.add_argument('--init-from-baseline', type=str, default='')\n"
    integrated_arguments = argument_anchor + """
    # Formal Integrated DEA and its parameter-matched ablations.
    parser.add_argument('--integrated-dea-route-channels', type=int, default=16)
    parser.add_argument('--integrated-dea-temperature', type=float, default=1.0)
    parser.add_argument(
        '--integrated-dea-routing-mode',
        type=str,
        default='dea',
        choices=['dea', 'soft_tri', 'attention'],
    )
    parser.add_argument(
        '--integrated-dea-decoder-routing',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
    )
    parser.add_argument(
        '--integrated-dea-scale-routing',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
    )
    parser.add_argument('--integrated-dea-update-limit', type=float, default=0.25)
    parser.add_argument('--integrated-dea-uncertain-margin', type=float, default=1.0)
    parser.add_argument(
        '--integrated-dea-route-upsample-mode',
        type=str,
        default='nearest-exact',
        choices=['nearest', 'nearest-exact', 'bilinear', 'bicubic'],
    )
"""
    text = replace_once(
        text,
        argument_anchor,
        integrated_arguments,
        "integrated DEA arguments",
    )

    validation_anchor = "def validate_args(args):\n"
    validation_block = validation_anchor + """
    if args.model_type == "dea_integrated":
        if args.if_checkpoint and args.init_from_baseline:
            raise ValueError(
                "--if-checkpoint and --init-from-baseline are separate paths."
            )
        if not args.init_from_baseline and not args.if_checkpoint:
            print(
                "warning: Integrated DEA is running without "
                "--init-from-baseline; paired comparison will not be valid."
            )
        lite_lambdas = (
            args.dea_lambda_single,
            args.dea_lambda_dec,
            args.dea_lambda_empty,
        )
        if any(float(value) != 0.0 for value in lite_lambdas):
            raise ValueError(
                "Integrated DEA must not be combined with DEA-lite losses."
            )
        if args.integrated_dea_route_channels < 1:
            raise ValueError("--integrated-dea-route-channels must be >= 1.")
        if args.integrated_dea_temperature <= 0:
            raise ValueError("--integrated-dea-temperature must be > 0.")
        if args.integrated_dea_update_limit <= 0:
            raise ValueError("--integrated-dea-update-limit must be > 0.")
        if args.integrated_dea_uncertain_margin <= 0.1:
            raise ValueError(
                "--integrated-dea-uncertain-margin must be > 0.1."
            )
        if (
            args.integrated_dea_routing_mode == "dea"
            and args.integrated_dea_scale_routing
            and args.integrated_dea_route_upsample_mode
            not in ("nearest", "nearest-exact")
        ):
            raise ValueError(
                "hard DEA scale routing requires nearest/nearest-exact upsampling."
            )

"""
    text = replace_once(
        text,
        validation_anchor,
        validation_block,
        "validate_args",
    )

    method_anchor = "def get_method_name(args):\n"
    method_block = method_anchor + """
    if args.model_type == "dea_integrated":
        mode = getattr(args, "integrated_dea_routing_mode", "dea")
        decoder_on = bool(getattr(args, "integrated_dea_decoder_routing", True))
        scale_on = bool(getattr(args, "integrated_dea_scale_routing", True))
        if mode == "soft_tri":
            return "DEAIntegrated-NoUncertainIdentity"
        if mode == "attention":
            return "DEAIntegrated-Attention"
        if not decoder_on and scale_on:
            return "DEAIntegrated-ScaleOnly"
        if decoder_on and not scale_on:
            return "DEAIntegrated-DecoderOnly"
        if not decoder_on and not scale_on:
            return "DEAIntegrated-IdentityControl"
        return "DEAIntegrated"

"""
    text = replace_once(
        text,
        method_anchor,
        method_block,
        "get_method_name",
    )

    metadata_anchor = '        "model_type": args.model_type,\n'
    metadata_block = metadata_anchor + """        "integrated_dea_route_channels": int(
            getattr(args, "integrated_dea_route_channels", 16)
        ),
        "integrated_dea_temperature": float(
            getattr(args, "integrated_dea_temperature", 1.0)
        ),
        "integrated_dea_routing_mode": getattr(
            args, "integrated_dea_routing_mode", "dea"
        ),
        "integrated_dea_decoder_routing": bool(
            getattr(args, "integrated_dea_decoder_routing", True)
        ),
        "integrated_dea_scale_routing": bool(
            getattr(args, "integrated_dea_scale_routing", True)
        ),
"""
    text = replace_once(
        text,
        metadata_anchor,
        metadata_block,
        "method metadata",
    )

    model_pattern = re.compile(
        r'        if args\.model_type == "full_dea":\n'
        r'            model = FullDEAMSHNet\(3, full_dea_version=args\.full_dea_version\)\n'
        r'        else:\n'
        r'            model = MSHNet\(3\)\n'
    )
    model_replacement = """        if args.model_type == "full_dea":
            model = FullDEAMSHNet(3, full_dea_version=args.full_dea_version)
        elif args.model_type == "dea_integrated":
            model = DEAIntegratedMSHNet(
                3,
                route_channels=args.integrated_dea_route_channels,
                route_temperature=args.integrated_dea_temperature,
                routing_mode=args.integrated_dea_routing_mode,
                decoder_routing=args.integrated_dea_decoder_routing,
                scale_routing=args.integrated_dea_scale_routing,
                route_upsample_mode=args.integrated_dea_route_upsample_mode,
                update_limit=args.integrated_dea_update_limit,
                uncertain_margin=args.integrated_dea_uncertain_margin,
            )
        else:
            model = MSHNet(3)
"""
    text, count = model_pattern.subn(model_replacement, text, count=1)
    if count != 1:
        raise RuntimeError("model construction: expected exactly one match, found %d" % count)

    loader_signature = (
        "    def load_model_state_partial(self, state_dict, allowed_missing_prefixes=()):\n"
    )
    text = replace_once(
        text,
        loader_signature,
        "    def load_model_state_partial(\n"
        "        self,\n"
        "        state_dict,\n"
        "        allowed_missing_prefixes=(),\n"
        "        allowed_unexpected_prefixes=(),\n"
        "    ):\n",
        "partial loader signature",
    )

    loader_body_anchor = """        if bad_missing or unexpected:
            raise RuntimeError(
                'Partial baseline load failed. bad_missing=%s unexpected=%s'
                % (bad_missing, unexpected)
            )
"""
    loader_body_replacement = """        bad_unexpected = [
            key
            for key in unexpected
            if not any(
                key.startswith(prefix)
                for prefix in allowed_unexpected_prefixes
            )
        ]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                'Partial baseline load failed. '
                'bad_missing=%s bad_unexpected=%s'
                % (bad_missing, bad_unexpected)
            )
"""
    text = replace_once(
        text,
        loader_body_anchor,
        loader_body_replacement,
        "partial loader unexpected-key filter",
    )

    init_anchor = """            self.load_model_state_partial(
                state_dict,
                allowed_missing_prefixes=("full_dea_head.",),
            )
"""
    init_replacement = """            if args.model_type == "full_dea":
                allowed_missing = ("full_dea_head.", "decidability_head.")
                allowed_unexpected = ()
            elif args.model_type == "dea_integrated":
                allowed_missing = (
                    "dea_cell_0.",
                    "dea_cell_1.",
                    "dea_cell_2.",
                    "dea_cell_3.",
                )
                # Current DEA-lite MSHNet checkpoints contain this legacy head;
                # original MSHNet checkpoints simply have no such keys.
                allowed_unexpected = ("decidability_head.",)
            else:
                allowed_missing = ("decidability_head.",)
                allowed_unexpected = ()
            self.load_model_state_partial(
                state_dict,
                allowed_missing_prefixes=allowed_missing,
                allowed_unexpected_prefixes=allowed_unexpected,
            )
"""
    text = replace_once(
        text,
        init_anchor,
        init_replacement,
        "baseline initialization",
    )

    text = replace_once(
        text,
        '            self.args.model_type != "full_dea"\n',
        '            self.args.model_type == "mshnet"\n',
        "DEA-lite model guard",
    )

    forward_tag_anchor = """    def get_forward_tag(self, epoch):
        if self.args.model_type == "full_dea":
            return epoch >= self.args.full_dea_start_epoch
        return epoch > self.warm_epoch
"""
    forward_tag_replacement = """    def get_forward_tag(self, epoch):
        if self.args.model_type == "full_dea":
            return epoch >= self.args.full_dea_start_epoch
        if self.args.model_type == "dea_integrated":
            return True
        return epoch > self.warm_epoch
"""
    text = replace_once(
        text,
        forward_tag_anchor,
        forward_tag_replacement,
        "complete Integrated DEA training forward",
    )

    train_forward_anchor = """                full_dea_out = out["full_dea"]
                dea_out = None
            elif use_dea:
"""
    train_forward_replacement = """                full_dea_out = out["full_dea"]
                dea_out = None
            elif self.args.model_type == "dea_integrated":
                out = self.model(data, tag, return_dict=True)
                masks = out["masks"]
                pred = out["pred"]
                dea_out = None
            elif use_dea:
"""
    text = replace_once(
        text,
        train_forward_anchor,
        train_forward_replacement,
        "Integrated DEA train branch",
    )

    text = replace_once(
        text,
        "        tag = False\n        with torch.no_grad():\n",
        "        tag = True\n        with torch.no_grad():\n",
        "complete evaluation forward",
    )
    obsolete_test_tag = """                if self.args.model_type == "full_dea":
                    tag = True
                elif epoch>self.warm_epoch:
                    tag = True

"""
    text = replace_once(
        text,
        obsolete_test_tag,
        "",
        "remove epoch-dependent test graph",
    )
    test_forward_anchor = """                if self.args.model_type == "full_dea":
                    out = self.model(data, tag, return_dict=True)
                    pred = out["pred"]
"""
    test_forward_replacement = """                if self.args.model_type in ("full_dea", "dea_integrated"):
                    out = self.model(data, tag, return_dict=True)
                    pred = out["pred"]
"""
    text = replace_once(
        text,
        test_forward_anchor,
        test_forward_replacement,
        "Integrated DEA evaluation branch",
    )

    return text, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", default="main.py")
    args = parser.parse_args()

    main_path = os.path.abspath(args.main)
    with open(main_path, "r", encoding="utf-8") as handle:
        original = handle.read()
    patched, changed = patch(original)
    if not changed:
        print("main.py already contains the dea_integrated entry; no changes made")
        return

    backup_path = main_path + ".pre_dea_integrated"
    if not os.path.exists(backup_path):
        shutil.copy2(main_path, backup_path)
    with open(main_path, "w", encoding="utf-8") as handle:
        handle.write(patched)
    print("patched %s" % main_path)
    print("backup: %s" % backup_path)


if __name__ == "__main__":
    main()

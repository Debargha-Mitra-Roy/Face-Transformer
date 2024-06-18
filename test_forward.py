import torch
import sys

from config import get_config

from vit_pytorch import ViT_face
from vit_pytorch import ViTs_face
from vit_pytorch import EfficientNet_V2_face
from vit_pytorch import EfficientNet_V2_ViT
from vit_pytorch import EfficientNet_V1_ViT
from vit_pytorch import EfficientNet_Trim_ViT
from vit_pytorch import CrossViT

from util.utils import (
    get_val_data,
    test_forward,
)

import argparse
import os


def main(args):
    print(args)
    DEVICE = torch.device("cuda:0")
    DATA_ROOT = "./Data/casia-webface/"

    # ======= Hyperparameters & Data Loaders =======#
    cfg = get_config(args)
    # Support:  ['Softmax', 'ArcFace', 'CosFace', 'SFaceLoss']
    HEAD_NAME = cfg["HEAD_NAME"]
    # Specify GPU ids
    GPU_ID = cfg["GPU_ID"]
    print("GPU_ID", GPU_ID)

    with open(os.path.join(DATA_ROOT, "property"), "r") as f:
        NUM_CLASS, h, w = [int(i) for i in f.read().split(",")]

    # ViT
    if args.network == "VIT":
        model = ViT_face(
            image_size=112,
            patch_size=8,
            loss_type=HEAD_NAME,
            GPU_ID=DEVICE,
            num_class=NUM_CLASS,
            dim=512,
            depth=20,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
        )

    # ViTs
    elif args.network == "VITs":
        model = ViTs_face(
            image_size=112,
            patch_size=8,
            loss_type=HEAD_NAME,
            GPU_ID=DEVICE,
            num_class=NUM_CLASS,
            ac_patch_size=12,
            pad=4,
            dim=512,
            depth=20,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
        )

    # EfficientNet_V2_face
    elif args.network == "EffNet_V2_face":
        model = (
            EfficientNet_V2_face(
                GPU_ID=GPU_ID,
                pretrained=True,
                fine_tune=True,
                loss_type=HEAD_NAME,
                dim=512,
                num_class=10572,
            ),
        )

    # EfficientNet_V2 + ViT
    elif args.network == "EffNet_V2_VIT":
        model = (
            EfficientNet_V2_ViT(
                GPU_ID=GPU_ID,
                pretrained=True,
                fine_tune=True,
                loss_type=HEAD_NAME,
                dim=512,
                num_class=10572,
            ),
        )

    # EfficientNet_V1 + ViT
    elif args.network == "EffNet_V1_VIT":
        model = (
            EfficientNet_V1_ViT(
                GPU_ID=GPU_ID,
                pretrained=True,
                fine_tune=True,
                loss_type=HEAD_NAME,
                dim=512,
                num_class=10572,
            ),
        )

    # Trimmed EfficientNet + ViT
    elif args.network == "EffNet_trim_VIT":
        model = (
            EfficientNet_Trim_ViT(
                GPU_ID,
                pretrained=True,
                fine_tune=False,
                loss_type=HEAD_NAME,
                dim=512,
                num_class=10572,
            ),
        )
    # CrossViT
    elif args.network == "CROSSVIT":
        model = (
            CrossViT(
                GPU_ID,
                loss_type=HEAD_NAME,
                dim=512,
                num_class=10572,
            ),
        )

    model_root = args.model
    model.load_state_dict(torch.load(model_root))

    TARGET = [i for i in args.target.split(",")]
    vers = get_val_data("./eval/", TARGET)
    for ver in vers:
        name, data_set, issame = ver
        time = test_forward(DEVICE, model, data_set)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="",
        help="pretrained model",
        type=str,
    )
    parser.add_argument(
        "--target",
        help="verification targets ['agedb_30', 'calfw','cfp_ff', 'cfp_fp', 'cplfw', 'lfw', 'sllfw', 'talfw']",
        default="lfw",
        type=str,
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))

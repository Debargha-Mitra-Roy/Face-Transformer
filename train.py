import os, argparse
import torch
import torch.nn as nn

from tensorboardX import SummaryWriter

from config import get_config
from image_iter import FaceDataset

from util.utils import (
    get_val_data,
    perform_val,
    buffer_val,
    AverageMeter,
    train_accuracy,
)

import time
from vit_pytorch import ViT_face
from vit_pytorch import ViTs_face
from vit_pytorch import EfficientNet_V2_face
from vit_pytorch import EfficientNet_V2_ViT
from vit_pytorch import EfficientNet_V1_ViT
from vit_pytorch import EfficientNet_Trim_ViT
from vit_pytorch import CrossViT

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer


# ======= Added epoch_change boolean value, such that the checkpoint is saved when a new epoch starts =======#
def need_save(acc, highest_acc):
    do_save = False
    save_cnt = 0
    if acc[0] > 0.49:
        do_save = True
    for i, accuracy in enumerate(acc):
        if accuracy > highest_acc[i]:
            highest_acc[i] = accuracy
            do_save = True
        if i > 0 and accuracy >= highest_acc[i] - 0.002:
            save_cnt += 1
    if save_cnt >= len(acc) * 3 / 4 and acc[0] > 0.99:
        do_save = True
    print("highest_acc:", highest_acc)
    return do_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="for face verification",
    )
    parser.add_argument(
        "-w",
        "--workers_id",
        help="gpu ids or cpu ['0', '1', '2', '3'] (default: cpu)",
        default="cpu",
        type=str,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        help="training epochs",
        default=125,
        type=int,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="batch size",
        default=256,
        type=int,
    )
    parser.add_argument(
        "-d",
        "--data_mode",
        help="use which database ['casia', 'vgg', 'ms1m', 'retina', 'ms1mr']",
        default="casia",
        type=str,
    )
    parser.add_argument(
        "-n",
        "--network",
        help="which network ['VIT', 'VITs', 'EffNet_V2_face', 'EffNet_V2_VIT', 'EffNet_V1_VIT', 'EffNet_trim_VIT', 'CROSSVIT']",
        default="VITs",
        type=str,
    )
    parser.add_argument(
        "-head",
        "--head",
        help="head type, ['Softmax', 'ArcFace', 'CosFace', 'SFaceLoss']",
        default="CosFace",
        type=str,
    )
    parser.add_argument(
        "-t",
        "--target",
        help="verification targets ['agedb_30', 'calfw','cfp_ff', 'cfp_fp', 'cplfw', 'lfw', 'sllfw', 'talfw']",
        default="lfw",
        type=str,
    )
    parser.add_argument(
        "-r",
        "--resume",
        help="resume model",
        default="",
        type=str,
    )
    parser.add_argument(
        "--outdir",
        help="output dir",
        default="",
        type=str,
    )
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw")',
    )
    parser.add_argument(
        "--opt-eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt-betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
        help="weight decay (default: 0.05)",
    )

    # ======= Learning rate schedule parameters =======#
    parser.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "cosine")',
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--lr-noise",
        type=float,
        nargs="+",
        default=None,
        metavar="pct, pct",
        help="learning rate noise on/off epoch percentages",
    )
    parser.add_argument(
        "--lr-noise-pct",
        type=float,
        default=0.67,
        metavar="PERCENT",
        help="learning rate noise limit percent (default: 0.67)",
    )
    parser.add_argument(
        "--lr-noise-std",
        type=float,
        default=1.0,
        metavar="STDDEV",
        help="learning rate noise std-dev (default: 1.0)",
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )

    parser.add_argument(
        "--decay-epochs",
        type=float,
        default=30,
        metavar="N",
        help="epoch interval to decay LR",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=3,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--cooldown-epochs",
        type=int,
        default=10,
        metavar="N",
        help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
    )
    parser.add_argument(
        "--patience-epochs",
        type=int,
        default=10,
        metavar="N",
        help="patience epochs for Plateau LR scheduler (default: 10)",
    )
    parser.add_argument(
        "--decay-rate",
        "--dr",
        type=float,
        default=0.1,
        metavar="RATE",
        help="LR decay rate (default: 0.1)",
    )
    args = parser.parse_args()

    # ======= Hyperparameters & Data Loaders =======#
    cfg = get_config(args)

    SEED = cfg["SEED"]  # Random Seed for Reproduce results
    torch.manual_seed(SEED)

    # Feature Dimension
    INPUT_SIZE = cfg["INPUT_SIZE"]
    EMBEDDING_SIZE = cfg["EMBEDDING_SIZE"]
    BATCH_SIZE = cfg["BATCH_SIZE"]
    NUM_EPOCH = cfg["NUM_EPOCH"]

    # The parent root where your train/val/test data are stored
    DATA_ROOT = cfg["DATA_ROOT"]
    EVAL_PATH = cfg["EVAL_PATH"]

    # The root to buffer your checkpoints and to log your train/val status
    WORK_PATH = cfg["WORK_PATH"]

    # The root to resume training from a saved checkpoint
    BACKBONE_RESUME_ROOT = cfg["BACKBONE_RESUME_ROOT"]
    BACKBONE_NAME = cfg["BACKBONE_NAME"]

    # Support:  ['Softmax', 'ArcFace', 'CosFace', 'SFaceLoss']
    HEAD_NAME = cfg["HEAD_NAME"]

    DEVICE = cfg["DEVICE"]
    MULTI_GPU = cfg["MULTI_GPU"]  # Flag to use multiple GPUs
    GPU_ID = cfg["GPU_ID"]  # Specify GPU IDs
    print("GPU_ID", GPU_ID)
    TARGET = cfg["TARGET"]

    print("=" * 60)

    print("Overall Configurations:")
    print(cfg)
    with open(os.path.join(WORK_PATH, "config.txt"), "w") as f:
        f.write(str(cfg))
    print("=" * 60)

    writer = SummaryWriter(WORK_PATH)  # Writer for buffering intermedium results
    torch.backends.cudnn.benchmark = True

    with open(os.path.join(DATA_ROOT, "property"), "r") as f:
        NUM_CLASS, h, w = [int(i) for i in f.read().split(",")]
    assert h == INPUT_SIZE[0] and w == INPUT_SIZE[1]

    dataset = FaceDataset(os.path.join(DATA_ROOT, "train.rec"), rand_mirror=True)
    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=len(GPU_ID),
        drop_last=True,
    )

    print("Number of Training Classes: {}".format(NUM_CLASS))

    vers = get_val_data(EVAL_PATH, TARGET)
    highest_acc = [0.0 for t in TARGET]

    # embed()

    # ======= Model, Loss & Optimizer =======#
    BACKBONE_DICT = {
        # ViT
        "VIT": ViT_face(
            loss_type=HEAD_NAME,
            GPU_ID=GPU_ID,
            num_class=NUM_CLASS,
            image_size=112,
            patch_size=8,
            dim=512,
            depth=20,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
        ),
        # ViTs
        "VITs": ViTs_face(
            loss_type=HEAD_NAME,
            GPU_ID=GPU_ID,
            num_class=NUM_CLASS,
            image_size=112,
            patch_size=8,
            ac_patch_size=12,
            pad=4,
            dim=512,
            depth=20,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
        ),
        # EfficientNet_V2_face
        "EffNet_V2_face": EfficientNet_V2_face(
            GPU_ID=GPU_ID,
            pretrained=True,
            fine_tune=True,
            loss_type=HEAD_NAME,
            dim=512,
            num_class=10572,
        ),
        # EfficientNet_V2 + ViT
        "EffNet_V2_VIT": EfficientNet_V2_ViT(
            GPU_ID,
            pretrained=False,
            fine_tune=True,
            loss_type=HEAD_NAME,
            dim=512,
            num_class=10572,
        ),
        # EfficientNet_V1 + ViT
        "EffNet_V1_VIT": EfficientNet_V1_ViT(
            GPU_ID,
            pretrained=True,
            fine_tune=True,
            loss_type=HEAD_NAME,
            dim=512,
            num_class=10572,
        ),
        # Trimmed EfficientNet + ViT
        "EffNet_trim_VIT": EfficientNet_Trim_ViT(
            GPU_ID,
            pretrained=True,
            fine_tune=False,
            loss_type=HEAD_NAME,
            dim=512,
            num_class=10572,
        ),
        # CrossViT
        "CROSSVIT": CrossViT(
            GPU_ID=GPU_ID,
            loss_type=HEAD_NAME,
            image_size=112,
            sm_dim=512,
            lg_dim=512,
            num_class=10572,
        ),
    }

    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]

    print("=" * 60)
    print(BACKBONE)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)

    LOSS = nn.CrossEntropyLoss()

    # embed()

    OPTIMIZER = create_optimizer(args, BACKBONE)
    print("=" * 60)
    print(OPTIMIZER)
    print("Optimizer Generated")
    print(f"When initialized Learning Rate = {OPTIMIZER.param_groups[0]['lr']}")
    print("=" * 60)

    lr_scheduler, _ = create_scheduler(args, OPTIMIZER)

    START_EPOCH = 0
    print("Device {}".format(cfg["DEVICE"]))
    print("GPU_ID allotted is {}".format(cfg["GPU_ID"][0]))

    # ======= Optionally resume from a checkpoint =======#
    if BACKBONE_RESUME_ROOT:
        print("=" * 60)
        print(BACKBONE_RESUME_ROOT)
        if os.path.isfile(BACKBONE_RESUME_ROOT):
            # BACKBONE = nn.DataParallel(BACKBONE, device_ids=GPU_ID)
            BACKBONE = BACKBONE.to(DEVICE)
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            checkpoint = torch.load(
                BACKBONE_RESUME_ROOT,
                map_location=torch.device("cuda:0"),
            )
            BACKBONE.load_state_dict(
                {
                    key.replace("module.", ""): val
                    for key, val in checkpoint["model_state_dict"].items()
                }
            )
            lr_scheduler, _ = create_scheduler(
                args, OPTIMIZER
            )  # Scheduler should be created after loading optimizer
            lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            OPTIMIZER.load_state_dict(checkpoint["optimizer_state_dict"])

        else:
            print(
                "No Checkpoint Found at '{}'. Please Have a Check or Continue to Train from Scratch".format(
                    BACKBONE_RESUME_ROOT
                )
            )
        print("=" * 60)

    # Multi-GPU setting
    if MULTI_GPU:
        BACKBONE = nn.DataParallel(BACKBONE, device_ids=GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)

    # Single-GPU setting
    else:
        BACKBONE = BACKBONE.to(DEVICE)

    # ======= Train, Validation & Save checkpoint =======#
    DISP_FREQ = 10  # Frequency to display training loss & accuracy
    VER_FREQ = 100
    SAVE_FREQ = 100

    batch = 0  # Batch Index

    losses = AverageMeter()
    top1 = AverageMeter()

    BACKBONE.train()  # Set to training mode

    # ======= The epoch starts from the checkpoint epoch =======#
    for epoch in range(START_EPOCH, NUM_EPOCH):  # Start Training process
        print(
            "Epoch {} : Learing rate before scheduler step is called {}".format(
                epoch, OPTIMIZER.param_groups[0]["lr"]
            )
        )
        lr_scheduler.step(epoch)
        print(
            "Epoch {} : Learing rate {} ".format(epoch, OPTIMIZER.param_groups[0]["lr"])
        )
        last_time = time.time()

        for inputs, labels in iter(trainloader):
            # ======= Compute output =======#
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).long()

            # ======= The EfficientNet_Trim_ViT backbone only requires the input values, thus when the backbone is EfficientNet_Trim_ViT, pass the inputs, else the labels and the inputs is passed =======#
            if BACKBONE_NAME == "EfficientNet_Trim_ViT":
                outputs, emb = BACKBONE(inputs.float(), labels)
            else:
                outputs, emb = BACKBONE(inputs.float(), labels)

            loss = LOSS(outputs, labels)

            # print("outputs", outputs, outputs.data)

            # ======= Measure accuracy and Record loss =======#
            prec1 = train_accuracy(outputs.data, labels, topk=(1,))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.data.item(), inputs.size(0))

            # ======= Compute Gradient Descent & do SGD step =======#
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()

            # ======= Display training loss & accuracy every DISP_FREQ (buffer for visualization) =======#
            if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                epoch_loss = losses.avg
                epoch_acc = top1.avg
                writer.add_scalar("Training/Training_Loss", epoch_loss, batch + 1)
                writer.add_scalar("Training/Training_Accuracy", epoch_acc, batch + 1)
                batch_time = time.time() - last_time
                last_time = time.time()
                print(
                    "Epoch {} Batch {}\t"
                    "Speed: {speed:.2f} samples/s\t"
                    "Training Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        epoch + 1,
                        batch + 1,
                        speed=inputs.size(0) * DISP_FREQ / float(batch_time),
                        loss=losses,
                        top1=top1,
                    )
                )
                losses = AverageMeter()
                top1 = AverageMeter()

            # ======= Added another condition that when epoch changes i.e (batch % number_of_batches) is 0 =======#
            if (
                (batch + 1) % VER_FREQ == 0
            ) and batch != 0:  # Perform validation & save checkpoints (buffer for visualization)
                for params in OPTIMIZER.param_groups:
                    lr = params["lr"]
                    break
                print("Learning rate %f" % lr)
                print("Perform Evaluation on ", TARGET, ", and Save Checkpoints...")
                acc = []
                for ver in vers:
                    name, data_set, issame = ver
                    accuracy, std, xnorm, best_threshold, roc_curve = perform_val(
                        MULTI_GPU,
                        DEVICE,
                        EMBEDDING_SIZE,
                        BATCH_SIZE,
                        BACKBONE,
                        data_set,
                        issame,
                    )
                    buffer_val(
                        writer,
                        name,
                        accuracy,
                        std,
                        xnorm,
                        best_threshold,
                        roc_curve,
                        batch + 1,
                    )
                    print("[%s][%d]XNorm: %1.5f" % (name, batch + 1, xnorm))
                    print(
                        "[%s][%d]Accuracy-Flip: %1.5f+-%1.5f"
                        % (name, batch + 1, accuracy, std)
                    )
                    print(
                        "[%s][%d]Best-Threshold: %1.5f"
                        % (name, batch + 1, best_threshold)
                    )
                    acc.append(accuracy)

                    # ======= Save checkpoints per epoch =======#
                    if (batch + 1) % SAVE_FREQ == 0:
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": BACKBONE.state_dict(),
                                "optimizer_state_dict": OPTIMIZER.state_dict(),
                                "scheduler_state_dict": lr_scheduler.state_dict(),
                            },
                            os.path.join(
                                WORK_PATH,
                                "Backbone_{}_checkpoint.pth".format(BACKBONE_NAME),
                            ),
                        )

                    BACKBONE.train()  # Set to Training mode

            batch += 1  # Batch Index

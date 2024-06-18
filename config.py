import torch
import os


def get_config(args):
    # ======= Feature Dimension =======#
    configuration = dict(
        SEED=1337,  # Random seed for reproduce results
        INPUT_SIZE=[112, 112],  # Support: [112, 112] and [224, 224]
        EMBEDDING_SIZE=512,
    )

    if args.workers_id == "cpu" or not torch.cuda.is_available():
        configuration["GPU_ID"] = []
        print("check", args.workers_id, torch.cuda.is_available())
    else:
        configuration["GPU_ID"] = [int(i) for i in args.workers_id.split(",")]

    if len(configuration["GPU_ID"]) == 0:
        configuration["DEVICE"] = torch.device("cpu")
        configuration["MULTI_GPU"] = False
    else:
        configuration["DEVICE"] = torch.device("cuda:%d" % configuration["GPU_ID"][0])
        if len(configuration["GPU_ID"]) == 1:
            configuration["MULTI_GPU"] = True
        else:
            configuration["MULTI_GPU"] = True

    configuration["NUM_EPOCH"] = args.epochs
    configuration["BATCH_SIZE"] = args.batch_size

    if args.data_mode == "retina":
        configuration["DATA_ROOT"] = "./Data/ms1m-retinaface-t1/"
    elif args.data_mode == "casia":
        configuration["DATA_ROOT"] = "./Data/casia-webface/"
    else:
        raise Exception(args.data_mode)

    configuration["EVAL_PATH"] = "./eval/"

    assert args.network in [
        "VIT",
        "VITs",
        "EffNet_V2_face",
        "EffNet_V2_VIT",
        "EffNet_V1_VIT",
        "EffNet_trim_VIT",
        "CROSSVIT",
    ]
    configuration["BACKBONE_NAME"] = args.network

    assert args.head in ["Softmax", "ArcFace", "CosFace", "SFaceLoss"]
    configuration["HEAD_NAME"] = args.head

    configuration["TARGET"] = [i for i in args.target.split(",")]

    # ======= The root to resume training from a saved checkpoint =======#
    if args.resume:
        configuration["BACKBONE_RESUME_ROOT"] = args.resume
    else:
        configuration["BACKBONE_RESUME_ROOT"] = ""

    # ======= The root to buffer your checkpoints =======#
    configuration["WORK_PATH"] = args.outdir

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    return configuration

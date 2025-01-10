import argparse

parser = argparse.ArgumentParser(description="CycleFusion Parameters")

# Architecture
parser.add_argument(
    "--network_type",
    default="CDDFuse",
    type=str,
    choices=["CDDFuse", "DenseFuse"],
    help="Choose the network type",
)
parser.add_argument(
    "--with_quality",
    default=True,
    type=bool,
    help="Train with quality",
)
parser.add_argument(
    "--with_revaluate",
    default=True,
    type=bool,
    help="Train with revaluate quality of fused image",
)
parser.add_argument(
    "--fuse_type",
    default="pow",
    type=str,
    choices=["add", "pow", "exp"],
    help="Choose the fuse type",
)

# Train
parser.add_argument(
    "--device",
    default=0,
    type=int,
    help="set gpu device",
)
parser.add_argument(
    "--pretrain_weight",
    default="output/xxx/models/final.pth",
    type=str,
    help="pretrain weight",
)
parser.add_argument(
    "--epochs",
    default=100,
    type=int,
    help="number of training epochs",
)
parser.add_argument(
    "--batch_size",
    default=2,
    type=int,
    help="batch size for training",
)
parser.add_argument(
    "--lr",
    default=1e-4,
    type=float,
    help="learning rate",
)
parser.add_argument(
    "--critic",
    default=5,
    type=int,
    help="train deformer and fuser at integer multiples of n",
)
parser.add_argument(
    "--patience",
    default=10,
    type=int,
    help="Reduce the learning rate when the train loss does not drop for this many epochs",
)

# loss weights
parser.add_argument(
    "--quality_weight",
    default=1.0,
    type=float,
    help="quality loss weight",
)
parser.add_argument(
    "--restore_weight",
    default=1.0,
    type=float,
    help="restore loss weight",
)
parser.add_argument(
    "--fuse_weight",
    default=1.0,
    type=float,
    help="fuse weight with tir in [0, 1]",
)

# Datasets
parser.add_argument(
    "--data_dir",
    default="images/dataset5",
    type=str,
    help="input data directory",
)
parser.add_argument(
    "--test_num",
    default=20,
    type=int,
    help="the number of testing to dataset",
)
parser.add_argument(
    "--image_size",
    default=256,
    type=int,
    help="size of training images",
)

# Output
parser.add_argument(
    "--output_dir",
    default="output",
    type=str,
    help="outputs directory",
)
parser.add_argument(
    "--save_interval",
    default=5,
    type=int,
    help="save test results at integer multiples of n",
)

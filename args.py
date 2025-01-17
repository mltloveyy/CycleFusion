import argparse

parser = argparse.ArgumentParser(description="CycleFusion Parameters")

# Architecture
parser.add_argument(
    "--network_type",
    default="cddfuse",
    type=str,
    choices=["cddfuse", "densefuse"],
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
    default="feature",
    type=str,
    choices=["add", "exp", "pow", "feature", "cddfuse"],
    help="Choose the fuse type",
)

# Train
parser.add_argument(
    "--device",
    default=0,
    type=int,
    help="Set gpu device",
)
parser.add_argument(
    "--pretrain_weight",
    default="output/xxx/models/final.pth",
    type=str,
    help="The path to pretrain weight",
)
parser.add_argument(
    "--epochs",
    default=100,
    type=int,
    help="Number of training epochs",
)
parser.add_argument(
    "--batch_size",
    default=2,
    type=int,
    help="Batch size for training",
)
parser.add_argument(
    "--lr",
    default=1e-4,
    type=float,
    help="Learning rate",
)
parser.add_argument(
    "--critic",
    default=5,
    type=int,
    help="Interval training at integer multiples of n",
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
    help="Quality loss weight",
)
parser.add_argument(
    "--restore_weight",
    default=1.0,
    type=float,
    help="Restore loss weight",
)
parser.add_argument(
    "--fuse_weight",
    default=1.0,
    type=float,
    help="Fuse loss weight",
)
parser.add_argument(
    "--regular_weight",
    default=1.0,
    type=float,
    help="Regular loss weight",
)
parser.add_argument(
    "--quality_threshold",
    default=0.8,
    type=float,
    help="The threshold of quality to be considered as good quality",
)

# Datasets
parser.add_argument(
    "--data_dir",
    default="images/dataset5",
    type=str,
    help="Input data directory",
)
parser.add_argument(
    "--test_num",
    default=20,
    type=int,
    help="The number of testing to dataset",
)
parser.add_argument(
    "--image_size",
    default=256,
    type=int,
    help="Size of training images",
)

# Output
parser.add_argument(
    "--output_dir",
    default="output",
    type=str,
    help="Outputs directory",
)
parser.add_argument(
    "--save_interval",
    default=5,
    type=int,
    help="Save test results at integer multiples of n",
)

import argparse

parser = argparse.ArgumentParser(description="CycleFusion Parameters")

# Train
parser.add_argument(
    "--is_train",
    default=True,
    type=bool,
    help="run train or test",
)
parser.add_argument(
    "--seed",
    default=40,
    type=int,
    help="random seed for training",
)
parser.add_argument(
    "--pretrain_weight",
    default="output/models/aaa",
    type=str,
    help="pretrain weight",
)
parser.add_argument(
    "--epochs",
    default=2,
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
    default=1e-3,
    type=float,
    help="learning rate",
)
parser.add_argument(
    "--critic",
    default=2,
    type=int,
    help="train deformer and fuser at integer multiples of n",
)
parser.add_argument(
    "--patience",
    default=5,
    type=int,
    help="training will stop when the validation loss does not improve for this many epochs",
)


# loss weights
parser.add_argument(
    "--ssim_weight",
    default=10.0,
    type=float,
    help="ssim loss weight",
)
parser.add_argument(
    "--fuse_weight",
    default=0.7,
    type=float,
    help="fuse weight with tir in [0, 1]",
)
parser.add_argument(
    "--quality_thresh",
    default=0.5,
    type=float,
    help="quality threshold",
)
parser.add_argument(
    "--regular_factor",
    default=0.1,
    type=float,
    help="regular factor with quality",
)

# Datasets
parser.add_argument(
    "--data_dir",
    default="images",
    type=str,
    help="input data directory",
)
parser.add_argument(
    "--test-num",
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

import inspect
import math
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from src.lightning import MLP, CNN, LinearModel, ResNet18, RiskModel
from src.dataset import PathMnist, NLST
from lightning.pytorch.cli import LightningArgumentParser
import lightning.pytorch as pl
from torch.cuda import device_count

NAME_TO_MODEL_CLASS = {
    "mlp": MLP,
    "cnn": CNN,
    "linear": LinearModel,
    "resnet": ResNet18,
    "risk_model": RiskModel
}

NAME_TO_DATASET_CLASS = {
    "pathmnist": PathMnist,
    "nlst": NLST
}

dirname = os.path.dirname(__file__)

def add_main_args(parser: LightningArgumentParser) -> LightningArgumentParser:

    parser.add_argument(
        "--model_name",
        default="mlp",
        choices=["mlp", "linear", "cnn", "resnet", "risk_model"],
        help="Name of model to use",
    )

    parser.add_argument(
        "--dataset_name",
        default="pathmnist",
        choices=["pathmnist", "nlst"],
        help="Name of dataset to use"
    )

    parser.add_argument(
        "--project_name",
        default="cornerstone_project_2",
        help="Name of project for wandb"
    )

    parser.add_argument(
        "--monitor_key",
        default="val_loss",
        help="Name of metric to use for checkpointing. (e.g. val_loss, val_acc)"
    )

    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Path to checkpoint to load from. If None, init from scratch."
    )

    parser.add_argument(
        "--train",
        default=False,
        action="store_true",
        help="Whether to train the model."
    )

    parser.add_argument(
        "--num_layers",
        default=1,
        type=int,
        help="Depth of the model (number of layers)",
    )

    parser.add_argument(
        "--use_bn",
        default=False,
        type=bool,
        help="Whether to batch normalize in each layer",
    )

    parser.add_argument(
        "--hidden_dim",
        default=512,
        type=int,
        help="The dimension of the hidden layer(s)"
    )

    parser.add_argument(
        "--use_data_augmentation",
        default=False,
        type=bool,
        help="Whether to augment the data"
    )

    parser.add_argument(
        "--pretraining",
        default=False,
        type=bool,
        help="Whether to use pretrained model weights (only used for resnet)"
    )

    parser.add_argument(
        "--wandb_entity",
        default='CPH29',
        type=str,
        help="The wandb account to log metrics and models to"
    )

    return parser

def parse_args() -> argparse.Namespace:
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(pl.Trainer, nested_key="trainer")
    for model_name, model_class in NAME_TO_MODEL_CLASS.items():
        parser.add_lightning_class_args(model_class, nested_key=model_name)
    for dataset_name, data_class in NAME_TO_DATASET_CLASS.items():
        parser.add_lightning_class_args(data_class, nested_key=dataset_name)
    parser = add_main_args(parser)
    args = parser.parse_args()
    return args

def get_caller():
    caller = os.path.split(inspect.getsourcefile(sys._getframe(1)))[-1]
    return caller

def get_datamodule_num_workers(num_process_workers=None):
    # set per https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5
    num_process_workers = num_process_workers if num_process_workers else 1
    datamodule_num_workers = device_count() * 4
    n_cpus = os.cpu_count()
    if datamodule_num_workers * num_process_workers >= n_cpus:
        datamodule_num_workers = math.floor(n_cpus/num_process_workers * .9) 
    return datamodule_num_workers

def main(args: argparse.Namespace):
    print(args)
    print("Loading data ..")

    print("Preparing lighning data module (encapsulates dataset init and data loaders)")
    """
        Most the data loading logic is pre-implemented in the LightningDataModule class for you.
        However, you may want to alter this code for special localization logic or to suit your risk
        model implementations
    """
    # get workers for datamodule
    num_workers = args.num_workers if 'num_workers' in vars(args) else None
    datamodule_num_workers = get_datamodule_num_workers(num_workers)
    
    # get datamodule args
    datamodule_args = vars(args[args.dataset_name])
    datamodule_args.update({'num_workers': datamodule_num_workers})

    # init data module
    datamodule = NAME_TO_DATASET_CLASS[args.dataset_name](**datamodule_args)

    print("Initializing model")
    if args.checkpoint_path is None:
        model = NAME_TO_MODEL_CLASS[args.model_name](**vars(args[args.model_name]))
    else:
        model = NAME_TO_MODEL_CLASS[args.model_name].load_from_checkpoint(args.checkpoint_path)

    print("Initializing trainer")
    logger = pl.loggers.WandbLogger(project=args.project_name,
                                    entity=args.wandb_entity,
                                    group=args.model_name)

    args.trainer.accelerator = 'auto'
    args.trainer.strategy = 'ddp'
    args.trainer.logger = logger
    args.trainer.precision = "bf16-mixed" ## This mixed precision training is highly recommended

    # set checkpoint save directory
    dirpath = os.path.join(dirname, '../models', args.model_name)
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)

    args.trainer.callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor=args.monitor_key,
            mode='min' if "loss" in args.monitor_key else "max",
            dirpath=dirpath,
            filename=args.model_name + '-{epoch:002d}-{val_loss:.2f}',
            save_last=True
        )]

    trainer = pl.Trainer(**vars(args.trainer))

    if args.train:
        print("Training model")
        trainer.fit(model, datamodule)

    print("Best model checkpoint path: ", trainer.checkpoint_callback.best_model_path)

    print("Evaluating model on validation set")
    trainer.validate(model, datamodule)

    print("Evaluating model on test set")
    trainer.test(model, datamodule)

    print("Done")


if __name__ == '__main__':
    __spec__ = None
    args = parse_args()
    main(args)
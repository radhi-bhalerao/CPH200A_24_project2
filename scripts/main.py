import inspect
import math
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from src.lightning import MLP, CNN, LinearModel, ResNet18, RiskModel, ResNet3D, Swin3DModel, ResNet18_adapted
from src.dataset import PathMnist, NLST
from lightning.pytorch.cli import LightningArgumentParser
import lightning.pytorch as pl
from torch.cuda import device_count
import wandb
from torch.backends import cudnn
from lightning import seed_everything
import json
import torch

dirname = os.path.dirname(__file__)
global_seed = json.load(open(os.path.join(dirname, '..', 'global_seed.json')))['global_seed']
seed_everything(global_seed)
cudnn.benchmark = False

NAME_TO_MODEL_CLASS = {
    "mlp": MLP,
    "cnn": CNN,
    "cnn3d": CNN3D,
    "linear": LinearModel,
    "resnet": ResNet18,
    "resnet_adapt": ResNet18_adapted,
    "resnet3d": ResNet3D,
    "swin3d": Swin3DModel,
    "risk_model": RiskModel
}


NAME_TO_DATASET_CLASS = {
    "pathmnist": PathMnist,
    "nlst": NLST
}

MODEL_TO_DATASET = {
    "mlp": "pathmnist",
    "cnn": "pathmnist",
    "cnn3d": "nlst",
    "linear": "pathmnist",
    "resnet": "pathmnist",
    "resnet_adapt": "nlst",
    "resnet3d": "nlst",
    "swin3d": "nlst",
    "risk_model": "nlst"
}

dirname = os.path.dirname(__file__)

def add_main_args(parser: LightningArgumentParser) -> LightningArgumentParser:

    parser.add_argument(
        "--model_name",
        default="mlp",
        choices=["mlp", "linear", "cnn", "cnn3d", "resnet", "resnet_adapt", "risk_model", "swin3d", "resnet3d"],  
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
        type=bool,
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

    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of processes to running in parallel"
    )

    parser.add_argument(
        "--class_balance",
        default=False,
        type=bool,
        help="Whether to perform class-balanced sampling (only used for nlst dataset)"
    )

    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Number of samples per batch"
    )

    parser.add_argument(
        "--group_keys",
        default=['race', 'educat', 'gender', 'age', 'ethnic'],
        nargs='*',
        help="The groups to perform subgroup analysis on (only used for nlst dataset)"
    )

    parser.add_argument(
    "--depth_handling",
    default="max_pool",
    choices=["max_pool", "avg_pool", "slice_attention", "3d_conv"],
    help="Method to handle depth dimension in ResNet18_adapted"
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
    datamodule_num_workers = device_count() * 8
    n_cpus = os.cpu_count()
    if datamodule_num_workers * num_process_workers >= n_cpus:
        datamodule_num_workers = math.floor(n_cpus/num_process_workers * .9) 
    return datamodule_num_workers

def get_datamodule(args):
    # get workers for datamodule
    datamodule_num_workers = get_datamodule_num_workers(args.num_workers)
    
    # get datamodule args
    datamodule_vars = vars(vars(args)[args.dataset_name])
    update_vars = {k:v for k,v in vars(args).items() if k in datamodule_vars}
    datamodule_vars.update(update_vars)
    datamodule_vars.update({'num_workers': datamodule_num_workers})

    # init data module
    datamodule = NAME_TO_DATASET_CLASS[args.dataset_name](**datamodule_vars)

    return datamodule

def get_model(args):
    print(f"Initializing {args.model_name} model")
    
    # Check if the model is `resnet_adapt` to handle the specific `depth_handling` parameter
    if args.model_name == "resnet_adapt":
        model_vars = vars(vars(args)[args.model_name])
        update_vars = {k: v for k, v in vars(args).items() if k in model_vars or k == 'depth_handling'}
        model_vars.update(update_vars)
    else:
        # General model initialization without `depth_handling`
        model_vars = vars(vars(args)[args.model_name])
        update_vars = {k: v for k, v in vars(args).items() if k in model_vars}
        model_vars.update(update_vars)
    
    # Initialize the model either from scratch or from a checkpoint
    if args.checkpoint_path is None:
        print('with params ', model_vars)
        model = NAME_TO_MODEL_CLASS[args.model_name](**model_vars)
    else:
        model = NAME_TO_MODEL_CLASS[args.model_name].load_from_checkpoint(args.checkpoint_path)

    return model


def get_trainer(args, strategy='ddp', logger=None, callbacks=[]):
    args.trainer.accelerator = 'auto'
    args.trainer.strategy = strategy
    args.trainer.logger = logger
    args.trainer.precision = "bf16-mixed" ## This mixed precision training is highly recommended
    args.trainer.min_epochs = 100

    # set checkpoint save directory
    dirpath = os.path.join(dirname, '../models', args.model_name)
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)

    args.trainer.callbacks = callbacks

    # init trainer
    trainer_args = vars(args.trainer)
    trainer = pl.Trainer(**trainer_args)

    return trainer

def get_logger(args):
    logger = pl.loggers.WandbLogger(project=args.project_name,
                                    entity=args.wandb_entity,
                                    group=args.model_name,
                                    dir=os.path.join(dirname, '..'))

    return logger

def get_callbacks(args):
    # set checkpoint save directory
    dirpath = os.path.join(dirname, '../models', args.model_name)
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor=args.monitor_key,
            mode='min' if "loss" in args.monitor_key else "max",
            dirpath=dirpath,
            filename=args.model_name + '-{epoch:002d}-{val_loss:.2f}',
            save_last=True,
            every_n_epochs=1
        ),
        pl.callbacks.EarlyStopping(
            monitor=args.monitor_key,
            mode='min' if "loss" in args.monitor_key else "max",
            patience=10,
            check_on_train_epoch_end=True
        )]
    
    return callbacks


def main(args: argparse.Namespace):
    print("Loading data ..")
    print(f"Training: {args.train}")
    print(f"Model Name: {args.model_name}")
    print(f"Dataset Name: {args.dataset_name}")
    print(f"Pretraining: {args.pretraining}")
    print(f"Num Workers: {args.num_workers}")

    print("Preparing lighning data module (encapsulates dataset init and data loaders)")
    """
        Most the data loading logic is pre-implemented in the LightningDataModule class for you.
        However, you may want to alter this code for special localization logic or to suit your risk
        model implementations
    """

    datamodule = get_datamodule(args)
    model = get_model(args)
    logger = get_logger(args)
    callbacks = get_callbacks(args)
    trainer = get_trainer(args, callbacks=callbacks, logger=logger)
    torch.cuda.set_per_process_memory_fraction(0.8, device=0)

    if args.train:
        print("Training model")
        trainer.fit(model, datamodule)

    print("Best model checkpoint path: ", trainer.checkpoint_callback.best_model_path)

    print("Evaluating model on validation set")
    trainer.validate(model, datamodule)

    print("Evaluating model on test set")
    trainer.test(model, datamodule)

    if logger:
        logger.finalize('success')
    wandb.finish()

    print("Done")


if __name__ == '__main__':
    __spec__ = None
    args = parse_args()
    main(args)


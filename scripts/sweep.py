import os
import json
import wandb
from operator import mul
from functools import reduce
from itertools import product
from copy import deepcopy
from argparse import Namespace
from main import get_model, get_callbacks, get_logger, get_datamodule, get_trainer, parse_args as train_model_parse_args, \
                NAME_TO_DATASET_CLASS, NAME_TO_MODEL_CLASS, MODEL_TO_DATASET
from scripts.dispatcher1 import parse_args as dispatcher_parse_args

dirname = os.path.dirname(__file__)

# get default args from dispatcher & main
train_model_args = train_model_parse_args()
dispatcher_args = dispatcher_parse_args()
param_config = json.load(open(dispatcher_args.config_path, "r"))
    
def get_train_model(model_name):
    # get training args
    train_model_vars = vars(deepcopy(train_model_args))

    # get dataset for model
    dataset_name = MODEL_TO_DATASET[model_name]
    
    # update default args
    train_model_vars['dataset_name'] = dataset_name
    train_model_vars['model_name'] = model_name

    # remove extra models from args
    for model_i in NAME_TO_MODEL_CLASS.keys():
        if model_i != model_name:
            train_model_vars.pop(model_i, None)

    # remove extra datasets from args
    for dataset_i in NAME_TO_DATASET_CLASS.keys():
        if dataset_i != dataset_name:
            train_model_vars.pop(dataset_i, None)

    def train_model():
        # init wandb
        wandb.init(project=train_model_vars['project_name'],
                    entity=train_model_vars['wandb_entity'],
                    group=model_name,
                    dir=os.path.join(dirname, '..'))

        # create config with required default args & wandb sweep params
        config = {**train_model_vars, **vars(dispatcher_args), **wandb.config}
        update_model_args(config)
        update_datamodule_args(config)

        config = Namespace(**config)

        # init training components
        datamodule = get_datamodule(config)
        model = get_model(config)
        callbacks = get_callbacks(config)
        logger = get_logger(config)
        trainer = get_trainer(config, strategy='ddp_notebook', logger=logger, callbacks=callbacks)

        # update wandb config
        wandb.config.update(vars(config))

        print("Training model")
        trainer.fit(model, datamodule)

        print("Best model checkpoint path: ", trainer.checkpoint_callback.best_model_path)

    return train_model   

def update_model_args(config):
    model_params = param_config['args_by_model_name'][model_name]
    model_vars = vars(config[model_name])
    model_vars.update({k:v for k,v in config.items() if k in model_params})

def update_datamodule_args(config):
    dataset_name = config['dataset_name']
    datamodule_vars = vars(config[dataset_name])
    datamodule_vars.update({k:v for k,v in config.items() if k in datamodule_vars})

if __name__ == '__main__':
    for model_name in ['linear', 'mlp', 'cnn', 'resnet']:
        print(f'Running sweep for model "{model_name}"')
    
        # get search parameters
        parameters = {k: {'values': v} for k,v in param_config.items() if k in param_config['args_by_model_name'][model_name]}

        # get sweep function
        train_model = get_train_model(model_name)
        count = reduce(mul, [len(v['values']) for v in parameters.values()], 1) # number of config trials to run

        # define sweep config
        sweep_config = {
            'method': 'grid',
            'name': f'sweep_{model_name}',
            'metric': {
                'goal': 'minimize',
                'name': 'val_loss'
            },
            'parameters': parameters
        }

        # init sweep
        sweep_id = wandb.sweep(sweep_config, 
                               project=train_model_args.project_name,
                               entity=train_model_args.wandb_entity)

        wandb.agent(sweep_id=sweep_id, 
                    function=train_model, 
                    count=count)
        
        wandb.teardown()
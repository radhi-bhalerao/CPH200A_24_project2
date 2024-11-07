import os
import json
import wandb
from operator import mul
from functools import reduce
from argparse import Namespace
from main import get_model, get_callbacks, get_datamodule, get_trainer, parse_args as train_model_parse_args
from scripts.dispatcher1 import parse_args as dispatcher_parse_args

dirname = os.path.dirname(__file__)

# get default args from dispatcher & main
dispatcher_args = dispatcher_parse_args()
train_model_args = train_model_parse_args()

def get_train_model(args):

    def train_model():
        # init callbacks & trainer
        callbacks = get_callbacks(args)
        trainer = get_trainer(args, strategy='ddp_notebook', callbacks=callbacks)

        # init wandb
        wandb.init(project=args.project_name,
                    entity=args.wandb_entity,
                    group=args.model_name,
                    dir=os.path.join(dirname, '..'))

        # create config with required default args & wandb sweep params
        config = {**vars(train_model_args), **vars(dispatcher_args), **wandb.config}

        # init model & datamodule with sweep params
        datamodule = get_datamodule(Namespace(**config))
        model = get_model(Namespace(**config))

        # fit the model
        trainer.fit(model, datamodule)    

    return train_model   


if __name__ == '__main__':
    for model_name in ['linear', 'mlp', 'cnn', 'resnet']:
        print(f'Running sweep for model "{model_name}"')
    
        # get search parameters
        param_config = json.load(open(dispatcher_args.config_path, "r"))
        parameters = {k: {'values': v} for k,v in param_config.items() if k in param_config['args_by_model_name'][model_name]}

        # get sweep function
        train_model = get_train_model(train_model_args)
        count = reduce(mul, [len(v['values']) for v in parameters.values()], 1)

        # define sweep config
        sweep_config = {
            'method': 'grid',
            'name': 'sweep_1.1',
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
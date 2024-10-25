import os
import argparse
import json
import random
import torch.multiprocessing as mp
from itertools import product
from main import main as train_model, parse_args as train_model_parse_args

dirname = os.path.dirname(__file__)

def add_main_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join(dirname, "../grid_search.json"),
        help="Location of config file"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of processes to run in parallel"
    )
    
    parser.add_argument(
        "--train",
        default=True,
        action="store_false",
        help="Whether to train the model."
    )

    return parser

def get_experiment_list(config: dict) -> (list[dict]):
    '''
    Parses an experiment config, and creates jobs. For flags that are expected to be a single item, but the config contains a list, this will return one job for each item in the list.
    :config - experiment_config

    returns: jobs - a list of dicts, each of which encapsulates one job.
        *Example: {learning_rate: 0.001 , batch_size: 16 ...}
    '''
    jobs = []

    # Go through the tree of possible jobs and enumerate into a list of jobs
    for model_name, valid_args in config['args_by_model_name'].items():

        for elements in product(*[config[arg] for arg in valid_args]):

            config_i = dict(zip(valid_args, elements))
            config_i.update({'model_name': model_name})

            jobs.append(config_i)

    return jobs

def worker(args: argparse.Namespace, job_queue: mp.SimpleQueue, done_queue: mp.SimpleQueue):
    '''
    Worker thread for each worker. Consumes all jobs and pushes results to done_queue.
    :args - command line args
    :job_queue - queue of available jobs.
    :done_queue - queue where to push results.
    '''
    while not job_queue.empty():
        params = job_queue.get()
        if params is None:
            return
        done_queue.put(
            launch_experiment(args, params))

def launch_experiment(args: argparse.Namespace, experiment_config: dict) ->  dict:
    '''
    Launch an experiment and direct logs and results to a unique filepath.
    :configs: flags to use for this model run. Will be fed into
    scripts/main.py

    '''

    # get default model training args
    train_model_args = train_model_parse_args()
    d = vars(train_model_args)

    # update train_model_args with items shared by args
    d.update({k: v for k, v in vars(args).items() if k in d})
    
    # update model namespace args
    model_d = vars(d[d[experiment_config['model_name']]])
    model_d.update({k: v for k, v in experiment_config.items() if k in model_d})
    
    # run experiment
    train_model(train_model_args)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_main_args(parser)
    args = parser.parse_args()
    return args

def main(args: argparse.Namespace) -> dict:
    print(args)
    config = json.load(open(args.config_path, "r"))
    print("Starting grid search with the following config:")
    print(config)

    # From config, generate a list of experiments to run
    experiments = get_experiment_list(config)
    random.shuffle(experiments)

    job_queue = mp.SimpleQueue()
    done_queue = mp.SimpleQueue()

    for exper in experiments:
        job_queue.put(exper)

    print("Launching dispatcher with {} experiments and {} workers".format(len(experiments), args.num_workers))

    # Define worker fn to launch an experiment as a separate process.
    # Note: all tensors sent through a multiprocessing.Queue, will have their data moved into shared memory 
    # and will only send a handle to another process
    processes = []
    for _ in range(args.num_workers):
        p = mp.Process(target=worker, args=(args, job_queue, done_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Done")

if __name__ == '__main__':
    __spec__ = None
    args = parse_args()
    main(args)
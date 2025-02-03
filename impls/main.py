# MISC
import json
import sys
import os
import random
import time
from collections import defaultdict
import rootutils
import tqdm
import wandb
from typing import *

os.environ['MUJOCO_GL']='egl'
ROOT = rootutils.setup_root(search_from=__file__, cwd=True, pythonpath=True)

# Logging
import hydra
from omegaconf import DictConfig, OmegaConf
from absl import app, flags
from ml_collections import config_flags
from colorama import Fore, Style
from rich.pretty import pprint

# JAX & NNs
import jax
import numpy as np

#
from impls.agents import agents
from impls.utils.datasets import Dataset, GCDataset, HGCDataset
from impls.utils.env_utils import make_env_and_datasets
from impls.utils.evaluation import evaluate
from impls.utils.flax_utils import restore_agent, save_agent
from impls.utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

# Use flags to define jit & algo config
# TODO: Change to hydra completely
FLAGS = flags.FLAGS
flags.DEFINE_integer('disable_jit', 0, 'Whether to disable JIT compilation.')
config_flags.DEFINE_config_file('agent', 'impls/agents/gciql.py', lock_config=False)


@hydra.main(version_base=None, config_name="entry", config_path=str(ROOT) + "/hydra_sweep/")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False) # make modifications available
    cfg: Dict = OmegaConf.to_container(cfg, resolve=True) # prepare for saving config to file
    config = FLAGS.agent
    cfg['agent'] = config.to_dict()
    pprint(cfg)
    
    cfg: DictConfig = OmegaConf.create(cfg)
    exp_name = get_exp_name(cfg.seed)
    hydra.utils.call(cfg.logging)(name=exp_name)
    
    save_dir = os.path.join(cfg.save_dir, cfg.logging.group, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    # flag_dict = get_flag_dict()
    with open(os.path.join(str(ROOT), save_dir, 'flags.json'), 'w') as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f)
        
    env, train_dataset, val_dataset = make_env_and_datasets(cfg.env_name, frame_stack=config['frame_stack'])
    
    # Set up environment and dataset.
    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), config)

    # Initialize agent.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    example_batch = train_dataset.sample(1)
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        cfg.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent.
    if cfg.restore_path is not None:
        agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, cfg.train_steps + 1), smoothing=0.1, dynamic_ncols=True, colour='green', leave=True, position=0):
        # Update agent.
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.update(batch)

        # Log metrics.
        if i % cfg.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / cfg.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if i == 1 or i % cfg.eval_interval == 0:
            if cfg.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = cfg.eval_tasks if cfg.eval_tasks is not None else len(task_infos)
            for task_id in tqdm.trange(1, num_tasks + 1, leave=False, position=1, desc='Task', colour='blue'):
                task_name = task_infos[task_id - 1]['task_name']
                eval_info, trajs, cur_renders = evaluate(
                    agent=eval_agent,
                    env=env,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=cfg.eval_episodes,
                    num_video_episodes=cfg.video_episodes,
                    video_frame_skip=cfg.video_frame_skip,
                    eval_temperature=cfg.eval_temperature,
                    eval_gaussian=cfg.eval_gaussian,
                )
                renders.extend(cur_renders)
                metric_names = ['success']
                eval_metrics.update(
                    {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)
            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            if cfg.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=num_tasks)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % cfg.save_interval == 0:
            save_agent(agent, cfg.save_dir, i)

    train_logger.close()
    eval_logger.close()

def entry(argv):
    sys.argv = argv
    disable_jit = FLAGS.disable_jit
    try:
        if disable_jit:
            with jax.disable_jit():
                main()
        else:
            main()
            
    except KeyboardInterrupt:
        wandb.finish()
        print(f"{Fore.GREEN}{Style.BRIGHT}Finished!{Style.RESET_ALL}")



if __name__ == '__main__':
    app.run(entry)

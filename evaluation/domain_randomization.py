# -*- coding: utf-8 -*-

import gym
import pandas as pd
import os
import csv
import json
import sys
sys.path.insert(1, './../')

from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from natsort import natsorted

bests = [f'best_{i+1}' for i in range(10)]
for best in bests:
    print(f'evaluating {best}')
    models_dir = f'./../models/domain_randomization/top_score/{best}'
    model_name = 'best_model.zip'
    meta_name = 'meta.json'
    save_name = 'domain_randomized.csv'
    rewards_dir = f'./rewards/domain_randomization/{best}'
    tuned_method = 'grid'
    
    eval_ep = 50
    rewards = {}
    rewards['learning_rate'] = []
    rewards['gamma'] = []
    rewards['clip_range'] = []
    rewards['ent_coef'] = []
    rewards['n_steps'] = []
    rewards['gae_lambda'] = []
    rewards['randomization distribution'] = []
    rewards['policy'] = []
    rewards['network'] = []
    rewards['tuned method'] = []
    rewards['eval episodes'] = []
    rewards['source reward'] = []
    rewards['target reward'] = []    
    models = natsorted([f'{rootdir}/{model_name}' for rootdir,_,_ in os.walk(models_dir)][1:])
    metas = natsorted([f'{rootdir}/{meta_name}' for rootdir,_,_ in os.walk(models_dir)][1:])
    
    c=1
    for meta_dir, model_dir in zip(metas, models):
        env_source = gym.make("CustomHopper-source-v0")
        env_target = gym.make("CustomHopper-target-v0")
        with open(meta_dir) as met:
            meta = json.load(met)
        
        print(f'loading model {model_dir.split("/")[3]} with meta {meta_dir.split("/")[3]}')
        envs = {
            'source': env_source,
            'target': env_target
        }
        
        if meta['domain_randomization']:
            randomization_distribution = meta['chosen_domain']
        else:
            randomization_distribution = "not randomized"
            
        print(f'randomization distribution: {randomization_distribution}\n')
        source_target_reward = {}
        for env in envs.items():
            
            print(f'evaluating model ({c}/{len(models)} on {env[0]})')
            model = PPO.load(model_dir, env=env[1])
            mean_reward, std_reward = evaluate_policy(model,
                                                      model.get_env(),
                                                      n_eval_episodes=eval_ep)
            
            source_target_reward[env[0]] = mean_reward
            print(f'mean reward: {mean_reward}\n')
            env[1].close()


        rewards['learning_rate'].append(meta['learning_rate'])
        rewards['gamma'].append(meta['gamma'])
        rewards['clip_range'].append(meta['clip_range'])
        rewards['ent_coef'].append(meta['ent_coef'])
        rewards['n_steps'].append(meta['n_steps'])
        rewards['gae_lambda'].append(meta['gae_lambda'])
        rewards['randomization distribution'].append(randomization_distribution)
        rewards['eval episodes'].append(eval_ep)
        rewards['policy'].append(meta['policy'])
        rewards['network'].append(meta['obs'])
        rewards['tuned method'].append(tuned_method)
        rewards['source reward'].append(source_target_reward['source'])
        rewards['target reward'].append(source_target_reward['target'])
        print('______________________________________________')
        c+=1
    
    if not os.path.exists(rewards_dir):
        os.makedirs(rewards_dir)
    df = pd.DataFrame.from_dict(rewards)
    if save_name: 
        df.to_csv(os.path.join(rewards_dir, save_name), encoding='utf-8', mode='a')
        print(f"rewards saved in {os.path.join(rewards_dir, save_name)}")
                        

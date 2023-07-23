#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 17:43:25 2023

@author: ahmadrezafrh
"""

import numpy as np
import gym
import os
import warnings
import itertools
import json
import shutil

from env.custom_hopper import *
from wrappers import PixelObservationWrapper
from wrappers import GrayScaleWrapper
from wrappers import ResizeWrapper
from wrappers import PreprocessWrapper
from wrappers import FrameStack


from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

def custom_extractor(model, n_features):
    
    """
    It will create custom feature extractor for cnn networks.
    
    :param model: feature extractor class that we want to use
    :param n_features: number of output features
    :type model: class
    :type n_features: int
    :return: kwargs of the model
    """
    
    kwargs = dict(
        features_extractor_class=model,
        features_extractor_kwargs=dict(features_dim=n_features)
    )
        
    return kwargs
  
  
def ignore_warnings(ignore):
    """
    Ignores all warnings.
    
    :param ignore: if it's boolean we ignore all the warnings
    :type ignore: bool
    :return: none
    """
    
    if ignore:
        warnings.filterwarnings("ignore")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def check_path(path):
    """
    Check whether a path exists or not. If not it creates the path.
    
    :param path: path we want to check
    :type path: str
    :return: none
    """
    if not os.path.exists(path):
        os.makedirs(path)

def create_params(configue):
    """
    create parameters for training in each step. it returns a list of parameters list.
    
    :param configue: path we want to check
    :type configue: dict
    :return: list of all possible combination of parameters (used for grid search)
    
    NOTE: there is another approach that that we can generate a list of dictionaries that can be much more simpler.
    It also prevents the brute force approach that we use in the next steps.
    
    -----------------------------------------------
    The substitution method is;
    
    def create_params(**kwargs):
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in itertools.product(*vals):
            yield dict(zip(keys, instance))
    -----------------------------------------------
            
    """
    params = [configue['learning_rate'], configue['gamma'], configue['clip_range'], configue['ent_coef'], configue['n_steps'], configue['gae_lambda']]      
    if configue['obs']=='cnn':
        cnn_params = [configue['stacked'], configue['gray_scale'], configue['resize'], configue['resize_shape'], configue['n_frame_stacks'], configue['preprocess'], configue['policy_kwargs'], configue['smooth']]
        params.extend(cnn_params) 
    
    
    
    if configue['domain_randomization']:
        params.append(domain_randomization(configue['chosen_domain']))
    
    return list(itertools.product(*params))


def domain_randomization(domain_spaces):
    """
    We use this function to generate the domain spaces used for traininng. If it is 'grid'
    we try to develop a grid search on domain randomization.
    
    :param domain_spaces: list of distributions
    :type domain_spaces: list
    :return: none
    """
    
    assert domain_spaces != None
    if domain_spaces=='grid':
        lower_bounds = np.arange(1, 5, 0.5).tolist()
        range_lower_upper = np.arange(1, 4, 0.5).tolist()
        ranges = list(itertools.product(lower_bounds, range_lower_upper))
        
        domain_randomization_space = []
        for n in ranges:
            domain_randomization_space.append([[n[0], n[0]+n[1]]])
                
    else:
        domain_randomization_space = domain_spaces
        
    return domain_randomization_space

def create_env(meta, env=None):
    """
    We use this function to generate the domain spaces used for traininng. If it is 'grid'
    we try to develop a grid search on domain randomization.
    
    :param meta: dictionary of the model we want to train
    :type meta: dict
    :return: training environment
    """
    
    if meta['obs'] == 'mlp':
        if env:
            env = gym.make(env)
        else:
            env = gym.make(meta['env'])

        print('State space:', env.observation_space.shape)
        print('Action space:', env.action_space)
        print('Dynamics parameters:', env.get_parameters()) 

    if meta['obs'] == 'cnn':
        if env:
            env = gym.make(env)
        else:
            env = gym.make(meta['env'])
        env = PixelObservationWrapper(env)
        print('state observation shape', env.observation_space.shape)
        if meta['preprocess']:
            env = PreprocessWrapper(env)
            print('obervation shape after preprocessing:', env.observation_space.shape)
            
        if meta['gray_scale']:
            env = GrayScaleWrapper(env, smooth=meta['smooth'], preprocessed=meta['preprocess'], keep_dim=False)
            print('obervation shape after gray scaling:', env.observation_space.shape)
            if meta['resize']:
                env = ResizeWrapper(env, shape=meta['resize_shape'])
                print('obervation shape after resizing:', env.observation_space.shape)

        if meta['env']:
            env = FrameStack(env, num_stack=meta['n_frame_stacks'])
            print('obervation shape after stacking:', env.observation_space.shape)
    
    if meta['domain_randomization']:
        env.set_distributions(meta['chosen_domain'])
        
    return env

def create_meta(configue, hp, method='grid'):
    """
    Creates the metadata of the model that is trained. It will be used for other 
    operations.
    
    :param configue: configue file that is loaded from configues
    :param hp: hyperparameter for the current model
    :type configue: dict
    :type hp: dict
    :return: metadata file
    
    
    NOTE: there are obviously more better approaches.
    """
    
    
    if method=='grid':
        model_conf = {
            'env' : configue["env"],    
            'print' : configue["print"],    
            'alg' : configue["alg"],
            'obs' : configue["obs"],
            'ignore_warnings' : configue['ignore_warnings'],
            'custom_arch' : configue['custom_arch'],
            'time_steps' : configue['time_steps'],
            'policy' : configue['policy'],
            
            
            'domain_randomization' : configue['domain_randomization'],
            'n_distributions' : configue['n_distributions'] if configue['domain_randomization'] else None,
            'chosen_domain' : hp[-1] if configue['domain_randomization'] else None,
            
            'stacked' : hp[6] if configue["obs"]=='cnn' else None,
            'gray_scale' : hp[7] if configue["obs"]=='cnn' else None,
            'smooth' : hp[13] if configue["obs"]=='cnn' and hp[7] else None,
            'resize' : hp[8] if configue["obs"]=='cnn' else None,
            'resize_shape' : hp[9] if configue["obs"]=='cnn' and hp[8] else None,
            'preprocess' : hp[11] if configue["obs"]=='cnn' else None,
            'n_frame_stacks' : hp[10] if configue["obs"]=='cnn' else None,
            "policy_kwargs" : hp[12] if configue["obs"]=='cnn' else None,
            

            'n_steps' : hp[4],
            'learning_rate' : hp[0],
            'gamma' : hp[1],
            'clip_range' : hp[2],
            'ent_coef' : hp[3],
            'gae_lambda' : hp[5],
            
        }
    

    elif method=='optuna':
        model_conf = {
            'env' : configue["env"],    
            'print' : configue["print"],    
            'alg' : configue["alg"],
            'obs' : configue["obs"],
            'ignore_warnings' : configue['ignore_warnings'],
            'custom_arch' : configue['custom_arch'],
            'time_steps' : configue['time_steps'],
            'policy' : configue['policy'],
            
            
            
            'domain_randomization' : configue['domain_randomization'],
            'chosen_domain' : hp['chosen_domain'] if configue['domain_randomization'] else None,
            
            'stacked' : configue['stacked'] if configue["obs"]=='cnn' else None,
            'gray_scale' : configue['gray_scale'] if configue["obs"]=='cnn' else None,
            'smooth' : hp['smooth'] if configue["obs"]=='cnn' else None,
            'resize' : configue['resize'] if configue["obs"]=='cnn' else None,
            'resize_shape' : [hp['resize_shape'], hp['resize_shape']] if configue["obs"]=='cnn' and configue['resize'] else None,
            'preprocess' : hp['preprocess'] if configue["obs"]=='cnn' else None,
            'n_frame_stacks' : hp['n_frame_stacks'] if configue["obs"]=='cnn' else None,
            "policy_kwargs" : hp['policy_kwargs'] if configue["obs"]=='cnn' else None,
            
            
            'n_steps' : hp['n_steps'],
            'learning_rate' : hp['learning_rate'],
            'gamma' : hp['gamma'],
            'clip_range' : hp['clip_range'],
            'ent_coef' : configue['ent_coef'],
            'gae_lambda' : hp['gae_lambda'],
        }
        
        
    return model_conf
  
def save_meta(meta, model_path) :
    """
    saves the meta file for each model.
    
    :param meta: dictionary of the model we want to train
    :param model_path: dictionary of the model we want to train
    :type meta: 'dict'
    :type model_path: 'str'
    :return: none
    """
    
    with open(os.path.join(model_path, 'meta.json'), 'w') as model:
        json.dump(meta, model)
        
def create_model(env, meta, logs_dir, policy_kwargs):

    if meta['custom_arch'] and meta['obs'] == 'cnn':
        model = PPO(env=env,
                    policy=meta['policy'],
                    learning_rate=meta['learning_rate'],
                    gamma=meta['gamma'],
                    clip_range=meta['clip_range'],
                    ent_coef=meta['ent_coef'],
                    n_steps=meta['n_steps'],
                    gae_lambda=meta['gae_lambda'],
                    verbose=0,
                    device="cuda",
                    policy_kwargs=policy_kwargs[meta['policy_kwargs']],
                    tensorboard_log=logs_dir)
        
    else:
        model = PPO(env=env,
                    #clip_range_vf=0.2,
                    policy=meta['policy'],
                    learning_rate=meta['learning_rate'],
                    gamma=meta['gamma'],
                    clip_range=meta['clip_range'],
                    ent_coef=meta['ent_coef'],
                    n_steps=meta['n_steps'],
                    gae_lambda=meta['gae_lambda'],
                    verbose=0,
                    device="cuda",
                    tensorboard_log=logs_dir)
    
    
    return model
    
    
    
def create_model_path(models_path):
    """
    create path for the current model.
    
    :param models_path: path to all of the models
    :type models_path: 'str'
    :return: model path
    """
    paths = []
    
    for file in os.listdir(models_path):
        if os.path.isdir(os.path.join(models_path, file)):
            paths.append(int(file.split('_')[-1]))
    
    if len(paths) == 0:
        model_num = 1
        new_path = os.path.join(models_path, f'model_{model_num}')
    else:
        
        model_num = max(paths)
        model_path = os.path.join(models_path, f'model_{model_num}')
        count = 0
        for file in os.listdir(model_path):
            if os.path.isfile(os.path.join(model_path, file)):
                count += 1

        if count == 1:
            shutil.rmtree(model_path)
            new_path = model_path
            
        else:
            new_path = os.path.join(models_path, f'model_{model_num+1}')
                

    return new_path
    

    
def load_configue(path):
    """
    Loads the current configue to be used for training.
    
    :param path: path to the configue file
    :type path: 'str'
    :return: configue file
    """
    with open(path) as model:
        configue = json.load(model)
    
    return configue

    
    
def create_callback(env, callback_types, save_path, logs_dir, checkpoint_freq, eval_freq):
    """
    Creates a list of callbacks we want to use in our model
    
    :param env: env
    :param callback_types: we can have two types of callbacks ('eval', 'checkpoint')
    :param save_path: where to save callbacked models
    :param logs_dir: where to save logs
    :param checkpoint_freq: frequency of saving a model with 'checkpoint' callback
    :param eval_freq: frequency of checking and evaluating the model for best results
    
    :type env: object
    :type callback_types: list of strs
    :type save_path: str
    :type logs_dir: str
    :type checkpoint_freq: int
    :type eval_freq: int

    :return: list of callbacks
    """
    callbacks = []
    for cb in callback_types:
        if cb=='eval':
            callback = EvalCallback(env, best_model_save_path=save_path,
                                 log_path=logs_dir, eval_freq=eval_freq,
                                 deterministic=True, render=False)
        elif cb=='checkpoint':
            callback = CheckpointCallback(save_freq=checkpoint_freq, save_path=save_path)
    
        else:
            raise NameError("the callback types are not supported")
        
        callbacks.append(callback)
    
           
    return CallbackList(callbacks)



def print_model(meta):
    """
    print the current models set of parameters.
    
    :param meta: dictionary of current parameters
    :type meta: dictionary

    :return: none
    """
    
    if meta['print']:
        print(f'\npolicy = {meta["policy"]}')
        print(f'learning_rate = {meta["learning_rate"]}')
        print(f'gamma = {meta["gamma"]}')
        print(f'clip_range = {meta["clip_range"]}')
        print(f'ent_coef = {meta["ent_coef"]}')
        print(f'n_steps = {meta["n_steps"]}')
        print(f'gae_lambda = {meta["gae_lambda"]}')
        
        if meta['domain_randomization']:
            print(f'domain_space = {meta["chosen_domain"]}\n')

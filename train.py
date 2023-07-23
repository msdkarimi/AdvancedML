#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 15:20:14 2023

@author: ahmadrezafrh
"""

import os
from models import CNNBaseExtractor
from models import CNNSimple
from models import CNNMobileNet
from utils import create_model, create_model_path
from utils import create_meta, create_env, create_params
from utils import check_path, save_meta, load_configue
from utils import ignore_warnings, print_model
from utils import custom_extractor, create_callback

    
def main():
    

    
    configues_dir = './configues/cnn/grid'
    conf_name = 'single.json'
    cnn_architectures = {
    'base': custom_extractor(CNNBaseExtractor, 256),
    'simple': custom_extractor(CNNSimple, 128),
    'mobilenet': custom_extractor(CNNMobileNet, 512),
    }
    configue = load_configue(os.path.join(configues_dir, conf_name))
    ignore_warnings(configue['ignore_warnings'])
    logs_dir = configue['logs_dir']
    models_dir = configue['models_dir']
    callback_types = configue["callbacks"]
    checkpoint_freq = configue["checkpoint_freq"]
    eval_freq = configue["eval_freq"]
    callback_logs_dir = os.path.join(configue["logs_dir"], "results")
    models_dir = os.path.join(models_dir, conf_name[:-5])
    check_path(logs_dir)
    check_path(models_dir)
    hyperparams = create_params(configue)

    
    for c, hp in enumerate(hyperparams):
    
        print(f'training {c+1}/{len(hyperparams)}:\n')
        meta = create_meta(configue, hp, method='grid')
        env = create_env(meta)
        model = create_model(env ,meta, logs_dir=logs_dir, policy_kwargs=cnn_architectures)
        print_model(meta)
        
        
        model_path = create_model_path(models_dir)
        callbacks = create_callback(env=env, callback_types=callback_types, save_path=f"{model_path}", logs_dir=callback_logs_dir, checkpoint_freq=checkpoint_freq, eval_freq=eval_freq)
        check_path(model_path)
        save_meta(meta, model_path)
        
        
        model.learn(total_timesteps=meta['time_steps'], tb_log_name=f"{model_path}", callback=callbacks, progress_bar=True)
        model.save(os.path.join(model_path, f"{meta['time_steps']}"))
        if configue['domain_randomization']:
            print('Dynamics parameters:', env.get_parameters())
            
        del model
        print('__________________________________________________________________________________\n')




            
if __name__ == '__main__':
    main()

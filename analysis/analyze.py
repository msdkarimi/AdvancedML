#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 20:18:14 2023

@author: ahmadrezafrh
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

best_n_rows = 20
scores_dir = './rewards'
tables_dir = './results/tables'



scores_name = 'optuna_params.csv'
save_name = './optuna_params.txt'


scores = pd.read_csv(os.path.join(scores_dir, scores_name))
sorted_scores = pd.DataFrame(scores.sort_values('target reward', ascending=False).to_numpy(), 
                             index=scores.index,
                             columns=scores.columns).drop(['Unnamed: 0'], axis=1)

sorted_scores.style.hide_index()

with open(os.path.join(tables_dir,save_name), 'a') as f:
    f.write(f'\t\t\t\t\tTOP {best_n_rows} BEST RESULTS FOR OPTUNA (randomized search) - TARGET\n')
    f.write('\t\t\t\t\t___________________________________________________________\n\n\n')
    f.write(sorted_scores.loc[0:best_n_rows].to_string(index=False))




#================================================================================================
#best results on source
#================================================================================================

    


scores_name = 'optuna_params.csv'
save_name = './optuna_params.txt'


scores = pd.read_csv(os.path.join(scores_dir, scores_name))
sorted_scores = pd.DataFrame(scores.sort_values('source reward', ascending=False).to_numpy(), 
                             index=scores.index,
                             columns=scores.columns).drop(['Unnamed: 0'], axis=1)

sorted_scores.style.hide_index()

with open(os.path.join(tables_dir,save_name), 'a') as f:
    f.write('\n\n\n')
    f.write(f'\t\t\t\t\tTOP {best_n_rows} BEST RESULTS FOR OPTUNA (randomized search) - SOURCE\n')
    f.write('\t\t\t\t\t___________________________________________________________\n\n\n')
    f.write(sorted_scores.loc[0:best_n_rows].to_string(index=False))



#================================================================================================
# GRID ANALYSIS
#================================================================================================


scores_dir = './rewards'
scores_name = 'grid_search.csv'
tables_dir = './results/tables'
save_name = './grid_search.txt'

scores = pd.read_csv(os.path.join(scores_dir, scores_name))
# scores = scores[(scores['gamma']==0.99) & (scores['gae_lambda']==0.95)]
sorted_scores = pd.DataFrame(scores.sort_values('target reward', ascending=False).to_numpy(), 
                             index=scores.index,
                             columns=scores.columns).drop(['Unnamed: 0'], axis=1)
sorted_scores.style.hide_index()
for lr in scores.learning_rate.unique():
    for cr in scores.clip_range.unique():
        with open(os.path.join(tables_dir,save_name), 'a') as f:
            f.write(f'\t\t\t\t\t\n')
            f.write('\t\t\t\t\t_____________________________________________________\n\n\n')
            f.write(sorted_scores.loc[0:best_n_rows].to_string(index=False))
    



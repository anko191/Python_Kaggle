import numpy as np
import pandas as pd

# n_seed, n_features,CV_name,n_folds, CV_score, LB_score, 一言コメント
commits_total = pd.DataFrame(columns = ['n_seed',
                                'n_features', 'CV_name',
                                'n_folds', 'CV_score',
                                'LB_score', 'comment'])

n = 0
commits_total.loc[n,'n_seed'] = 0
commits_total.loc[n,'n_features'] = 0
commits_total.loc[n,'CV_name'] = '?'
commits_total.loc[n,'n_folds'] = 0
commits_total.loc[n,'CV_score'] = 0.00
commits_total.loc[n,'LB_score'] = 0.01
commits_total.loc[n, 'comment'] = 'nyaaaa'

print(commits_total)
commits_total.sort_values(by = ['CV_score'], ascending = True)

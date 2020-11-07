# my notebook : https://www.kaggle.com/anko191/elbow-method
#
# ref : https://en.wikipedia.org/wiki/Elbow_method_(clustering)
#
# Search the best parameter in k-means' number of clusters
#
# gene and cell
# ref https://qiita.com/deaikei/items/11a10fde5bb47a2cf2c2
# クラスタ内誤差平方和（SSE）を計測
import warnings
warnings.simplefilter('ignore', FutureWarning)

import numpy as np
import pandas as pd

from tqdm.notebook import tqdm

# + cluster
from sklearn.cluster import KMeans

# plotly
import plotly.express as px

# k_means ++
# エルボー法で最適な地点数を探す

features_g = [name for name in train.columns if name.startswith('some')]

some_SSE = []
for i in tqdm(range(5,250,5)): # これもお好みで
    k_train = train[features_g].copy()
    k_test = test[features_g].copy()
    k_data = pd.concat([k_train, k_test], axis = 0)
    # kmeans++ で初期化, k-means++で初期クラスタを設定
    # n_init = 10
    # 異なるセントロイドを用いたアルゴリズムの実行回数，最大イテレーション数， 相対許容誤差
    # max_iter = 300, tol = 1e-04...
    kmeans = KMeans(n_clusters = i, init='k-means++', n_init=10, max_iter=300, tol=1e-04,random_state = 77)
    kmeans.fit(k_data)
    some_SSE.append(kmeans.inertia_)

# .inertia_  クラスタ内誤差平方和（SSE） をそれぞれ計算し、追加
# 図示して、肘となっているところを探す

df_some = pd.DataFrame({"some_SSE":some_SSE,
                    'num':list(range(5,250,5))})

fig = px.line(df_some, x = 'num', y = "some_SSE",
             title = "some's SSE of some clusters")
fig.show()

# 適当に名前を変える

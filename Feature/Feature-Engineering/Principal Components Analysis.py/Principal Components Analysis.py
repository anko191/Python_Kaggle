# ref:https://towardsdatascience.com/pca-clearly-explained-how-when-why-to-use-it-and-feature-importance-a-guide-in-python-7c274582c37e
# 本来は次元を減らすために使われたりするが、
# ノイズを減らしたり、データ圧縮にも使える。
# PCA can be also used for denoising and data compression.
# PCA technique is particularly useful in processing data
# where multi-colinearity exists between the features/variables.
#
# Let X be a matrix containing the original data with shape [n_samples, n_features] .

# https://www.haya-programming.com/entry/2018/03/19/172315

# 正規分布 (Gaussian)変換後にすること。QuantileTransformerを使ったあとに

# 変換する前のもの


from sklearn.decomposition import PCA

def _PCA(train,test,kind = None, n_components = 50):
    features = [col for col in train.columns if col.startswith(kind)]

    data = pd.concat([pd.DataFrame(train[features]), pd.DataFrame(test[features])])
    pca = PCA(n_components = n_components, random_state = 77)
    pca_data = pca.fit_transform(data[features])

    pca_train = pca_data[:train.shape[0]]
    pca_test = pca_data[train.shape[0]:]

    trf = pd.DataFrame(pca_train, columns = [f'{kind}-{i}-PCA' for i in range(n_components)])
    ttf = pd.DataFrame(pca_test, columns = [f'{kind}-{i}-PCA' for i in range(n_components)])

    ptr = pd.concat((train, trf), axis = 1)
    ptt = pd.concat((test, ttf), axis = 1)
    return ptr, ptt

# 目的のもののみを考えたいとき、
train, test = _PCA(train, test, kind, n_components)
# kind を指定すればよい なければ None

# make cluster of ...

# ref : https://qiita.com/NoriakiOshita/items/cbd46d907d196efe64a3
# クラスタの中心を乱数で割り当てているので、ちょっとバグる時がある

# k_means ++ としたほうがいい
# 重み付き確率分布を取り入れる D(x)^2 / (\sum(D(x)^2))

from sklearn.cluster import KMeans


def fe_cluster(train, test, n_clusters_g = 35, n_clusters_c = 5, SEED = 77):
    print("Let's cluster ! ")
    # cluster

    features_g = [name for name in train.columns if name.startswith('g-')]
    features_c = [name for name in train.columns if name.startswith('c-')]
    def create_cluster(train, test, features, kind = 'g', n_clusters = n_clusters_g):
        con_train = train[features].copy()
        con_test = test[features].copy()
        data = pd.concat([con_train, con_test], axis = 0)
        # kmeans++ で初期化, k-means++で初期クラスタを設定
        # n_init = 10
        # 異なるセントロイドを用いたアルゴリズムの実行回数，最大イテレーション数， 相対許容誤差
        # max_iter = 300, tol = 1e-04...
        kmeans = KMeans(n_clusters = n_clusters, init='k-means++', n_init=10, max_iter=300, tol=1e-04,random_state = SEED).fit(data)
        # kmeans = KMeans(n_clusters = n_clusters, random_state = SEED).fit(data)
        train[f'clusters_{kind}'] = kmeans.labels_[:train.shape[0]]
        test[f'clusters_{kind}'] = kmeans.labels_[train.shape[0]:]

        train = pd.get_dummies(train, columns = [f'clusters_{kind}'])
        test = pd.get_dummies(test, columns = [f'clusters_{kind}'])

        return train, test

    train, test = create_cluster(train, test, features_g, kind = 'g', n_clusters = n_clusters_g)
    train, test = create_cluster(train, test, features_c, kind = 'c', n_clusters = n_clusters_c)
    return train, test

clusterd_train, clustered_test = fe_cluster(r_train,r_test)


# 最適なクラスター数を見つける
# https://qiita.com/deaikei/items/11a10fde5bb47a2cf2c2

# エルボー法
# クラスタごとのSSE値をプロットした図
# SSE クラスタ内誤差平方和を見る !
# kmeans.inertia_ で与えられる。
# 小さいほど、上手く行っているモデル！
# を使う

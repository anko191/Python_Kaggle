# sklearn.preprocessing.QuantileTransformer
# values should be withing a range between 0 and 1,
# and the distribution of values should be uniform.

# https://tsumit.hatenablog.com/entry/2020/06/20/044835
# RankGauss 問者らしい
# 数値変換をガウス分布に変換して、
# 対象となる変数の値を順位付けし、その順位を-1 ~ 1 の範囲にスケーリングする

# QuantileTransformer が用意されている

from sklearn.preprocessing import QuantileTransformer

# uniform 一様分布
# normal 正規分布
# に従うように変換思案す。
# transfroms the features to follow a uniform or a normal distribution
#
# normal だと 分布が ガウス分布
# uniform　一様分布
# RankGaussなら normalとする


def QTf(train, test, features, n_quantiles = 100, seed = 77, output_distribution = 'normal'):
    for col in features:
        transformer = QuantileTransformer(n_quantiles = 100, random_state = seed, output_distribution = output_distribution)
        trainlen = len(train[col].values)
        testlen = len(test[col].values)
        reshaped_train = train[col].values.reshape(trainlen, 1)
        # どうして縦に併せるんだ？
        transformer.fit(reshaped_train)
        # どうして fit ?
        # Compute the quantiles used for transforming.
        # Xndarray or sparse matrix, shape (n_samples, n_features)

        train[col] = transformer.transform(reshaped_train).reshape(1, trainlen)[0]
        # なにこれ
        test[col] = transformer.transform(test[col].values.reshape(testlen, 1)).reshape(1,testlen)[0]

    return train, test

# - - - - - - 試す - - - - - - -
from sklearn.preprocessing import QuantileTransfromer

qt = QuantileTransformer(random_state = 0, output_distribution = 'normal')
# 変換する列を指定
num_cols = ['nanka', 'wha', 'nya']
# num_colsに対して変換用の変位地を計算
qt.fit(df[num_cols])

# RankGaussによる変換を行う
df[num_cols] = qt.transform(df[num_cols])

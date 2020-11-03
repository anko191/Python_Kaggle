# sklearn.preprocessing.QuantileTransformer
# values should be withing a range between 0 and 1,
# and the distribution of values should be uniform.

from sklearn.preprocessing import QuantileTransformer

# uniform 一様分布
# normal 正規分布
# に従うように変換思案す。
# transfroms the features to follow a uniform or a normal distribution

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
        

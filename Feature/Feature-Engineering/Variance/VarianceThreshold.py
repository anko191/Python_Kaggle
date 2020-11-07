# https://scikit-learn.org/stable/modules/feature_selection.html#variance-threshold
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html#sklearn.feature_selection.VarianceThreshold

from sklearn.feature_selection import VarianceThreshold

# SelectKBestでは、分散が0の時にエラーを吐いてしまう。その対処法として
# VarianceThresholdを使う

THRESHOLD = ? # 定数 どのくらい以下の特徴量を削除するか

def VT(train, test, THRESHOLD = 0.80):
    vt = VarianceThreshold(THRESHOLD)
    data = train.append(test)
    # axis = 0 に追加
    # 変形する位置は適切に変える
    data_transformed = vt.fit_transform(data.iloc[:,4:])
    trf = data_transformed[:train.shape[0]]
    tef = data_transformed[-test.shape[0]:]

    train = pd.DataFrame(train[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].values.reshape(-1,4), columns = ['sig_id', 'cp_type', 'cp_time', 'cp_dose'])
    train = pd.concat([train, pd.DataFrame(trf)], axis = 1)

    test = pd.DataFrame(test[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].values.reshape(-1,4), columns = ['sig_id', 'cp_type', 'cp_time', 'cp_dose'])
    test = pd.concat([test, pd.DataFrame(tef)], axis = 1)

    print(train.shape,test.shape)
    return train,test

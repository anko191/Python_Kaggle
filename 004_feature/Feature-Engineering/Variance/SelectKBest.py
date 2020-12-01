# すべてを特徴量として使うのではなく、
# 上位K個を返す。
# 単変量統計
# ref : https://qiita.com/rockhopper/items/a68ceb3248f2b3a41c89


from sklearn.feature_selection import SelectKBest, f_classif
def SelectKB(train,test,train_target, ok = True):
    if not ok:
        return train,test

    print(' - SelectKB - ')
    # target_cols = 何か求めたいもの
    dic = defaultdict(int)
    # ここのk
    selector = SelectKBest(f_classif, k = K_1)
    # 2値分類なら、ここは1つのみ
    for columns in tqdm(target_cols.tolist()):
        X_new = selector.fit_transform(train, train_target[columns])
        selected_features = pd.DataFrame(selector.inverse_transform(X_new),
                                        index = train.index,
                                        columns = train.columns)

        selected_columns = selected_features.columns[selected_features.var() != 0]
        for c in selected_columns.to_list():
            dic[c] += 1
    # 上位から何個か、沢山選ばれたものをすべて選択している。
    # 例えば、上位から600個も可能である
    # HighCol = HighCol[:min(len(Highcol), 600)]
    Highall = sorted(dic.items(), key = lambda x:x[1], reverse=True)
    print('len:',len(Highall))
    HighCol = []
    for i in tqdm(range(len(Highall))):
        HighCol.append(Highall[i][0])
    ttrain = train[HighCol]
    ttest = test[HighCol]
    return ttrain, ttest

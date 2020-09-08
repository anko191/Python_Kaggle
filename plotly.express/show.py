import plotly.express as px
dcpd = df.groupby(['---', '--'])['---'].count().reset_index()
dcpd.columns = ['--', '--', '--']

# - - - - 

棒グラフ
orientation : v -> 縦に積む, h -> 横に積む
barmode : group -> グループに, 重ねて表示とかもある stack, overlay, relative

fig = px.bar(
    dcpd, 
    x='cp_dose', 
    y="count", 
    color = 'dataset',
    barmode='group',
    orientation='v', 
    title='---- train/test counts', 
    width=600,
    height=600
)

fig.show()


# - - - 

なんか円グラフが積み重なったやつが出来る

dall = df[df['dataset'] == 'train']
dall = dall.groupby(['--', '--','---'])['---'].count().reset_index()
dall.columns = ['--', '--', '--','---']
fig = px.sunburst(
dall,
path = ['ここ', 'に', '入るもの'],
values = 'count',
title = 'Sunburst char for all a',
width = 600,
height = 600)

fig.show()

# - - - - - -

円グラフ %表示

data = train_target.drop(['---'], axis=1).astype(bool).sum(axis=1).reset_index()
data.columns = ['row', 'count']
data = data.groupby(['count'])['row'].count().reset_index()
fig = px.pie(
    data, 
    values=100 * data['row']/len(train_target), 
    names="count", 
    title='Number of activations in targets for every sample (Percent)', 
    width=800, 
    height=500
)
fig.show()

# - - - - 

散布図 カラーってなに サイズもなに

fig = px.scatter(
    analysis, 
    x=col_df.iloc[index]['trainの1_column'], 
    y=col_df.iloc[index]['trainの2_column'], 
    color="color", 
    size='size', 
    height=800,
    title='Scatter plot for ' + col_df.iloc[index]['column']
)

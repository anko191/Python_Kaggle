import pandas as pd

shaka_0130data = pd.read_csv(r'D:\Pycharm\API_Ex\2021_1_30_shakakinTV.csv', index_col = 0)
shaka_0222data = pd.read_csv(r'D:\Pycharm\API_Ex\20210222__shakakinTV.csv', index_col = 0)

shaka_data_raw = pd.concat([shaka_0130data, shaka_0222data])
# (135, 11)
print(shaka_data_raw.columns)
shaka_data = shaka_data_raw.drop_duplicates(ignore_index=True)
shaka_data.to_csv(r'D:\Pycharm\API_Ex\shaka_data.csv')
# print(shaka_data.iloc[0])
shaka_data = shaka_data[~shaka_data.duplicated(subset = 'title')].sort_values(['created_at'], ascending=False).reset_index(drop = True)
shaka_data.to_csv(r'D:\Pycharm\API_Ex\shaka_data.csv')
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
plt.plot(shaka_data['view_count'][::-1])
plt.tight_layout()
plt.savefig('shaka_0102.png')
plt.show()


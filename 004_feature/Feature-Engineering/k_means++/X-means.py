# Automation decision of numbers of k in k-means++
# ref - https://qiita.com/simonritchie/items/e48a19128a08244fcc83#%E3%83%9C%E3%83%AD%E3%83%8E%E3%82%A4%E5%9B%B3
#
import pyclustering

from pyclustering.cluster import xmeans


initializer = xmeans.kmeans_plusplus_initializer(data = X, amount_centers = 2)
# start k = 2
# renew k's value

initial_centers = initializer.initialize()
xm = xmeans.xmeans(data = X, initial_centers = initial_centers)
xm.process()

# There is function for plotting clusters in library.
clusters = xm.get_clusters()
pyclustering.utils.draw_clusters(data=X, clusters= clusters)

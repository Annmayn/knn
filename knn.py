# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#custom Input
df = pd.DataFrame()
arr=np.array([[1,1],[1,2],[1,3],[4,1],[5,1],[0,0],[6,1],[5,5],[5,6],[6,5]])
clusterNum = [0, 0, 0, 2, 2, 0, 2, 1, 1, 1]
point = [0,1]
#end of custom input


def findClusterId(point, df, k):
    arr = np.concatenate( (np.array(df['x']).reshape(df['x'].size,1), np.array(df['y']).reshape(df['y'].size,1)), axis=1)
    dist = np.sqrt(np.sum( (arr-point)**2, axis=1))
    df['distance'] = dist
    df.sort_values(by='distance', inplace=True)
    df = df.head(k)
    return (df['cluster'].mode())[0]

def runKNN(test_point, array, cluster, k=3):
    df['point'] = array.tolist()
    df['x'] = array[:,0]
    df['y'] = array[:,1]
    df['cluster'] = clusterNum
    return findClusterId(test_point, df, k)

clusterId = runKNN(test_point=point, array=arr, cluster=clusterNum, k=3)

plt.scatter(df['x'], df['y'], c=df['cluster'])
plt.scatter(point[0], point[1], s=100, marker='*', c=clusterId)
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seaborn import xkcd_rgb
import copy
import os


def generate_center(df, k, seed=None):
    # 1.
    if seed is not None:
        np.random.seed(seed)
    idx_0 = np.random.randint(0, df.shape[0])
    centroids = {
        0: [df['x'][idx_0], df['y'][idx_0]]
    }
    # 2.
    if k > 1:
        for i in range(1, k):
            centroids = generate_probability(df, centroids)
    return centroids


def generate_probability(df, centroids):
    df = assignment(df, centroids, None)
    distance = df['closest'].map(lambda x: df['distance_from_{}'.format(x)])
    acc_distance = [sum(distance[0:_+1]) / sum(distance) for _ in range(df.shape[0])]
    p = np.random.random()
    idx = np.argwhere(np.array(acc_distance) > p).ravel().tolist()[0]
    centroids[max(centroids.keys())+1] = [df['x'][idx], df['y'][idx]]
    return centroids


def generate_colmap(k):
    color_list = list(xkcd_rgb.values())
    colmap = {
        i: color_list[i]
        for i in range(k)
    }
    return colmap


def assignment(df, centroids, colmap=None):
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )
    distance_from_centroid_id = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, distance_from_centroid_id].idxmin(axis=1)  # length: n; element: 'distance_from_i'
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))  # length: n; element: i
    if colmap is not None:
        df['color'] = df['closest'].map(lambda x: colmap[x])
    return df


def update(df, centroids):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return centroids


def main(k=3, seed=None):
    """ K-Means++ Main function """
    save_path = "./result/example_{}_classes".format(k)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # step 0: generate source data
    df = pd.DataFrame({
        'x': [12, 20, 28, 18, 10, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 23],
        'y': [39, 36, 30, 52, 54, 20, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 77]
    })

    # step 1: generate center
    centroids = generate_center(df, k, seed)

    # step 2: assign centroid for each source data
    colmap = generate_colmap(k)
    df = assignment(df, centroids, colmap)

    plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
    for i in centroids.keys():
        plt.scatter(*centroids[i], color=colmap[i], linewidths=6)
    plt.xlim(0, 80)
    plt.ylim(0, 80)
    plt.savefig(save_path + "/origin.jpg")

    for i in range(10):
        plt.close()
        # closest_centroids = df['closest'].copy(deep=True)
        closest_centroids = copy.deepcopy(df['closest'])
        print(id(closest_centroids))
        centroids = update(df, centroids)

        plt.figure(i)
        plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
        for k in centroids.keys():
            plt.scatter(*centroids[k], color=colmap[k], linewidths=6)
        plt.xlim(0, 80)
        plt.ylim(0, 80)
        plt.savefig(save_path + "/{}.jpg".format(i))

        df = assignment(df, centroids, colmap)

        if closest_centroids.equals(df['closest']):
            break


if __name__ == '__main__':
    main(k=4)

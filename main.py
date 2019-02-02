# coding: utf-8
from utils.data_processing import Dataset
from utils.semi_kmedoids import Semi_KMedoids
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils.plot_center import plot_center

def get_best_k(k_range, data, seeds):
    Score = {}
    Inertia = {}
    for k in k_range:
        clf = Semi_KMedoids(k)
        clf.fit(data, seeds=seeds)
        Score[k] = clf._silhouette_score
        Inertia[k] = clf._inertia

    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.plot(k_range, Score.values(), 'o-')
    plt.savefig('fig/silhouette_score.eps')
    plt.show()

    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.plot(k_range, Inertia.values(), 'o-')
    plt.savefig('fig/inertia.eps')

    return max(Score, key=Score.get)

def get_medoids(k, data, seeds):
    clf = Semi_KMedoids(k=k)
    clf.fit(data, seeds=seeds)

    medoid_users = dataset.users[clf._medoid_ids]
    plot_center(medoid_users, 'fig/center.pdf')

    with open('data/center.json', 'w') as wf:
        for m_user in medoid_users:
            wf.write(json.dumps(m_user) + '\n')

def kmeans(k_range, data):
    from sklearn.metrics import silhouette_score
    Scores = {}
    for k in k_range:
        estimator = KMeans(n_clusters=k)
        estimator.fit(data)
        Scores[k] = silhouette_score(data, estimator.labels_, metric='euclidean')

    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.plot(k_range, Scores.values(), 'o-')
    plt.savefig('fig/silhouette_score.eps')
    return max(Scores, key=Scores.get)


if __name__ == '__main__':
    dataset = Dataset(data_rpath='data/collection.json',
                      daily_representation_path='data/daily_representation',
                      daily_seeds_rpath='data/daily_seeds.json',
                      daily_distance_rpath='data/daily_distance',
                      )

    data = np.hstack((dataset.rep['daily'], dataset.rep['weekly']))
    best_k = get_best_k(range(3, 9), data, dataset.seeds)
    get_medoids(3, data, dataset.seeds)
    #kmeans(range(4, 20), seq)
# coding: utf-8
from utils.data_processing import Dataset
from utils.semi_kmedoids import Semi_KMedoids
import json
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
def get_best_k(k_range, seq):
    Scores = {}
    for k in k_range:
        clf = Semi_KMedoids(k)
        clf.fit(dataset.rep[seq], seeds=dataset.seeds[seq])
        Scores[k] = clf.silhouette_score

    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.plot(k_range, Scores.values(), 'o-')
    plt.savefig('fig/silhouette_score.eps')
    return max(Scores, key=Scores.get)

def get_medoids(k, seq):
    clf = Semi_KMedoids(k=k)
    clf.fit(dataset.rep[seq], seeds=dataset.seeds[seq])

    medoid_users = dataset.users[clf._medoid_ids]

    with open('data/' + seq + '_center.json', 'w') as wf:
        for m_user in medoid_users:
            wf.write(json.dumps(m_user) + '\n')

def kmeans(k_range, seq):
    from sklearn.metrics import silhouette_score
    Scores = {}
    for k in k_range:
        estimator = KMeans(n_clusters=k)
        estimator.fit(dataset.rep[seq])
        Scores[k] = silhouette_score(dataset.rep[seq], estimator.labels_, metric='euclidean')

    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.plot(k_range, Scores.values(), 'o-')
    plt.savefig('fig/silhouette_score.eps')
    return max(Scores, key=Scores.get)


if __name__ == '__main__':
    seq = sys.argv[1]
    dataset = Dataset(data_rpath='data/encoded.json',
                      weekday_seeds_rpath='data/weekday_seeds.json',
                      weekend_seeds_rpath='data/weekend_seeds.json',
                      weekday_distance_rpath='data/distance_weekdays',
                      weekend_distance_rpath='data/distance_weekends',
                      )

    #best_k = get_best_k(range(6, 15), seq)
    #get_medoids(best_k, seq)
    kmeans(range(4, 20), seq)
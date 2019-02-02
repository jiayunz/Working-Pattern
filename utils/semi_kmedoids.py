# coding: utf-8
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from sklearn import metrics
import random
from scipy.spatial import distance

class Semi_KMedoids:
    def __init__(self, k):
        self._k = k
        self._inertia = 0

    def _get_distance_matrix(self, data):
        return distance.cdist(data, data, metric='euclidean')

    # 从labeled数据中随机取，剩余的cluster从unlabeled数据中按kmeans++方法取
    def _init_medoids(self, distance, seeds):
        medoids = []
        for i in xrange(self._k):
            if i < len(seeds):
                medoids.append(random.sample(seeds[i], 1)[0])
            else:
                rest = list(set(range(distance.shape[0])) - set(medoids))
                # axis=1 按行取最小/相加
                shortest_distance_to_medoids = np.min(np.square(distance[rest, :][:, medoids]), axis=1).reshape(-1, 1)
                sum_distance_to_medoids = np.sum(np.square(distance[rest, :][:, medoids]), axis=1).reshape(-1, 1)
                probs = 1. * shortest_distance_to_medoids / sum_distance_to_medoids
                s = random.random()
                for i in xrange(len(probs)):
                    if s < sum(probs[:i + 1]):
                        medoids.append(rest[i])
                        break
        return medoids

    def _update_medoids_with_clusters(self, distance, clusters):
        new_medoids = []
        # 和簇内所有点的距离之和最小的点作为新的medoids
        for c in clusters:
            new_medoid_idx = np.argmin(np.sum(distance[c, :][:, c], axis=0))
            new_medoids.append(c[new_medoid_idx])

        return new_medoids

    def _update_clusters_with_medoids(self, distance, medoids, seeds):
        rest = np.setdiff1d(range(distance.shape[0]), medoids)
        # distance[M,:]是中心点与每个点之间的距离，设axis=0表示对每一列在M行中的大小比较
        c = np.argmin(distance[medoids, :], axis=0)[rest]
        clusters = [[m] for m in medoids]
        for i in xrange(self._k):
            clusters[i] = np.concatenate([clusters[i], rest[np.where(c == i)[0]]])

        return clusters

    def _constrained_update_clusters_with_medoids(self, distance, medoids, seeds):
        # TODO: seeds必须在原始cluster中
        # distance[M,:]是中心点与每个点之间的距离，设axis=0表示对每一列在M行中的大小比较
        rest = np.setdiff1d(np.setdiff1d(range(distance.shape[0]), np.concatenate(seeds)), medoids)
        c = np.argmin(distance[medoids, :], axis=0)[rest]
        clusters = [[m] for m in medoids]
        for i in xrange(self._k):
            if i < len(seeds):
                clusters[i] = np.concatenate([clusters[i], np.setdiff1d(seeds[i], medoids), rest[np.where(c == i)[0]]])
            else:
                clusters[i] = np.concatenate([clusters[i], rest[np.where(c == i)[0]]])

        return clusters

    # Parameters:
    # 1. X - numpy.array, data
    # 2. pre_cls: numpy.array, pre-defined clusters
    # 3. max_iter: int, Maximum Iterations
    def fit(self, X, seeds, max_iter=300):
        # determine dimensions of distance matrix
        if X.shape[0] < self._k:
            raise Exception('k is too large.')
        elif len(seeds) > self._k:
            raise Exception('k is too small.')

        distance = self._get_distance_matrix(X)

        # initialize medoids with labeled data (and unlabeled data)
        self._medoid_ids = self._init_medoids(distance, seeds)

        for t in xrange(max_iter):
            clusters = self._update_clusters_with_medoids(distance, self._medoid_ids, seeds)
            new_medoids = self._update_medoids_with_clusters(distance, clusters)
            if np.array_equal(self._medoid_ids, new_medoids):
                break
            self._medoid_ids = np.copy(new_medoids)
        else:
            # final update of cluster memberships
            clusters = self._update_clusters_with_medoids(distance, self._medoid_ids, seeds)

        # compute inertia: sum of squared distances of samples to their closest cluster center.
        self._labels = np.zeros([X.shape[0]])
        for i, c in enumerate(clusters):
            self._inertia += np.sum(distance[self._medoid_ids[i]][c])
            for j in c:
                self._labels[j] = i

        self._silhouette_score = metrics.silhouette_score(X, self._labels)

        self._medoids = X[self._medoid_ids, :]

if __name__ == '__main__':
    data = np.array([[1., 2., 3., 4],
                     [1., 2., 3., 4.],
                     [2., 3., 4., 1.],
                     [1., 3., 5., 4.],
                     [3., 5., 4., 1.],
                     [5., 4., 1., 3.],
                     [3., 5., 4., 1.]])
    clf = Semi_KMedoids(k=3)
    clf.fit(data, seeds=[[1, 2],[4, 6]])
    print clf._medoids
    print clf._medoid_ids

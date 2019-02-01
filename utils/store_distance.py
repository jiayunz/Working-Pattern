# coding: utf-8
from data_processing import Dataset
from BiKMedoids import BiKMedoids
import json
import numpy as np

if __name__ == '__main__':

    print 'loading dataset...'
    dataset = Dataset(rpath='data/collection.json')
    clf = BiKMedoids(k=8)
    data_weekdays = dataset.data['weekdays']
    data_weekends = dataset.data['weekends']

    print 'generating ground distance...'
    ground_distance = clf._ground_distance(24)

    print 'generating weekdays distance...'
    distance_weekdays = clf._get_distance_matrix(data_weekdays, ground_distance)
    with open('../data/distance_weekdays', 'w') as wf:
        json.dump(distance_weekdays.tolist(), wf)

    print 'generating weekends distance...'
    distance_weekends = clf._get_similarity_matrix(data_weekends, ground_distance)
    with open('../data/distance_weekends', 'w') as wf:
        json.dump(distance_weekends.tolist(), wf)
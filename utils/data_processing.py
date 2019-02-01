# coding: utf-8
import json
import numpy as np
from datetime import datetime
from wasserstein_distance import *

class Dataset:
    def __init__(self, data_rpath, daily_seeds_rpath=None, weekly_seeds_rpath=None, daily_distance_rpath=None, weekly_distance_rpath=None):
        self.data_rpath = data_rpath
        self.daily_seeds_rpath = daily_seeds_rpath
        self.weekly_seeds_rpath = weekly_seeds_rpath
        self.daily_distance_rpath = daily_distance_rpath
        self.weekly_distance_rpath = weekly_distance_rpath
        self.load_seeds()
        self.load_data()
        self.load_distance()

    def get_commit_profile(self, commits):
        commit_profile = {
            'daily': {},
            'weekly': {}
        }

        for i in range(24): # hour of day
            commit_profile['daily'][i] = 0
        for i in range(1, 8):
            commit_profile['weekly'][i] = 0

        for c in commits:
            time = c['commit']['author']['date'][:19]
            time = datetime.strptime(time, '%Y-%m-%dT%H:%M:%S')
            commit_profile['daily'][time.hour] += 1. / len(commits)
            commit_profile['weekly'][time.isoweekday()] += 1. / len(commits)

        return commit_profile

    def load_data(self):
        self.users = []
        self.rep = {'daily': [], 'weekly': []}
        self.data = {'daily': [], 'weekly': []}

        with open(self.data_rpath, 'r') as rf:
            for i, line in enumerate(rf.readlines()):
                user = json.loads(line.strip())

                if str(user['id']) in self.daily_seed_dict:
                    self.seeds['daily'][self.daily_seed_dict[str(user['id'])]].append(i)

                if str(user['id']) in self.weekly_seed_dict:
                    self.seeds['weekly'][self.weekly_seed_dict[str(user['id'])]].append(i)

                if 'commit_profile' not in user:
                    user['commit_profile'] = self.get_commit_profile(user['commits_list'])
                if 'daily_rep' in user and 'weekly_rep' in user:
                    self.rep['daily'].append(user['daily_rep'])
                    self.rep['weekly'].append(user['weekly_rep'])
                weekly = np.array(sorted(user['commit_profile']['weekly'].items(), key=lambda x: int(x[0])))[:, -1]
                daily = np.array(sorted(user['commit_profile']['daily'].items(), key=lambda x: int(x[0])))[:, -1]
                self.data['daily'].append(daily.astype('float'))
                self.data['weekly'].append(weekly.astype('float'))
                self.users.append(user)

        self.data['daily'] = np.array(self.data['daily'])
        self.data['weekly'] = np.array(self.data['weekly'])
        self.rep['daily'] = np.array(self.rep['daily'])
        self.rep['weekly'] = np.array(self.rep['weekly'])
        self.users = np.array(self.users)

    def load_distance(self):
        self.distance = {'daily': [], 'weekly': []}
        if self.daily_distance_rpath:
            with open(self.daily_distance_rpath, 'r') as rf:
                self.distance['daily'] = np.array(json.load(rf))
        else:
            print 'Cannot load distance from', self.daily_distance_rpath

        if self.weekly_distance_rpath:
            with open(self.weekly_distance_rpath, 'r') as rf:
                self.distance['weekly'] = np.array(json.load(rf))
        else:
            print 'Cannot load distance from', self.weekly_distance_rpath

    def load_seeds(self):
        self.seeds = {'daily': [], 'weekly': []}

        if self.daily_seeds_rpath:
            with open(self.daily_seeds_rpath, 'r') as rf:
                self.daily_seed_dict = json.load(rf)
            self.seeds['daily'] = [[] for _ in range(len(set(self.daily_seed_dict.values())))]
        else:
            print 'Cannot load seeds from', self.daily_seeds_rpath
            self.daily_seed_dict = {}

        if self.weekly_seeds_rpath:
            with open(self.weekly_seeds_rpath, 'r') as rf:
                self.weekly_seed_dict = json.load(rf)
            self.seeds['weekly'] = [[] for _ in range(len(set(self.weekly_seed_dict.values())))]
        else:
            print 'Cannot load seeds from', self.weekly_seeds_rpath
            self.weekly_seed_dict = {}

    def merge_representation(self, daily_rep_rpath, weekly_rep_rpath, wpath):
        with open(daily_rep_rpath, 'r') as rf:
            daily_rep = json.load(rf)
        with open(weekly_rep_rpath, 'r') as rf:
            weekly_rep = json.load(rf)

        with open(wpath, 'w') as wf:
            for u, daily_r, weekly_r in zip(self.users, daily_rep, weekly_rep):
                u['daily_rep'] = daily_r
                u['weekly_rep'] = weekly_r
                wf.write(json.dumps(u) + '\n')

if __name__ == '__main__':
    print 'loading dataset...'
    dataset = Dataset(data_rpath='data/balanced_all.json')

    print 'Store user collection'
    with open('../data/collection.json', 'w') as wf:
        for user in dataset.users:
            wf.write(json.dumps(user) + '\n')

    print 'generating daily distance...'
    ground_dist = ground_distance(24)
    daily_distance = get_distance_matrix(dataset.data['daily'], ground_dist)
    with open('../data/daily_distance', 'w') as wf:
        json.dump(daily_distance.tolist(), wf)

    print 'generating weekly distance...'
    ground_dist = ground_distance(7)
    weekly_distance = get_distance_matrix(dataset.data['weekly'], ground_dist)
    with open('../data/weekly_distance', 'w') as wf:
        json.dump(weekly_distance.tolist(), wf)

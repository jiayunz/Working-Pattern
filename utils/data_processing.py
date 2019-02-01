# coding: utf-8
import json
import numpy as np
from datetime import datetime

class Dataset:
    def __init__(self, data_rpath, weekday_seeds_rpath, weekend_seeds_rpath, weekday_distance_rpath, weekend_distance_rpath):
        self.data_rpath = data_rpath
        self.weekday_seeds_rpath = weekday_seeds_rpath
        self.weekend_seeds_rpath = weekend_seeds_rpath
        self.weekday_distance_rpath = weekday_distance_rpath
        self.weekend_distance_rpath = weekend_distance_rpath
        self.load_data()
        self.load_distance()

    def get_commit_profile(self, commits):
        commit_profile = {
            'weekdays': {},
            'weekends': {}
        }

        for i in range(24):
            commit_profile['weekdays'][i] = 0
            commit_profile['weekends'][i] = 0

        for c in commits:
            time = datetime.strptime(c['commit']['author']['date'][:19], '%Y-%m-%dT%H:%M:%S')
            # 工作日/周末 平均每天在每一时间段占所有工作量的比重
            if time.isoweekday() <= 5:
                commit_profile['weekdays'][time.hour] += 1
            else:
                commit_profile['weekends'][time.hour] += 1

        total_commits_on_weekends = 1. * np.sum(commit_profile['weekends'].values())
        total_commits_in_weekdays = 1. * np.sum(commit_profile['weekdays'].values())
        if total_commits_on_weekends != 0:
            commit_profile['weekends'] = {h: commit_profile['weekends'][h] / total_commits_on_weekends for h in commit_profile['weekends']}
        if total_commits_in_weekdays != 0:
            commit_profile['weekdays'] = {h: commit_profile['weekdays'][h] / total_commits_in_weekdays for h in commit_profile['weekdays']}
        return commit_profile

    def load_data(self):
        self.users = []
        self.rep = {'weekdays': [], 'weekends': []}
        self.data = {'weekdays': [], 'weekends': []}
        self.seeds = {'weekdays': [], 'weekends': []}

        #print 'Loading data from the file...'
        with open(self.weekday_seeds_rpath, 'r') as rf:
            weekday_seed_dict = json.load(rf)
        self.seeds['weekdays'] = [[] for _ in range(len(set(weekday_seed_dict.values())))]

        with open(self.weekend_seeds_rpath, 'r') as rf:
            weekend_seed_dict = json.load(rf)
        self.seeds['weekends'] = [[] for _ in range(len(set(weekend_seed_dict.values())))]

        with open(self.data_rpath, 'r') as rf:
            for i, line in enumerate(rf.readlines()):
                user = json.loads(line.strip())

                if str(user['id']) in weekday_seed_dict:
                    self.seeds['weekdays'][weekday_seed_dict[str(user['id'])]].append(i)

                if str(user['id']) in weekend_seed_dict:
                    self.seeds['weekends'][weekend_seed_dict[str(user['id'])]].append(i)

                if 'commit_profile' not in user:
                    user['commit_profile'] = self.get_commit_profile(user['commits_list'])
                if 'weekday_rep' in user and 'weekend_rep' in user:
                    self.rep['weekdays'].append(user['weekday_rep'])
                    self.rep['weekends'].append(user['weekend_rep'])
                weekends = np.array(sorted(user['commit_profile']['weekends'].items(), key=lambda x: int(x[0])))[:, -1]
                weekdays = np.array(sorted(user['commit_profile']['weekdays'].items(), key=lambda x: int(x[0])))[:, -1]
                self.data['weekdays'].append(weekdays.astype('float'))
                self.data['weekends'].append(weekends.astype('float'))
                self.users.append(user)

        self.data['weekdays'] = np.array(self.data['weekdays'])
        self.data['weekends'] = np.array(self.data['weekends'])
        self.rep['weekdays'] = np.array(self.rep['weekdays'])
        self.rep['weekends'] = np.array(self.rep['weekends'])
        self.users = np.array(self.users)

    def load_distance(self):
        self.distance = dict()
        with open(self.weekday_distance_rpath, 'r') as rf:
            self.distance['weekdays'] = np.array(json.load(rf))
        with open(self.weekend_distance_rpath, 'r') as rf:
            self.distance['weekends'] = np.array(json.load(rf))


    def merge_representation(self, weekday_rep_rpath, weekend_rep_rpath, wpath):
        with open(weekday_rep_rpath, 'r') as rf:
            weekday_rep = json.load(rf)
        with open(weekend_rep_rpath, 'r') as rf:
            weekend_rep = json.load(rf)

        with open(wpath, 'w') as wf:
            for u, weekday_r, weekend_r in zip(self.users, weekday_rep, weekend_rep):
                u['weekday_rep'] = weekday_r
                u['weekend_rep'] = weekend_r
                wf.write(json.dumps(u) + '\n')

if __name__ == '__main__':
    dataset = Dataset(data_rpath='../data/balanced_samples.json',
                      weekday_seeds_rpath='../data/weekday_seeds.json',
                      weekend_seeds_rpath='../data/weekend_seeds.json',
                      weekday_distance_rpath='../data/distance_weekdays',
                      weekend_distance_rpath='../data/distance_weekends',
                      )
    #print dataset.data['weekdays'][0]
    print dataset.seeds['weekdays']
    #with open('../data/collection.json', 'w') as wf:
    #    for u in dataset.users:
    #        wf.write(json.dumps(u) + '\n')

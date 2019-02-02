# coding: utf-8
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_commit_distribution(commits):
    first_date = datetime.strptime(commits[0]['commit']['author']['date'][:19], '%Y-%m-%dT%H:%M:%S').date()
    last_date = datetime.strptime(commits[-1]['commit']['author']['date'][:19], '%Y-%m-%dT%H:%M:%S').date()
    commit_profile = {
        'daily': [],
        'weekly': []
    }

    for c in commits:
        time = c['commit']['author']['date'][:19]
        time = datetime.strptime(time, '%Y-%m-%dT%H:%M:%S')
        commit_profile['daily'].append(time.hour)
        commit_profile['weekly'].append(time.isoweekday())

    return commit_profile['daily']

def load_data(rpath, start):
    cmap = ['RdPu', 'YlOrRd', 'YlGnBu', 'plasma_r', 'rocket_r', 'viridis_r', 'Wistia', 'summer_r']
    fig = plt.figure(figsize=(18, 96))
    count = 0
    with open(rpath, 'r') as rf:
        for line in rf.readlines():
            user = json.loads(line.strip())
            commit_profile = get_commit_distribution(user['commits_list'])

            count += 1
            if count <= start:
                continue
            elif count > start + 300:
                break
            with sns.axes_style('darkgrid'):
                fig.add_subplot(60, 5, count - start)
                plt.title(user['id'])
                sns.distplot(commit_profile, bins=12)
                plt.xticks(fontsize='small')
                plt.yticks(rotation='0', fontsize='small')
                plt.xlim(0, 23)
                plt.ylim(0, 0.2)
                plt.xticks(np.arange(1, 24, 3))
                plt.yticks(np.arange(0, 0.21, 0.05))

    plt.tight_layout()
    plt.savefig('../fig/samples_' + str(start) + '.pdf', format='pdf', dpi=1000)
    plt.show()

if __name__ == '__main__':
    #read_center('../data/center.json')
    #load_data('../data/center.json')
    load_data('../data/balanced_samples.json', start=0)

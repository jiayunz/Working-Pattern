# coding: utf-8
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_commit_hist(commits):
    commit_profile = {
        'daily': [],
        'weekly': []
    }

    for c in commits:
        time = datetime.strptime(c['commit']['author']['date'][:19], '%Y-%m-%dT%H:%M:%S')
        # 每条序列加起来等于1，再设一个特征作为weekends/weekdays
        commit_profile['daily'].append(time.hour)
        commit_profile['weekly'].append(time.isoweekday())

    return commit_profile['daily']

def load_data(rpath):
    cmap = ['RdPu', 'YlOrRd', 'YlGnBu', 'plasma_r', 'rocket_r', 'viridis_r', 'Wistia', 'summer_r']
    fig = plt.figure(figsize=(16, 9))
    count = 0
    with open(rpath, 'r') as rf:
        for line in rf.readlines():
            count += 1
            with sns.axes_style('darkgrid'):
                fig.add_subplot(3, 4, count)
                plt.title('Mode #' + str(count))
                user = json.loads(line.strip())
                commit_profile = get_commit_hist(user['commits_list'])
                #plot_heatmap(commit_profile)
                sns.distplot(commit_profile)
                #plt.xticks(fontsize='small')
                #plt.xticks(rotation='90', fontsize='small')
                #plt.yticks(rotation='0', fontsize='small')
                plt.xlim(0, 23)
                plt.ylim(0, 0.20)
                plt.xticks(np.arange(0, 24, 3))
                plt.yticks(np.arange(0, 0.21, 0.1))
                plt.xlabel('Hour of Day')
                plt.ylabel('Ratio of Commits')

    plt.tight_layout()
    sav_fig = plt.gcf()  # 'get current figure'
    sav_fig.savefig('../fig/center.pdf', format='pdf', dpi=1000)
    plt.show()

def plot_center(medoid_users, fig_wpath):
    fig = plt.figure(figsize=(16, 9))
    count = 0
    for user in medoid_users:
        count += 1
        with sns.axes_style('darkgrid'):
            fig.add_subplot(3, 4, count)
            plt.title('Mode #' + str(count))
            commit_profile = []

            for c in user['commits_list']:
                time = datetime.strptime(c['commit']['author']['date'][:19], '%Y-%m-%dT%H:%M:%S')
                # 每条序列加起来等于1，再设一个特征作为weekends/weekdays
                commit_profile.append(time.hour)

            sns.distplot(commit_profile)
            #plt.xticks(fontsize='small')
            #plt.xticks(rotation='90', fontsize='small')
            #plt.yticks(rotation='0', fontsize='small')
            plt.xlim(0, 23)
            plt.ylim(0, 0.20)
            plt.xticks(np.arange(0, 24, 3))
            plt.yticks(np.arange(0, 0.21, 0.1))
            plt.xlabel('Hour of Day')
            plt.ylabel('Ratio of Commits')

    plt.tight_layout()
    sav_fig = plt.gcf()  # 'get current figure'
    sav_fig.savefig(fig_wpath, format='pdf', dpi=1000)
    plt.show()

if __name__ == '__main__':
    #load_data('weekend_center.json')
    load_data('../data/daily_center.json')

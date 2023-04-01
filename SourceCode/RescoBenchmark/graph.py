from curses import window
import matplotlib.pyplot as plt 
import numpy as np
from collections import deque

from avg_duration import durations
from avg_queue import queue
from avg_timeLoss import delays
from avg_waitingTime import waiting

map_title = {
    'grid4x4': '4x4 Grid',
    'arterial4x4': '4x4 Avenues',
    'ingolstadt1': 'Ingolstadt Single Signal',
    'ingolstadt7': 'Ingolstadt Corridor',
    'ingolstadt21': 'Ingolstadt Region',
    'cologne1': 'Cologne Single Signal',
    'cologne3': 'Cologne Corridor',
    'cologne8': 'Cologne Region'
}

alg_name = {
    'FIXED': 'Fixed Time',
    'STOCHASTIC': 'Random',
    'MAXWAVE': 'Greedy',
    'MAXPRESSURE': 'Max Pressure',
    'FULLMAXPRESSURE': 'Max Pressure w/ All phases',
    'IDQN': 'IDQN',
    'MPLight': 'MPLight',
    'MPLightFULL': 'Full State MPLight',
    'FMA2C': 'FMA2C',
    'IPPO': 'IPPO'
}

statics = ['IDQN', 'IPPO', 'MPLight', 'MPLightFULL']

num_n = -1
num_episodes = 100
fs = 21
window_size = 5

# duration is the time the vehicle is in that lane.
metrics = [delays, durations, queue, waiting]
metric_str = ['avg_delay', 'trip_time', 'avg_queue', 'avg_waiting']

chart = {
    'IDQN': {
        'avg_delay': [],
        'trip_time': [],
        'avg_queue': [],
        'avg_waiting': []
    },

    'IPPO': {
        'avg_delay': [],
        'trip_time': [],
        'avg_queue': [],
        'avg_waiting': []
    }, 

    'MPLight':{
        'avg_delay': [],
        'trip_time': [],
        'avg_queue': [],
        'avg_waiting': []
    }, 

    'Full State MPLight': {
        'avg_delay': [],
        'trip_time': [],
        'avg_queue': [],
        'avg_waiting': []
    }
}
for i, metric in enumerate(metrics):
    #print('\n', metric_str[i])
    for map in map_title.keys():
        # print(' ')
        # print(map_title[map])
        dqn_max = 0
        plt.gca().set_prop_cycle(None)  # set default color
        for key in metric:
            if map in key and 'yerr' not in key:
                alg = key.split(' ')[0]  # get agent
                key_map = key.split(' ')[1]  # get map
                
                # get the largest element in the array 
                if alg == 'IDQN': dqn_max = np.max(metric[key]) 

                # if not value, 
                if len(metric[key]) == 0:
                    plt.plot([], [])
                    plt.fill_between([], [], [])
                    continue 
                
                # get minimum value
                if num_n == -1:
                    last_n_ind = np.argmin(metric[key])  # Returns the indices of the minimum values
                    last_n = metric[key][last_n_ind]  # get minimum value
                else:
                    last_n_ind = np.argmin(metric[key][-num_n :])
                    last_n = metric[key][-num_n: ][last_n_ind]
                
                # process error
                err = metric.get(key + '_yerr')
                last_n_err = 0 if err is None else err[last_n_ind] # get error at the positon of minimum values
                avg_tot = np.round(np.mean(metric[key]), 2) # get average
                last_n = np.round(last_n, 2)  # rounding minimum value
                last_n_err = np.round(last_n_err, 2)  # rounding minimum err

                # process with chart
                if alg in statics: 
                    print('{} {}'.format(alg_name[alg], avg_tot))
                else:
                    print('{} {} +- {}'.format(alg_name[alg], last_n, last_n_err ))
                    if not(map == 'grid4x4' or map == 'arterial4x4'):
                        chart[alg_name[alg]][metric_str[i]].append(str(last_n))   # put minimum value into array
                    
                # Build plots
                if alg in statics:
                    plt.plot([avg_tot] * num_episodes, '--', label= alg_name[alg])
                    plt.fill_between([], [], [])
    
                elif not('FMA2C' in alg or 'IPPO' in alg):
                    windowed = []
                    queue = deque(maxlen= window_size) # size of sliding window
                    std_q = deque(maxlen= window_size)
                    
                    windowed_yerr = []
                    x = []
                    for i, eps in enumerate(metric[key]):
                        x.append(i)
                        queue.append(eps)
                        windowed.append(np.mean(queue))
                        if err is not None:
                            std_q.append(err[i])
                            windowed_yerr.append(np.mean(std_q))
                    windowed = np.asarray(windowed)
                    if err is not None:
                        windowed_yerr = np.asarray(windowed_yerr)
                        low = windowed - windowed_yerr
                        high = windowed + windowed_yerr
                    else:
                        low = windowed
                        high = windowed
                    plt.plot(windowed, label = alg_name[alg])
                    plt.fill_between(x, low, high, alpha= 0.3)
                else:
                    if alg == 'FMA2C':   
                        plt.plot([], [])
                        plt.fill_between([],[],[])
                    alg = key.split(' ')[0]
                    x = [num_episodes - 1, num_episodes]
                    
                
        #     print(key)
        # print(" ")
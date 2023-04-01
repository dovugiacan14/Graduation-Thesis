from cProfile import label
import os
from tkinter.font import BOLD
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import sys
from config.map_config import map_configs

log_dir = '/home/gca/Desktop/RESCOmain/logs_ingolstadt1/'
#log_dir = '/home/gca/Desktop/RESCOmain/logs_cologne1/'
env_base = '..'+os.sep+'environments'+os.sep
names = [folder for folder in next(os.walk(log_dir))[1]]
# names = ['MPLight-tr0-ingolstadt21-21-mplight-pressure', 
#         'MPLight-tr0-ingolstadt21-21-mplight_full-pressure', 
#         'MPLight-tr0-ingolstadt1-0-mplight-pressure', 
#         'MPLight-tr0-ingolstadt1-0-mplight_full-pressure']
metric = 'queue'
output_file = 'avg_{}.py'.format(metric)
run_avg = dict()
for name in names:
    split_name = name.split('-')
    # split_name = ['MPLight', 'tr0', 'ingolstadt21', '21', 'mplight', 'pressure']
    map_name = split_name[2] # map_name = ingolstadt21
    average_per_episode = [] 
    for i in range(1, 10000):
        trip_file_name = log_dir + name + os.sep + 'metrics_'+ str(i) + '.csv'
        if not os.path.exists(trip_file_name):
            print('No '+trip_file_name)
            break
# trip_file_name : the directory to each file 
        num_steps, total = 0, 0.0
        last_departure_time = 0
        last_depart_id = ''
        with open(trip_file_name) as fp:
            reward, wait, steps = 0, 0, 0
            for line in fp:
                line = line.split('}') # get all lines in all file
                queues = line[2]   # get queue_length row 
                # if queues = , {'gneJ207': 1
                signals = queues.split(':')
                # then signals = [", {'gneJ207'", ' 1']
                step_total = 0
                for s, signal in enumerate(signals):
                    if s == 0: continue
                    queue = signal.split(',')
                    queue = float(queue[0])
                    step_total += queue
                step_avg = step_total / len(signals)  # get the average queue length at each line 
                total += step_avg # get total average queue length in one episode
                num_steps += 1
        average = total / num_steps # average queue length in one episode 
        average_per_episode.append(average)

    run_name = split_name[0]+' '+split_name[2]+' '+split_name[3]+' '+split_name[4]+' '+split_name[5]
    average_per_episode = np.asarray(average_per_episode)
    if run_name in run_avg:
        run_avg[run_name].append(average_per_episode)
    else:
        run_avg[run_name] = [average_per_episode]
    # run_avg : a dictionary with keys are the name of files that need to plot, 
    # values are arrays that contain the average value of each episode in file. 


alg_name = []
alg_res = []
# save metric
_name = []
metrics = []
error = []
for run_name in run_avg:
    list_runs = run_avg[run_name] # get the list files in file log to plot 
    min_len = min([len(run) for run in list_runs])   
    list_runs = [run[:min_len] for run in list_runs]
    avg_delays = np.sum(list_runs, 0)/len(list_runs) # get the average of all runs 
    err = np.std(list_runs,axis= 0)
    metrics.append(avg_delays)
    error.append(err)
    _name.append(run_name)

    alg_name.append(run_name)
    alg_res.append(avg_delays)

    alg_name.append(run_name+'_yerr')
    alg_res.append(err)

# get name of agents 
n = []
for i in range(len(_name)):
    a = _name[i].split(' ')
    n.append(a[0])
print(n)


# Plot
x = np.arange(0,100,1)
plt.title(map_name, fontsize = 50, weight= BOLD)
plt.xlabel("number episode", fontsize= 30)
plt.ylabel("queue length", fontsize= 30)
plt.tick_params(axis= 'x', labelsize= 30)
plt.tick_params(axis= 'y', labelsize= 30)
colors = ['red','green', 'grey','brown','purple', 'yellow', 'blue', ]
for i in range(len(_name)):
    temp = colors[1]
    if str(n[i]) == 'MPLightFULL':
        temp = colors[5]
    elif str(n[i]) == 'MPLight':
        temp = colors[4]
    elif str(n[i]) == 'IDQN':
        temp = colors[0]
    elif str(n[i]) == 'IPPO':
        temp = colors[3]
    elif str(n[i]) == 'FMA2C':
        temp = colors[2]
    elif str(n[i]) == 'FixedTime':
        temp = colors[6]
    plt.plot(metrics[i], label = str(n[i]), linewidth = 7, color= temp)
    plt.fill_between(x, metrics[i] - error[i], metrics[i] + error[i], alpha = 0.1, color = temp)
    plt.legend(loc= 'lower left', title = "Agent", fontsize = 20)
plt.show()


# np.set_printoptions(threshold=sys.maxsize)
# with open(output_file, 'a') as out:
#     for i, res in enumerate(alg_res):
#         out.write("'{}': {},\n".format(alg_name[i], res.tolist()))


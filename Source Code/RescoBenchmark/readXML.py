import os
from tkinter.font import BOLD
from turtle import color
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import sys
from config.map_config import map_configs

log_dir = '/home/gca/Desktop/RESCOmain/logs_ingolstadt1/'
#log_dir = '/home/gca/Desktop/RESCOmain/logggs/'
env_base = '/home/gca/Desktop/RESCOmain'+os.sep+'environments'+os.sep
names = [folder for folder in next(os.walk(log_dir))[1]]

metrics = ['timeLoss', 'duration', 'waitingTime']
list_metrics = []
list_err = []
n = []
for metric in metrics:
    output_file = 'avg_{}.py'.format(metric)
    run_avg = dict()
    for name in names:
        split_name = name.split('-')
        #print(split_name)
        # spit_name = ['MPLight', 'tr0', 'ingolstadt1', '0', 'mplight_full', 'pressure']
        map_name = split_name[2]
        average_per_episode = []
        for i in range(1, 10000):
            trip_file_name = log_dir + name + os.sep + 'tripinfo_'+ str(i) +'.xml'
            if not os.path.exists(trip_file_name):
                print('No '+trip_file_name)
                break
            try:
                tree = ET.parse(trip_file_name)  # parse each file .xml in file logs 
                root = tree.getroot()
                num_trips, total = 0, 0.0
                last_departure_time = 0
                last_depart_id = ''
                for child in root:
                    try:
                        num_trips += 1
                        total += float(child.attrib[metric]) # get total timeLoss in one file
                        if metric == 'timeLoss':
                            total += float(child.attrib['departDelay'])
                            depart_time = float(child.attrib['depart'])
                            if depart_time > last_departure_time:
                                last_departure_time = depart_time   # get depart time of vehicle in that opisode
                                last_depart_id = child.attrib['id']  # get id of that vehicle
                    except Exception as e:
                        break

                route_file_name = env_base + map_name + os.sep + map_name + '_' + str(i) + '.rou.xml'
                # get file rou of map
                if metric == 'timeLoss':    # Calc. departure delays
                    try:
                        tree = ET.parse(route_file_name)
                    except FileNotFoundError:
                        route_file_name = env_base + map_name + os.sep + map_name + '.rou.xml'
                        tree = ET.parse(route_file_name)
                    root = tree.getroot()
                    last_departure_time = None
                    
                    for child in root: 
                        if child.attrib['id'] == last_depart_id:
                            last_departure_time = float(child.attrib['depart'])  
                    
                    # Get the time it was suppose to depart (the suppose time have in file rou)
                    never_departed = []
                    if last_departure_time is None: raise Exception('Wrong trip file')
                    
                    for child in root:
                        if child.tag != 'vehicle': continue
                        depart_time = float(child.attrib['depart'])
                        # depart_time in file rou of the environment and last_depart_time still in file rou of tripinfo 
                        if depart_time > last_departure_time:
                            never_departed.append(depart_time)

                    never_departed = np.asarray(never_departed)
                    never_departed_delay = np.sum(float(map_configs[map_name]['end_time']) - never_departed)
                    total += never_departed_delay
                    num_trips += len(never_departed)
                average = total / num_trips
                average_per_episode.append(average)
                

            except ET.ParseError as e:
                #raise e
                break

# get metrics 
        run_name = split_name[0]+' '+split_name[2]+' '+split_name[3]+' '+split_name[4]+' '+split_name[5]
        average_per_episode = np.asarray(average_per_episode)

        if run_name in run_avg:
            run_avg[run_name].append(average_per_episode)
        else:
            run_avg[run_name] = [average_per_episode]

# save metrics 
    alg_res = []
    alg_name = []
    _name = []
    for run_name in run_avg:
        list_runs = run_avg[run_name]
        min_len = min([len(run) for run in list_runs])
        list_runs = [run[:min_len] for run in list_runs]
        avg_delays = np.sum(list_runs, 0)/len(list_runs)
        err = np.std(list_runs, axis= 0)
        _name.append(run_name)

        alg_name.append(run_name)
        alg_res.append(avg_delays)

        alg_name.append(run_name+'_yerr')
        alg_res.append(err)
        list_metrics.append(avg_delays)
        list_err.append(err)
        

    # get name of agents 
    for i in range(len(_name)):
        a = _name[i].split(' ')
        n.append(a[0])
    # # Create saved file
    # np.set_printoptions(threshold=sys.maxsize)
    # with open(output_file, 'a') as out:
    #     for i, res in enumerate(alg_res):
    #         out.write("'{}': {},\n".format(alg_name[i], res.tolist()))
# print('-----------------------')
# print(list_metrics)
# print(len(list_metrics))
# # Plot 
# x = np.arange(1, 101, 1)
# fig, axis = plt.subplots(1,3, figsize= (20,15))
# colors = ['red','green', 'grey','brown','purple', 'yellow', 'blue']

# axis[0].set_title("Time Loss", fontsize = 30, fontweight ="bold")
# axis[0].set_xlabel('number episode',fontsize =20)
# axis[0].set_ylabel('timeLoss', fontsize = 20)
# axis[0].tick_params(axis= 'x', labelsize= 20)
# axis[0].tick_params(axis= 'y', labelsize= 20)
# for i in range(0,7):
#     temp = colors[1]
#     if str(n[i]) == 'MPLightFULL':
#         temp = colors[5]
#     elif str(n[i]) == 'MPLight':
#         temp = colors[4]
#     elif str(n[i]) == 'IDQN':
#         temp = colors[0]
#     elif str(n[i]) == 'IPPO':
#         temp = colors[3]
#     elif str(n[i]) == 'FMA2C':
#         temp = colors[2]
#     elif str(n[i]) == 'FixedTime':
#         temp = colors[6]
#     axis[0].plot(x, list_metrics[i],label = str(n[i]), linewidth = 4, color= temp)
#     axis[0].fill_between(x, list_metrics[i] - list_err[i], list_metrics[i] + list_err[i], alpha= 0.3, color = temp)
#     axis[0].legend(loc= 'upper left', title = "Agent", fontsize = 20)
# fig.suptitle(map_name, fontsize = 40, fontweight = 'bold')

# axis[1].set_title("Duration", fontsize = 30, fontweight ="bold")
# axis[1].set_xlabel('number episode',fontsize = 20)
# axis[1].set_ylabel('duration',fontsize = 20)
# axis[1].tick_params(axis= 'x', labelsize= 20)
# axis[1].tick_params(axis= 'y', labelsize= 20)
# for i in range(7,14):
#     temp = colors[1]
#     if str(n[i]) == 'MPLightFULL':
#         temp = colors[5]
#     elif str(n[i]) == 'MPLight':
#         temp = colors[4]
#     elif str(n[i]) == 'IDQN':
#         temp = colors[0]
#     elif str(n[i]) == 'IPPO':
#         temp = colors[3]
#     elif str(n[i]) == 'FMA2C':
#         temp = colors[2]
#     elif str(n[i]) == 'FixedTime':
#         temp = colors[6]
#     axis[1].plot(x, list_metrics[i],label = str(n[i]), linewidth = 4, color = temp)
#     axis[1].fill_between(x, list_metrics[i] - list_err[i], list_metrics[i] + list_err[i], alpha= 0.3, color = temp)
#     axis[1].legend(loc= 'upper left', title = "Agent", fontsize = 20)


# axis[2].set_title("The Waiting Time", fontsize = 30,fontweight ="bold")
# axis[2].set_xlabel('number episode',fontsize = 20)
# axis[2].set_ylabel('wait_time',fontsize = 20)
# axis[2].tick_params(axis= 'x', labelsize= 20)
# axis[2].tick_params(axis= 'y', labelsize= 20)
# for i in range(14,21):
#     temp = colors[1]
#     if str(n[i]) == 'MPLightFULL':
#         temp = colors[5]
#     elif str(n[i]) == 'MPLight':
#         temp = colors[4]
#     elif str(n[i]) == 'IDQN':
#         temp = colors[0]
#     elif str(n[i]) == 'IPPO':
#         temp = colors[3]
#     elif str(n[i]) == 'FMA2C':
#         temp = colors[2]
#     elif str(n[i]) == 'FixedTime':
#         temp = colors[6]
#     axis[2].plot(x, list_metrics[i],label = str(n[i]), linewidth = 4, color = temp)
#     axis[2].fill_between(x, list_metrics[i] - list_err[i], list_metrics[i] + list_err[i], alpha= 0.2, color = temp)
#     axis[2].legend(loc= 'upper left', title = "Agent", fontsize = 20)

# plt.tight_layout()
# plt.show()

# -------------------------------------------------
x = np.arange(1, 101, 1)
colors = ['red','green', 'grey','brown','purple', 'yellow', 'blue']
tmp = 0

for m in range(len(metrics)):
    if m == 0: 
        metrics[m] = 'Delays'
        plt.ylim(20,100)
    elif m == 1:
        plt.ylim(30,80)
    else:
        plt.ylim(0,40)
    for i in range(tmp, tmp + 7):
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
        plt.plot(x, list_metrics[i],label = str(n[i]), linewidth = 7, color = temp)
        plt.tick_params(axis= 'x', labelsize= 30)
        plt.tick_params(axis= 'y', labelsize= 30) 
        plt.fill_between(x, list_metrics[i] - list_err[i], list_metrics[i] + list_err[i], alpha= 0.2, color = temp)   
        plt.legend(loc= 'upper left', title = "Agent", fontsize = 20)  
        plt.xlabel("Episode", fontsize= 40, weight= BOLD)
        plt.ylabel(metrics[m], fontsize= 40, weight= BOLD)
    plt.show()  
    tmp += 7    

# -------------------------------------------------

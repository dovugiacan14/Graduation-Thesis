import pathlib
import os
import multiprocessing as mp
import random


from multi_signal import MultiSignal
import argparse
from config.agent_config import agent_configs
from config.map_config import map_configs
from config.mdp_config import mdp_configs
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", type=str, default='MPLight',
                    choices=['STOCHASTIC', 'MAXWAVE', 'MAXPRESSURE', 'IDQN', 'IPPO', 'MPLight', 'MA2C', 'FMA2C',
                             'MPLightFULL', 'FMA2CFull', 'FMA2CVAL', 'IDoubleDQN'])
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--eps", type=int, default=100)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--map", type=str, default='ingolstadt1',
                    choices=['grid4x4', 'arterial4x4', 'ingolstadt1', 'ingolstadt7', 'ingolstadt21',
                             'cologne1', 'cologne3', 'cologne8',
                             ])
    ap.add_argument("--pwd", type=str, default=str(pathlib.Path().absolute())+os.sep)
    ap.add_argument("--log_dir", type=str, default=str(pathlib.Path().absolute())+os.sep+'logs'+os.sep)
    ap.add_argument("--gui", type=bool, default=False)
    ap.add_argument("--libsumo", type=bool, default=False)
    ap.add_argument("--tr", type=int, default=0)  # Can't multi-thread with libsumo, provide a trial number
    args = ap.parse_args()
    for seed in range(1,2):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if args.procs == 1 or args.libsumo:
            run_trial(args, args.tr, seed)
        else:
            pool = mp.Pool(processes=args.procs)
            for trial in range(1, args.trials+1):
                pool.apply_async(run_trial, args=(args, trial))
            pool.close()
            pool.join()
# trial <=> thread number of libsumo (of agent)
# procescs can be understood that if we give in more than one agent, it will be run parallel.
# proc <=> process

def run_trial(args, trial, seed):
    mdp_config = mdp_configs.get(args.agent)
    if mdp_config is not None:
        mdp_map_config = mdp_config.get(args.map)
        if mdp_map_config is not None:
            mdp_config = mdp_map_config
        mdp_configs[args.agent] = mdp_config

    agt_config = agent_configs[args.agent]
    agt_map_config = agt_config.get(args.map)
    if agt_map_config is not None:
        agt_config = agt_map_config
    alg = agt_config['agent']

    if mdp_config is not None:
        agt_config['mdp'] = mdp_config
        management = agt_config['mdp'].get('management')
        if management is not None:    # Save some time and precompute the reverse mapping
            supervisors = dict()
            for manager in management:
                workers = management[manager]
                for worker in workers:
                    supervisors[worker] = manager
            mdp_config['supervisors'] = supervisors

    map_config = map_configs[args.map]
    num_steps_eps = int((map_config['end_time'] - map_config['start_time']) / map_config['step_length'])
    route = map_config['route']
    if route is not None: route = args.pwd + route

    env = MultiSignal(alg.__name__+'-tr'+str(trial),
                      args.map,
                      seed,
                      args.pwd + map_config['net'],
                      agt_config['state'],
                      agt_config['reward'],
                      route=route, step_length=map_config['step_length'], yellow_length=map_config['yellow_length'],
                      step_ratio=map_config['step_ratio'], end_time=map_config['end_time'],
                      max_distance=agt_config['max_distance'], lights=map_config['lights'], gui=args.gui,
                      log_dir=args.log_dir, libsumo=args.libsumo, warmup=map_config['warmup'])

    agt_config['episodes'] = int(args.eps * 0.8)    # schedulers decay over 80% of steps
    agt_config['steps'] = agt_config['episodes'] * num_steps_eps
    agt_config['log_dir'] = args.log_dir + env.connection_name + os.sep
    agt_config['num_lights'] = len(env.all_ts_ids)

    # Get agent id's, observation shapes, and action sizes from env
    obs_act = dict()
    for key in env.obs_shape:
        obs_act[key] = [env.obs_shape[key], len(env.phases[key]) if key in env.phases else None]
    agent = alg(agt_config, obs_act, args.map, trial)

# # Load model 
#     for key in obs_act:
#         agent.agent.load('./MPLightFULLModel/mplight_{}'.format(key))
    
    # for key in obs_act:
    #     agent.agents[key].load('./IDQNModel/idqn_{}'.format(key))
    #agent.agent.load('./log/MPLight-tr0-ingolstadt21-21-mplight-pressure/agent')

    for _ in range(args.eps):
        obs = env.reset()
        done = False
        sgn = 0
        while not done:
            act = agent.act(obs)
            # for key in act.keys():
            #     act[key] = sgn
            # sgn += 1
            # if sgn > 2:
            #     sgn = 0
            # for key in act.keys():
            #     act[key] = random.randint(0, 2)
            # print(act)       
            #print(act)
            obs, rew, done, info = env.step(act)
            agent.observe(obs, rew, done, info)
    env.close()

    # save MPLight model 
    # for key in obs_act:
    #     agent.agent.save('./model/cologne3/IDQNModel/idqn_{}'.format(key))
    
    # for key in obs_act:
    #     agent.agents[key].save('./model/cologne3/IDoubleDQNModel/idoubledqn_{}'.format(key))
    
    

if __name__ == '__main__':
    main()

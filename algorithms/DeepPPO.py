# Author: Aqeel Anwar(ICSRL)
# Created: 10/4/2020, 10:43 PM
# Email: aqeel.anwar@gatech.edu

import sys, cv2, time
import nvidia_smi
import psutil
import numpy as np
import pygame
import matplotlib.pyplot as plt  # Added for plotting
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D trajectory plot
from network.agent import PedraAgent
from unreal_envs.initial_positions import *
from os import getpid
from network.Memory import Memory
from aux_functions import *
import os
from util.transformations import euler_from_quaternion, quaternion_from_euler
from configs.read_cfg import read_cfg, update_algorithm_cfg
import pandas as pd

def DeepPPO(cfg, env_process, env_folder):
    algorithm_cfg = read_cfg(config_filename='configs/DeepPPO.cfg', verbose=True)
    algorithm_cfg.algorithm = cfg.algorithm

    if 'GlobalLearningGlobalUpdate-SA' in algorithm_cfg.distributed_algo:
        # algorithm_cfg = update_algorithm_cfg(algorithm_cfg, cfg)
        cfg.num_agents = 1
    client = []
    # Connect to Unreal Engine and get the drone handle: client
    client, old_posit, initZ = connect_drone(ip_address=cfg.ip_address, phase=cfg.mode,
                                             num_agents=cfg.num_agents, client=client)
    # Save the initial position for later use in inference
    initial_pos = old_posit.copy()
    # Load the initial positions for the environment
    reset_array, reset_array_raw, level_name, crash_threshold = initial_positions(cfg.env_name, initZ, cfg.num_agents)

    # Initialize System Handlers
    process = psutil.Process(getpid())

    # Load PyGame Screen
    screen = pygame_connect(phase=cfg.mode)

    fig_z = []
    fig_nav = []
    debug = False
    # Generate path where the weights will be saved
    cfg, algorithm_cfg = save_network_path(cfg=cfg, algorithm_cfg=algorithm_cfg)
    current_state = {}
    new_state = {}
    posit = {}
    name_agent_list = []
    data_tuple = {}
    agent = {}
    epi_num = {}

    if cfg.mode == 'train':
        iter = {}
        wait_for_others = {}
        if algorithm_cfg.distributed_algo == 'GlobalLearningGlobalUpdate-MA':
            print_orderly('global', 40)
            # Multiple agent acts as data collector and one global learner
            global_agent = PedraAgent(algorithm_cfg, client, name='DQN', vehicle_name='global')

        for drone in range(cfg.num_agents):
            name_agent = "drone" + str(drone)
            wait_for_others[name_agent] = False
            iter[name_agent] = 1
            epi_num[name_agent] = 1
            data_tuple[name_agent] = []
            name_agent_list.append(name_agent)
            print_orderly(name_agent, 40)
            # TODO: turn the neural network off if global agent is present
            agent[name_agent] = PedraAgent(algorithm_cfg, client, name='DQN', vehicle_name=name_agent)
            current_state[name_agent] = agent[name_agent].get_state()

    elif cfg.mode == 'infer':
        # In inference, we use only one agent
        iter = 1
        name_agent = 'drone0'
        name_agent_list.append(name_agent)
        agent[name_agent] = PedraAgent(algorithm_cfg, client, name=name_agent + 'DQN', vehicle_name=name_agent)

        env_cfg = read_cfg(config_filename=env_folder + '/config.cfg', verbose=True)
        nav_x = []
        nav_y = []
        altitude = {}
        altitude[name_agent] = []
        p_z, f_z, fig_z, ax_z, line_z, fig_nav, ax_nav, nav = initialize_infer(
            env_cfg=env_cfg, client=client, env_folder=env_folder)
        nav_text = ax_nav.text(0, 0, '')

        # ---- New: Initialize additional figures for inference metrics ----
        # Figure for iteration time per loop
        fig_time, ax_time = plt.subplots()
        ax_time.set_title("Iteration Time (s)")
        ax_time.set_xlabel("Iteration")
        ax_time.set_ylabel("Time (s)")
        time_exec_list = []  # list to store execution time per iteration

        # Figure for cumulative distance vs. iterations
        fig_distance, ax_distance = plt.subplots()
        ax_distance.set_title("Cumulative Distance")
        ax_distance.set_xlabel("Iteration")
        ax_distance.set_ylabel("Distance")
        distance_list = []  # list to store cumulative distance values

        # New Plot: 3D trajectory plot (X, Y, Z)
        fig_3d = plt.figure()
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        ax_3d.set_title("3D Trajectory")
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        traj_x = []  # list to store X coordinates
        traj_y = []  # list to store Y coordinates
        traj_z = []  # list to store Z coordinates

        # New Plot: Orientation (Yaw) over time
        fig_yaw, ax_yaw = plt.subplots()
        ax_yaw.set_title("Orientation (Yaw) Over Time")
        ax_yaw.set_xlabel("Iteration")
        ax_yaw.set_ylabel("Yaw (radians)")
        yaw_list = []  # list to store yaw values

        # New Plot: Action distribution histogram
        fig_action, ax_action = plt.subplots()
        ax_action.set_title("Action Distribution")
        ax_action.set_xlabel("Action")
        ax_action.set_ylabel("Frequency")
        action_list = []  # list to store predicted action strings
        # --------------------------------------------------------------------

        # Select initial position
        reset_to_initial(0, reset_array, client, vehicle_name=name_agent)
        old_posit[name_agent] = client.simGetVehiclePose(vehicle_name=name_agent)
    # Initialize variables (common to both modes)
    episode = {}
    active = True

    print_interval = 1
    automate = True
    choose = False
    print_qval = False
    last_crash = {}
    ret = {}
    distance = {}
    num_collisions = {}
    level = {}
    level_state = {}
    level_posit = {}
    times_switch = {}
    last_crash_array = {}
    ret_array = {}
    distance_array = {}
    epi_env_array = {}
    log_files = {}

    # If the phase is inference force the num_agents to 1
    hyphens = '-' * int((80 - len('Log files')) / 2)
    print(hyphens + ' ' + 'Log files' + ' ' + hyphens)
    for name_agent in name_agent_list:
        ret[name_agent] = 0
        num_collisions[name_agent] = 0
        last_crash[name_agent] = 0
        level[name_agent] = 0
        episode[name_agent] = 0
        level_state[name_agent] = [None] * len(reset_array[name_agent])
        level_posit[name_agent] = [None] * len(reset_array[name_agent])
        times_switch[name_agent] = 0
        last_crash_array[name_agent] = np.zeros(shape=len(reset_array[name_agent]), dtype=np.int32)
        ret_array[name_agent] = np.zeros(shape=len(reset_array[name_agent]))
        distance_array[name_agent] = np.zeros(shape=len(reset_array[name_agent]))
        epi_env_array[name_agent] = np.zeros(shape=len(reset_array[name_agent]), dtype=np.int32)
        distance[name_agent] = 0
        # Log file
        log_path = algorithm_cfg.network_path + '/' + name_agent + '/' + cfg.mode + 'log.txt'
        print("Log path: ", log_path)
        log_files[name_agent] = open(log_path, 'w')

    print_orderly('Simulation begins', 80)

    while active:
        try:
            active, automate, algorithm_cfg, client = check_user_input(
                active, automate, agent[name_agent], client, old_posit[name_agent], initZ,
                fig_z, fig_nav, env_folder, cfg, algorithm_cfg)

            if automate:
                if cfg.mode == 'train':
                    if iter[name_agent] % algorithm_cfg.switch_env_steps == 0:
                        switch_env = True
                    else:
                        switch_env = False

                    for name_agent in name_agent_list:
                        while not wait_for_others[name_agent]:
                            start_time = time.time()
                            if switch_env:
                                posit1_old = client.simGetVehiclePose(vehicle_name=name_agent)
                                times_switch[name_agent] = times_switch[name_agent] + 1
                                level_state[name_agent][level[name_agent]] = current_state[name_agent]
                                level_posit[name_agent][level[name_agent]] = posit1_old
                                last_crash_array[name_agent][level[name_agent]] = last_crash[name_agent]
                                ret_array[name_agent][level[name_agent]] = ret[name_agent]
                                distance_array[name_agent][level[name_agent]] = distance[name_agent]
                                epi_env_array[name_agent][level[name_agent]] = episode[name_agent]

                                level[name_agent] = (level[name_agent] + 1) % len(reset_array[name_agent])

                                print(name_agent + ' :Transferring to level: ', level[name_agent], ' - ',
                                      level_name[name_agent][level[name_agent]])

                                if times_switch[name_agent] < len(reset_array[name_agent]):
                                    reset_to_initial(level[name_agent], reset_array, client, vehicle_name=name_agent)
                                else:
                                    current_state[name_agent] = level_state[name_agent][level[name_agent]]
                                    posit1_old = level_posit[name_agent][level[name_agent]]
                                    reset_to_initial(level[name_agent], reset_array, client, vehicle_name=name_agent)
                                    client.simSetVehiclePose(posit1_old, ignore_collison=True, vehicle_name=name_agent)
                                    time.sleep(0.1)

                                last_crash[name_agent] = last_crash_array[name_agent][level[name_agent]]
                                ret[name_agent] = ret_array[name_agent][level[name_agent]]
                                distance[name_agent] = distance_array[name_agent][level[name_agent]]
                                episode[name_agent] = epi_env_array[name_agent][int(level[name_agent] / 3)]
                            else:
                                if algorithm_cfg.distributed_algo == 'GlobalLearningGlobalUpdate-MA':
                                    agent_this_drone = global_agent
                                else:
                                    agent_this_drone = agent[name_agent]

                                action, p_a, action_type = policy_PPO(current_state[name_agent], agent_this_drone)
                                action_word = translate_action(action, algorithm_cfg.num_actions)
                                agent[name_agent].take_action(action, algorithm_cfg.num_actions, Mode='static')
                                new_state[name_agent] = agent[name_agent].get_state()
                                new_depth1, thresh = agent[name_agent].get_CustomDepth(cfg)

                                posit[name_agent] = client.simGetVehiclePose(vehicle_name=name_agent)
                                position = posit[name_agent].position
                                old_p = np.array([old_posit[name_agent].position.x_val, old_posit[name_agent].position.y_val])
                                new_p = np.array([position.x_val, position.y_val])
                                distance[name_agent] = distance[name_agent] + np.linalg.norm(new_p - old_p)
                                old_posit[name_agent] = posit[name_agent]
                                reward, crash = agent[name_agent].reward_gen(new_depth1, action, crash_threshold, thresh,
                                                                             debug, cfg)

                                ret[name_agent] = ret[name_agent] + reward
                                agent_state = agent[name_agent].GetAgentState()

                                if agent_state.has_collided or distance[name_agent] < 0.01:
                                    num_collisions[name_agent] = num_collisions[name_agent] + 1
                                    if agent_state.has_collided:
                                        print('Crash: Collision detected from environment')
                                    else:
                                        print('Crash: Collision detected from distance')
                                    crash = True
                                    reward = -1

                                data_tuple[name_agent].append([current_state[name_agent], action, new_state[name_agent],
                                                               reward, p_a, crash])

                                if crash:
                                    wait_for_others[name_agent] = True
                                    if distance[name_agent] < 0.01:
                                        print('Recovering from drone mobility issue')
                                        agent[name_agent].client, old_posit, initZ = connect_drone(
                                            ip_address=cfg.ip_address, phase=cfg.mode,
                                            num_agents=cfg.num_agents, client=client)
                                        agent[name_agent].client, old_posit, initZ = connect_drone(
                                            ip_address=cfg.ip_address, phase=cfg.mode,
                                            num_agents=cfg.num_agents, client=client)
                                        time.sleep(2)
                                        wait_for_others[name_agent] = False
                                    else:
                                        agent[name_agent].network_model.log_to_tensorboard(tag='Return',
                                                                                           group=name_agent,
                                                                                           value=ret[name_agent],
                                                                                           index=epi_num[name_agent])
                                        agent[name_agent].network_model.log_to_tensorboard(tag='Safe Flight',
                                                                                           group=name_agent,
                                                                                           value=distance[name_agent],
                                                                                           index=epi_num[name_agent])
                                        agent[name_agent].network_model.log_to_tensorboard(tag='Episode Length',
                                                                                           group=name_agent,
                                                                                           value=len(data_tuple[name_agent]),
                                                                                           index=epi_num[name_agent])
                                        train_PPO(data_tuple[name_agent], algorithm_cfg, agent_this_drone,
                                                  algorithm_cfg.learning_rate, algorithm_cfg.input_size,
                                                  algorithm_cfg.gamma, epi_num[name_agent])
                                        c = agent_this_drone.network_model.get_vars()[15][0]
                                        agent_this_drone.network_model.log_to_tensorboard(tag='weight', group=name_agent,
                                                                                          value=c[0],
                                                                                          index=epi_num[name_agent])
                                        data_tuple[name_agent] = []
                                        epi_num[name_agent] += 1
                                        ret[name_agent] = 0
                                        distance[name_agent] = 0
                                        last_crash[name_agent] = 0
                                        reset_to_initial(level[name_agent], reset_array, client, vehicle_name=name_agent)
                                        current_state[name_agent] = agent[name_agent].get_state()
                                        old_posit[name_agent] = client.simGetVehiclePose(vehicle_name=name_agent)
                                        if epi_num[name_agent] % 100 == 0:
                                            agent_this_drone.network_model.save_network(algorithm_cfg.network_path,
                                                                                        epi_num[name_agent])
                                        if all(wait_for_others.values()):
                                            print('Communicating the weights and averaging them')
                                            communicate_across_agents(agent, name_agent_list, algorithm_cfg)
                                            for n in name_agent_list:
                                                wait_for_others[n] = False
                                else:
                                    current_state[name_agent] = new_state[name_agent]

                                time_exec = time.time() - start_time
                                gpu_memory, gpu_utilization, sys_memory = get_SystemStats(process, cfg.NVIDIA_GPU)

                                for i in range(0, len(gpu_memory)):
                                    tag_mem = 'GPU' + str(i) + '-Memory-GB'
                                    tag_util = 'GPU' + str(i) + 'Utilization-%'
                                    agent_this_drone.network_model.log_to_tensorboard(tag=tag_mem, group='SystemStats',
                                                                                      value=gpu_memory[i],
                                                                                      index=iter[name_agent])
                                    agent_this_drone.network_model.log_to_tensorboard(tag=tag_util, group='SystemStats',
                                                                                      value=gpu_utilization[i],
                                                                                      index=iter[name_agent])
                                agent_this_drone.network_model.log_to_tensorboard(tag='Memory-GB', group='SystemStats',
                                                                                  value=sys_memory,
                                                                                  index=iter[name_agent])

                                s_log = '{:<6s} - Level {:>2d} - Iter: {:>6d}/{:<5d} {:<8s}-{:>5s} lr: {:>1.8f} Ret = {:>+6.4f} Last Crash = {:<5d} t={:<1.3f} SF = {:<5.4f}  Reward: {:<+1.4f}  '.format(
                                    name_agent,
                                    int(level[name_agent]),
                                    iter[name_agent],
                                    epi_num[name_agent],
                                    action_word,
                                    action_type,
                                    algorithm_cfg.learning_rate,
                                    ret[name_agent],
                                    last_crash[name_agent],
                                    time_exec,
                                    distance[name_agent],
                                    reward)
                                if iter[name_agent] % print_interval == 0:
                                    print(s_log)
                                log_files[name_agent].write(s_log + '\n')
                                last_crash[name_agent] = last_crash[name_agent] + 1

                                if debug:
                                    cv2.imshow(name_agent, np.hstack((np.squeeze(current_state[name_agent], axis=0),
                                                                      np.squeeze(new_state[name_agent], axis=0))))
                                    cv2.waitKey(1)

                                if epi_num[name_agent] % algorithm_cfg.total_episodes == 0:
                                    print(automate)
                                    automate = False

                                iter[name_agent] += 1

                elif cfg.mode == 'infer':
                    # ------------------ Inference mode with additional plotting ------------------
                    iter_start_time = time.time()  # Start time for current iteration

                    agent_state = agent[name_agent].GetAgentState()
                    # Trigger return if collision OR total distance traveled is >= 50
                    if agent_state.has_collided or distance[name_agent] >= 50:
                        print('Drone collided or distance threshold reached')
                        print("Total distance traveled: ", np.round(distance[name_agent], 2))
                        # Hover at the current position for 5 seconds
                        print("Hovering at current position for 5 seconds...")
                        client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=5, vehicle_name=name_agent).join()
                        
                        # Retrace the path using the stored trajectory (in reverse order)
                        if len(traj_x) > 0:
                            print("Retracing path back to initial position...")
                            for i in range(len(traj_x) - 1, -1, -1):
                                target_x = traj_x[i]
                                target_y = traj_y[i]
                                target_z = traj_z[i]
                                print("Moving to waypoint:", target_x, target_y, target_z)
                                client.moveToPositionAsync(target_x, target_y, target_z, 5, vehicle_name=name_agent).join()
                                time.sleep(0.5)  # short pause between waypoints
                        else:
                            print("No trajectory data available, flying directly to initial position...")
                            init_x, init_y, init_z = 0, 0, 0
                            try:
                                init_x = initial_pos.position.x_val
                                init_y = initial_pos.position.y_val
                                init_z = initial_pos.position.z_val
                            except Exception:
                                pos_dict = initial_pos.get("position", initial_pos)
                                if "x_val" in pos_dict:
                                    init_x = pos_dict["x_val"]
                                    init_y = pos_dict["y_val"]
                                    init_z = pos_dict["z_val"]
                                elif "x" in pos_dict:
                                    init_x = pos_dict["x"]
                                    init_y = pos_dict["y"]
                                    init_z = pos_dict["z"]
                            client.moveToPositionAsync(init_x, init_y, init_z, 5, vehicle_name=name_agent).join()
                        
                        # Wait a moment to ensure arrival
                        time.sleep(2)
                        
                        # Hover at the initial position for 10 seconds
                        print("Hovering at initial position for 10 seconds...")
                        client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=10, vehicle_name=name_agent).join()
                        active = False

                        # Define the directory to save results
                        file_path = os.path.join(os.path.expanduser("~"), "Pictures", "DeepPPO_Results") + os.sep
                        if not os.path.exists(file_path):
                            os.makedirs(file_path)

                        # ✅ Save trajectory data (3D trajectory)
                        trajectory_df = pd.DataFrame({'x': traj_x, 'y': traj_y, 'z': traj_z})
                        trajectory_df.to_csv(file_path + '3d_trajectory.csv', index=False)

                        # ✅ Save altitude variation
                        altitude_df = pd.DataFrame({'iteration': range(len(altitude[name_agent])), 'altitude': altitude[name_agent]})
                        altitude_df.to_csv(file_path + 'altitude_variation.csv', index=False)

                        # ✅ Save navigation data
                        navigation_df = pd.DataFrame({'x': nav_x, 'y': nav_y})
                        navigation_df.to_csv(file_path + 'navigation.csv', index=False)

                        # ✅ Save iteration time per loop
                        time_df = pd.DataFrame({'iteration': range(1, len(time_exec_list) + 1), 'execution_time': time_exec_list})
                        time_df.to_csv(file_path + 'iteration_time.csv', index=False)

                        # ✅ Save cumulative distance over iterations
                        distance_df = pd.DataFrame({'iteration': range(1, len(distance_list) + 1), 'cumulative_distance': distance_list})
                        distance_df.to_csv(file_path + 'cumulative_distance.csv', index=False)

                        # ✅ Save yaw orientation over time
                        yaw_df = pd.DataFrame({'iteration': range(1, len(yaw_list) + 1), 'yaw': yaw_list})
                        yaw_df.to_csv(file_path + 'yaw_over_time.csv', index=False)

                        # ✅ Save action distribution
                        action_df = pd.DataFrame({'action': action_list})
                        action_df.to_csv(file_path + 'action_distribution.csv', index=False)

                        print("✅ All data saved as CSV files in:", file_path)
                        if len(time_exec_list) > 0:
                            print("Average iteration time: {:.3f} s".format(np.mean(time_exec_list)))
                    else:
                        posit[name_agent] = client.simGetVehiclePose(vehicle_name=name_agent)
                        distance[name_agent] = distance[name_agent] + np.linalg.norm(np.array(
                            [old_posit[name_agent].position.x_val - posit[name_agent].position.x_val,
                             old_posit[name_agent].position.y_val - posit[name_agent].position.y_val]))
                        altitude[name_agent].append(-posit[name_agent].position.z_val - f_z)

                        quat = (posit[name_agent].orientation.w_val, posit[name_agent].orientation.x_val,
                                posit[name_agent].orientation.y_val, posit[name_agent].orientation.z_val)
                        yaw = euler_from_quaternion(quat)[2]

                        x_val = posit[name_agent].position.x_val
                        y_val = posit[name_agent].position.y_val
                        z_val = posit[name_agent].position.z_val

                        nav_x.append(env_cfg.alpha * x_val + env_cfg.o_x)
                        nav_y.append(env_cfg.alpha * y_val + env_cfg.o_y)
                        nav.set_data(nav_x, nav_y)
                        nav_text.remove()
                        nav_text = ax_nav.text(25, 55, 'Distance: ' + str(np.round(distance[name_agent], 2)),
                                               style='italic',
                                               bbox={'facecolor': 'white', 'alpha': 0.5})

                        line_z.set_data(np.arange(len(altitude[name_agent])), altitude[name_agent])
                        ax_z.set_xlim(0, len(altitude[name_agent]))
                        #fig_z.canvas.draw()
                        #fig_z.canvas.flush_events()

                        # Update 3D trajectory plot
                        traj_x.append(x_val)
                        traj_y.append(y_val)
                        traj_z.append(z_val)
                        ax_3d.clear()
                        ax_3d.plot(traj_x, traj_y, traj_z, marker='o')
                        ax_3d.set_title("3D Trajectory")
                        ax_3d.set_xlabel("X")
                        ax_3d.set_ylabel("Y")
                        ax_3d.set_zlabel("Z")
                        #fig_3d.canvas.draw()
                        #fig_3d.canvas.flush_events()

                        # Update orientation (yaw) plot
                        yaw_list.append(yaw)
                        ax_yaw.clear()
                        ax_yaw.plot(range(1, len(yaw_list) + 1), yaw_list, marker='o')
                        ax_yaw.set_title("Orientation (Yaw) Over Time")
                        ax_yaw.set_xlabel("Iteration")
                        ax_yaw.set_ylabel("Yaw (radians)")
                        #fig_yaw.canvas.draw()
                        #fig_yaw.canvas.flush_events()

                        # ------------------ Compute Action First ------------------
                        current_state[name_agent] = agent[name_agent].get_state()
                        action, action_type = policy_REINFORCE(current_state[name_agent], agent[name_agent])
                        action_word = translate_action(action, algorithm_cfg.num_actions)
                        # ------------------------------------------------------------

                        # Update action distribution histogram using the newly computed action
                        action_list.append(action_word)
                        ax_action.clear()
                        unique_actions, counts = np.unique(action_list, return_counts=True)
                        ax_action.bar(unique_actions, counts)
                        ax_action.set_title("Action Distribution")
                        ax_action.set_xlabel("Action")
                        ax_action.set_ylabel("Frequency")
                        #fig_action.canvas.draw()
                        #fig_action.canvas.flush_events()

                        iter_end_time = time.time()
                        iter_time = iter_end_time - iter_start_time
                        time_exec_list.append(iter_time)
                        distance_list.append(distance[name_agent])

                        # Update the iteration time plot (printing disabled)
                        ax_time.clear()
                        ax_time.plot(range(1, len(time_exec_list) + 1), time_exec_list, marker='o', label="Iteration Time")
                        ax_time.set_title("Iteration Time (s)")
                        ax_time.set_xlabel("Iteration")
                        ax_time.set_ylabel("Time (s)")
                        ax_time.legend()
                        # Commented out to disable figure printing
                        # fig_time.canvas.draw()
                        # fig_time.canvas.flush_events()

                        # Update the cumulative distance plot (printing disabled)
                        ax_distance.clear()
                        ax_distance.plot(range(1, len(distance_list) + 1), distance_list, marker='o', label="Cumulative Distance")
                        ax_distance.set_title("Cumulative Distance")
                        ax_distance.set_xlabel("Iteration")
                        ax_distance.set_ylabel("Distance")
                        ax_distance.legend()
                        # Commented out to disable figure printing
                        # fig_distance.canvas.draw()
                        # fig_distance.canvas.flush_events()

                        agent[name_agent].take_action(action, algorithm_cfg.num_actions, Mode='static')
                        old_posit[name_agent] = posit[name_agent]

                        s_log = 'Position = ({:<3.2f},{:<3.2f}, {:<3.2f}) Orientation={:<1.3f} Predicted Action: {:<8s}  '.format(
                            x_val, y_val, z_val, yaw, action_word)
                        print(s_log)
                        log_files[name_agent].write(s_log + '\n')
                    iter += 1
            # End of if automate

        except Exception as e:
            if str(e) == 'cannot reshape array of size 1 into shape (0,0,3)':
                print('Recovering from AirSim error')
                client, old_posit, initZ = connect_drone(ip_address=cfg.ip_address, phase=cfg.mode,
                                                         num_agents=cfg.num_agents, client=client)
                time.sleep(2)
                agent[name_agent].client = client
                if cfg.mode == 'train':
                    wait_for_others[name_agent] = False
            else:
                print('------------- Error -------------')
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(exc_obj)
                automate = False
                print('Hit r and then backspace to start from this point')

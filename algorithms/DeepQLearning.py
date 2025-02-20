# Author: Aqeel Anwar(ICSRL)
# Created: 2/19/2020, 8:39 AM
# Email: aqeel.anwar@gatech.edu

import sys, cv2, time, psutil, numpy as np, pygame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D trajectory plot
import nvidia_smi
from network.agent import PedraAgent
from unreal_envs.initial_positions import *
from os import getpid
from network.Memory import Memory
from aux_functions import *
import os
from util.transformations import euler_from_quaternion
from configs.read_cfg import read_cfg, update_algorithm_cfg
import pandas as pd

def DeepQLearning(cfg, env_process, env_folder):
    # Read algorithm configuration
    algorithm_cfg = read_cfg(config_filename='configs/DeepQLearning.cfg', verbose=True)
    algorithm_cfg.algorithm = cfg.algorithm

    # Connect to Unreal Engine and get the drone handle: client
    client, old_posit, initZ = connect_drone(ip_address=cfg.ip_address, phase=cfg.mode, num_agents=cfg.num_agents)
    initial_pos = old_posit.copy()
    # Load the initial positions for the environment
    reset_array, reset_array_raw, level_name, crash_threshold = initial_positions(cfg.env_name, initZ, cfg.num_agents)

    # Initialize System Handlers
    process = psutil.Process(getpid())

    # Load PyGame Screen
    screen = pygame_connect(phase=cfg.mode)

    # Initialize basic figures (altitude and navigation) via your helper function
    fig_z = []
    fig_nav = []
    debug = False

    # Generate path where the weights will be saved
    cfg, algorithm_cfg = save_network_path(cfg=cfg, algorithm_cfg=algorithm_cfg)
    current_state = {}
    new_state = {}
    posit = {}
    agent = {}

    if cfg.mode == 'train':
        ReplayMemory = {}
        target_agent = {}

        name_agent = "drone0"
        print_orderly(name_agent, 40)
        agent[name_agent] = PedraAgent(algorithm_cfg, client, name='DQN', vehicle_name=name_agent)
        ReplayMemory[name_agent] = Memory(algorithm_cfg.buffer_len)
        target_agent[name_agent] = PedraAgent(algorithm_cfg, client, name='Target', vehicle_name=name_agent)
        current_state[name_agent] = agent[name_agent].get_state()

    elif cfg.mode == 'infer':
        print("This is folder")
        print(env_folder)
        name_agent = 'drone0'
        agent[name_agent] = PedraAgent(algorithm_cfg, client, name=name_agent + 'DQN', vehicle_name=name_agent)

        env_cfg = read_cfg(config_filename=os.path.join(env_folder, 'config.cfg'), verbose=True)
        print("This is config")
        print(env_cfg)

        # Basic navigation and altitude figures (from your helper function)
        nav_x = []
        nav_y = []
        altitude = {}
        altitude[name_agent] = []
        p_z, f_z, fig_z, ax_z, line_z, fig_nav, ax_nav, nav = initialize_infer(env_cfg=env_cfg, client=client,
                                                                                 env_folder=env_folder)
        nav_text = ax_nav.text(0, 0, '')

        # === Additional Figures to match DeepPPO plots ===
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

        # 3D Trajectory plot (X, Y, Z)
        fig_3d = plt.figure()
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        ax_3d.set_title("3D Trajectory")
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        traj_x = []  # list to store X coordinates
        traj_y = []  # list to store Y coordinates
        traj_z = []  # list to store Z coordinates

        # Orientation (Yaw) over time plot
        fig_yaw, ax_yaw = plt.subplots()
        ax_yaw.set_title("Orientation (Yaw) Over Time")
        ax_yaw.set_xlabel("Iteration")
        ax_yaw.set_ylabel("Yaw (radians)")
        yaw_list = []  # list to store yaw values

        # Action distribution histogram
        fig_action, ax_action = plt.subplots()
        ax_action.set_title("Action Distribution")
        ax_action.set_xlabel("Action")
        ax_action.set_ylabel("Frequency")
        action_list = []  # list to store predicted actions

        # Select initial position
        reset_to_initial(0, reset_array, client, vehicle_name=name_agent)
        old_posit[name_agent] = client.simGetVehiclePose(vehicle_name=name_agent)

        # Save the initial position for later return
        initial_pose = old_posit[name_agent]

    # Initialize common variables
    iter = 1
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

    hyphens = '-' * int((80 - len('Log files')) / 2)
    print(hyphens + ' ' + 'Log files' + ' ' + hyphens)

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
    log_path = os.path.join(algorithm_cfg.network_path, name_agent, cfg.mode + 'log.txt')
    print("Log path: ", log_path)
    log_files[name_agent] = open(log_path, 'w')

    print_orderly('Simulation begins', 80)

    while active:
        try:
            active, automate, algorithm_cfg, client = check_user_input(
                active, automate, agent[name_agent], client,
                old_posit[name_agent], initZ, fig_z, fig_nav, env_folder, cfg, algorithm_cfg)

            if automate:
                if cfg.mode == 'train':
                    # Training logic (unchanged in this example)
                    pass

                elif cfg.mode == 'infer':
                    # Inference phase
                    agent_state = agent[name_agent].GetAgentState()

                    # Trigger return if a collision occurs OR the drone has reached 50 units of distance
                    if agent_state.has_collided or distance[name_agent] >= 50:
                        print('Drone collided or distance threshold reached')
                        print("Total distance traveled: ", np.round(distance[name_agent], 2))
                        active = False

                        # Hover for 5 seconds
                        client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=5, vehicle_name=name_agent).join()
                        print("Drone is hovering for 5 seconds.")

                        # Retrace the path using the stored trajectory positions
                        if len(traj_x) > 0:
                            print("Retracing path back to initial position...")
                            for i in range(len(traj_x) - 1, -1, -1):
                                target_x = traj_x[i]
                                target_y = traj_y[i]
                                target_z = traj_z[i]
                                print("Moving to waypoint:", target_x, target_y, target_z)
                                client.moveToPositionAsync(target_x, target_y, target_z, 5, vehicle_name=name_agent).join()
                                time.sleep(0.5)
                        else:
                            print("No trajectory data available, moving directly to initial position...")
                            client.moveToPositionAsync(
                                initial_pose.position.x_val,
                                initial_pose.position.y_val,
                                initial_pose.position.z_val,
                                velocity=1.0,
                                vehicle_name=name_agent
                            ).join()
                        print("Drone has reached the initial position.")

                        # Hover for 10 seconds at the initial position
                        print("Hovering for 10 seconds...")
                        client.hoverAsync(vehicle_name=name_agent).join()
                        time.sleep(10)

                        # Land the drone
                        print("Landing the drone...")
                        client.landAsync(vehicle_name=name_agent).join()
                        print("Drone has landed. Ending simulation.")

                                                # Define the directory to save results
                        file_path = os.path.join(os.path.expanduser("~"), "Pictures", "DeepQLearning_Results") + os.sep
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

                    else:
                        # ==================== Inference Loop with Additional Plot Updates ====================
                        iter_start_time = time.time()

                        # Update state and compute distance/altitude
                        posit[name_agent] = client.simGetVehiclePose(vehicle_name=name_agent)
                        distance[name_agent] += np.linalg.norm(np.array([
                            old_posit[name_agent].position.x_val - posit[name_agent].position.x_val,
                            old_posit[name_agent].position.y_val - posit[name_agent].position.y_val
                        ]))
                        altitude[name_agent].append(-posit[name_agent].position.z_val - f_z)

                        # Compute orientation (yaw) from quaternion
                        quat = (posit[name_agent].orientation.w_val,
                                posit[name_agent].orientation.x_val,
                                posit[name_agent].orientation.y_val,
                                posit[name_agent].orientation.z_val)
                        yaw = euler_from_quaternion(quat)[2]

                        x_val = posit[name_agent].position.x_val
                        y_val = posit[name_agent].position.y_val
                        z_val = posit[name_agent].position.z_val

                        # Update navigation plot data
                        nav_x.append(env_cfg.alpha * x_val + env_cfg.o_x)
                        nav_y.append(env_cfg.alpha * y_val + env_cfg.o_y)
                        nav.set_data(nav_x, nav_y)
                        nav_text.remove()
                        nav_text = ax_nav.text(25, 55,
                                               'Distance: ' + str(np.round(distance[name_agent], 2)),
                                               style='italic',
                                               bbox={'facecolor': 'white', 'alpha': 0.5})

                        # Update altitude plot
                        line_z.set_data(np.arange(len(altitude[name_agent])), altitude[name_agent])
                        ax_z.set_xlim(0, len(altitude[name_agent]))
                        fig_z.canvas.draw()
                        fig_z.canvas.flush_events()

                        # ----- Additional Plot Updates -----
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
                        fig_3d.canvas.draw()
                        fig_3d.canvas.flush_events()

                        # Update orientation (yaw) plot
                        yaw_list.append(yaw)
                        ax_yaw.clear()
                        ax_yaw.plot(range(1, len(yaw_list) + 1), yaw_list, marker='o')
                        ax_yaw.set_title("Orientation (Yaw) Over Time")
                        ax_yaw.set_xlabel("Iteration")
                        ax_yaw.set_ylabel("Yaw (radians)")
                        fig_yaw.canvas.draw()
                        fig_yaw.canvas.flush_events()

                        # Compute predicted action
                        current_state[name_agent] = agent[name_agent].get_state()
                        action, action_type, algorithm_cfg.epsilon, qvals = policy(
                            1, current_state[name_agent], iter,
                            algorithm_cfg.epsilon_saturation,
                            'inference',
                            algorithm_cfg.wait_before_train,
                            algorithm_cfg.num_actions,
                            agent[name_agent])
                        action_word = translate_action(action, algorithm_cfg.num_actions)

                        # Update action distribution histogram
                        action_list.append(action_word)
                        ax_action.clear()
                        unique_actions, counts = np.unique(action_list, return_counts=True)
                        ax_action.bar(unique_actions, counts)
                        ax_action.set_title("Action Distribution")
                        ax_action.set_xlabel("Action")
                        ax_action.set_ylabel("Frequency")
                        fig_action.canvas.draw()
                        fig_action.canvas.flush_events()

                        # Take the action and update the old position
                        agent[name_agent].take_action(action, algorithm_cfg.num_actions, Mode='static')
                        old_posit[name_agent] = posit[name_agent]

                        s_log = 'Position = ({:<3.2f},{:<3.2f}, {:<3.2f}) Orientation={:<1.3f} Predicted Action: {:<8s}'.format(
                            x_val, y_val, z_val, yaw, action_word)
                        print(s_log)
                        log_files[name_agent].write(s_log + '\n')

                        # Update iteration timing and cumulative distance plots
                        iter_end_time = time.time()
                        iter_time = iter_end_time - iter_start_time
                        time_exec_list.append(iter_time)
                        distance_list.append(distance[name_agent])

                        ax_time.clear()
                        ax_time.plot(range(1, len(time_exec_list) + 1), time_exec_list, marker='o')
                        ax_time.set_title("Iteration Time (s)")
                        ax_time.set_xlabel("Iteration")
                        ax_time.set_ylabel("Time (s)")
                        fig_time.canvas.draw()
                        fig_time.canvas.flush_events()

                        ax_distance.clear()
                        ax_distance.plot(range(1, len(distance_list) + 1), distance_list, marker='o')
                        ax_distance.set_title("Cumulative Distance")
                        ax_distance.set_xlabel("Iteration")
                        ax_distance.set_ylabel("Distance")
                        fig_distance.canvas.draw()
                        fig_distance.canvas.flush_events()
                        # --------------------------------------------------------------------------------

                        # Increment iteration counter
                        iter += 1

        except Exception as e:
            if str(e) == 'cannot reshape array of size 1 into shape (0,0,3)':
                print('Recovering from AirSim error')
                client, old_posit, initZ = connect_drone(ip_address=cfg.ip_address, phase=cfg.mode,
                                                         num_agents=cfg.num_agents, client=client)
                agent[name_agent].client = client
            else:
                print('------------- Error -------------')
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(exc_obj)
                automate = False
                print('Hit r and then backspace to start from this point')

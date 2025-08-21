import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import meta_learning_base as mlb

class ModQuadEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, render_mode=None, max_steps=20000):
        super().__init__()
        self.sim_Flag = False
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.dt = 0.01
        self.g = 9.81
        self.robot = None
        self.target = None
        self.trajectory_step = None
        self.current_step = 0
        self.current_pos = None
        # Action space bounds based on observed data from replay buffer
        # Minimums: [-2.97384988 -0.0322127  -0.04703512 -0.57251428]
        # Maximums: [3.32227485e+00 4.76690636e-02 4.60323018e-02 3.87281920e-05]
        self.action_space = spaces.Box(
            low=np.array([2, -0.15, -0.39, -0.48], dtype=np.float32),
            high=np.array([5, 0.15, 0.39, 0.48], dtype=np.float32),
            shape=(4,), dtype=np.float32
        )  # wrench: 3 force, 3 torque
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )  # 1x(vel(3)+ang_vel(3)+quat(4))
        
        print('Connecting to CoppeliaSim...')
        mlb.sim.simxFinish(-1)  # Just in case, close all old connections
        self.client_id = mlb.sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        mlb.sim.simxStartSimulation(self.client_id, mlb.sim.simx_opmode_blocking)
        self.last_action = [0,0,0,0]
        self.last_action_taken = [0,0,0,0]

        
        # Create robot and target (desired box)
        if self.client_id != -1:
            print('Connected!')

            self.sim_Flag = True
            # print(self.client_id)
            self.robot = mlb.Robot(
                    'MultiRotor', self.client_id,
                    ['/propeller{}'.format(i+1) for i in range(8)],
                    mlb.PID_param(
                        mass=0.32, inertia=0.03,
                    KZ=(5.0, 3.5, 0.2),
                KX=(4.8, 5.8, 0.0),
                KY=(1.45, 2.0, 0.0),
                KR=(4.0, 2.0, 0.0),
                KP=(0.2, 0.15, 0.0),
                KYAW=(-1.0, -0.8, 0.0)
                )
                )
            self.target = mlb.Robot('DesiredBox', self.client_id)
            
        else:
            print("Not connected")
        
         
        self.robot_init_pos = self.robot.get_position()
        self.robot_init_ori = self.robot.get_orientation()
        self.robot_position_now = self.robot_init_pos

        obs = self._get_obs()
        return obs, {}

    def end_simulation(self):
        mlb.sim.simxStopSimulation(self.client_id, mlb.sim.simx_opmode_blocking)
        # self.robot.close_connection()
        time.sleep(1)

    def __reset(self, seed=None, options=None):
        super().reset(seed=seed)


        self.current_step = 0
        # Set fixed initial positions
        self.robot.set_position(self.robot_init_pos)
        self.robot.set_orientation(self.robot_init_ori)
        self.target.set_position(np.array([0.8, 0.0, 2.8]))
        self.target.set_orientation(np.array([0.0, 0.0, 0.0]))
        time.sleep(0.1)
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        # Concatenate robot and target state
        self.robot_position_now = self.robot.get_position()
        ang1 = self.robot.get_orientation()
        vel1, ang_vel1 = self.robot.get_velocity()
        quat1 = self.robot.get_quaternion()
        # pos2 = self.target.get_position()
        # ang2 = self.target.get_orientation()
        # vel2, ang_vel2 = self.target.get_velocity()
        obs = np.concatenate([vel1, ang_vel1, quat1]) #, pos2, ang2, vel2, ang_vel2])
        return obs.astype(np.float32)

    def step(self, action):

        """trajectory tracking is used to track the target trajectory and terminate the episode when the target trajectory is completed
        trajectory_step is the current step in the target trajectory
        trajectory_step is none by default """

        # pos = self.robot.get_position()  # Current x, y, z position
        crash = self.robot.send_4x1frpy_to_sim(action)
        time.sleep(self.dt)
        obs = self._get_obs()
        # new_pos = self.robot.get_position()
        # calculated_pos = pos + self.dt * obs[:3]


        # action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.trajectory_step is not None:
            # Calculate position error (desired - current) for x, y, z
            
            desired_pos = np.array([self.x[self.trajectory_step], self.y[self.trajectory_step], self.z[self.trajectory_step]])
            desired_ori = np.array([self.roll[self.trajectory_step], self.pitch[self.trajectory_step], self.yaw[self.trajectory_step]])

            # tracking_error = desired_pos - new_pos  # desired - current
            # calculation_error = (new_pos - calculated_pos)
            # print("Calculation error: ", np.linalg.norm(calculation_error))
            # print("Delta matrix: ", calculation_error)
            # Replace the first 3 elements (x, y, z) with the position error
            # obs = np.concatenate([pos_error, obs[3:]])

            # Reward: negative distance between robot and target positions
            # Write a better reward function that takes velocity as well
            # reward = -np.linalg.norm(obs[:3] - desired_pos) - np.linalg.norm(obs[3:6] - desired_ori)
    
            reward = -np.linalg.norm(desired_pos - obs[:3])
            # print(reward)
            terminated = self.trajectory_step + 1>= self.num_waypoints
        else:
            reward = 0
            terminated = False
        
        truncated = crash
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        # No-op for now (visualization is in CoppeliaSim)
        pass

    def close(self):
        if self.robot is not None:
            self.robot.close_connection()
        if self.target is not None:
            self.target.close_connection() 

    def get_replay_buffer(self):
        replay_buffer_collector = mlb.collect_dynamics_training_data(self.robot, self.target)
        return replay_buffer_collector.get_all()
    
    def run_gym_simulation_and_collect_data(self, takeoff_ht = 1.0, cut_at = 300, time_duration = 80):
        
        replay_buffer = mlb.SimpleReplayBuffer()
        start_pos = self.robot.get_position()
        print("start_pos: ", start_pos)
        trajectory_length, total_time, pos_traj, orient_traj = self.initialize_target_trajectory(traj = "random trajectory",
               start_pos = [start_pos[0], start_pos[1], takeoff_ht], position_change_scale=0.5, num_hover_points=2, num_waypoints=5, 
               time_step_duration=10)   
        # (self, traj, start_pos=[0.0, 0.0, 2.0], start_yaw=0.0, 
        # start_vel=[0.0, 0.0, 0.0], start_yaw_rate=0.0,
        # position_change_scale=1.0, fixed_pos_change_dist=True,
        # orientation_change_scale=0.1,
        # std_velocity_change=0.0,
        # std_angular_velocity_change=0.0,
        # std_acceleration_change=0.0,
        # std_angular_acceleration_change=0.0,
        # num_waypoints=20, 
        # num_hover_points=3,
        # time_step_duration=20,
        # num_samples=3):

        print(self.x, self.y, self.z)
        obs, _ = self.reset()    

        # current_pos = obs_p[:3]  # Current x, y, z position
        # desired_pos = np.array([self.x[self.trajectory_step], self.y[self.trajectory_step], self.z[self.trajectory_step]])
        # pos_error = desired_pos - current_pos  # desired - current
        # # Replace the first 3 elements (x, y, z) with the position error
        # obs = np.concatenate([pos_error, obs_p[3:]])
        terminated = False
        truncated = False
        self.trajectory_step = 2
        self.update_setpoint(self.trajectory_step)
        last_setpoint_set = time.time()
        simulation_start = time.time()

        # update_axes(axs, env.render(), ax_text, trial, steps_trial, all_rewards)
        while not terminated or truncated:

            if time.time() - last_setpoint_set > total_time/trajectory_length: #or (np.linalg.norm(np.array([self.x[self.trajectory_step], self.y[self.trajectory_step], self.z[self.trajectory_step]]) - self.robot.get_position()) < 0.05 and \
                # abs(self.yaw[self.trajectory_step] - self.robot.get_orientation()[2]) < 0.05) and \
                # np.linalg.norm(self.robot.get_velocity()[0]) < 0.1:

                # print("Reached the target trajectory")
                self.trajectory_step += 1
                self.update_setpoint(self.trajectory_step)
                last_setpoint_set = time.time()
                
                # print(traj_step)

            

            # --- Doing env step using the agent and adding to model dataset ---
            f, R_d = self.robot.get_geometric_attitude_control_input(np.array([self.x[self.trajectory_step], self.y[self.trajectory_step], self.z[self.trajectory_step]]),
                                                            mlb.euler2quat(self.roll[self.trajectory_step], self.pitch[self.trajectory_step], self.yaw[self.trajectory_step]),
                                                            np.array([self.roll[self.trajectory_step], self.pitch[self.trajectory_step], self.yaw[self.trajectory_step]]),
                                                            (np.array([self.dx[self.trajectory_step], self.dy[self.trajectory_step], self.dz[self.trajectory_step]]), 
                                                                    np.array([self.droll[self.trajectory_step], self.dpitch[self.trajectory_step], self.dyaw[self.trajectory_step]])),
                                                            (np.array([self.ddx[self.trajectory_step], self.ddy[self.trajectory_step], self.ddz[self.trajectory_step]]), 
                                                                np.array([self.ddroll[self.trajectory_step], self.ddpitch[self.trajectory_step], self.ddyaw[self.trajectory_step]])))
            
            """NOTE: flavor 1: take the desired thrust and roll, pitch, yaw angles as actions 
            -- learning-based control WILL NOT bypass geometric control """
            action = self.robot.get_geometric_attitude_control_input_as_actions(f, R_d)
            
            
            next_obs, reward, terminated, truncated, _ = self.step(action)
            print("action: ", action)

            replay_buffer.add(obs, action, next_obs, reward, terminated, truncated)
                
            # update_axes(
            #     axs, env.render(), ax_text, trial, steps_trial, all_rewards)
            
            obs = next_obs
            # if (np.linalg.norm(np.array([self.x[self.trajectory_step], self.y[self.trajectory_step], self.z[self.trajectory_step]]) - self.robot.get_position()) < 0.05 and \
            #     abs(self.yaw[self.trajectory_step] - self.robot.get_orientation()[2]) < 0.05) and \
            #     np.linalg.norm(self.robot.get_velocity()[0]) < 0.1:
            #     print("Reached the target trajectory")
            #     break
            if (time.time() - simulation_start>cut_at):
                break

        self.end_simulation()
        self.reset()

        return replay_buffer.get_all()
                
    def initialize_target_trajectory(self, traj, start_pos=[0.0, 0.0, 2.0], start_yaw=0.0, 
        start_vel=[0.0, 0.0, 0.0], start_yaw_rate=0.0,
        position_change_scale=1.0, fixed_pos_change_dist=True,
        orientation_change_scale=0.1,
        std_velocity_change=0.0,
        std_angular_velocity_change=0.0,
        std_acceleration_change=0.0,
        std_angular_acceleration_change=0.0,
        num_waypoints=20, 
        num_hover_points=3,
        time_step_duration=20,
        num_samples=3):
        """Initialize the target trajectory for the environment.
           Can initialize it as a predefined trajectory as well, like circle or square.
        """
        self.trajectory_step = 0
        if traj == "random trajectory":
            pos_traj, orient_traj, total_time = mlb.generate_training_trajectory(
                    start_pos=start_pos, start_yaw=start_yaw, 
                    start_vel=start_vel, start_yaw_rate=start_yaw_rate,
                    position_change_scale=position_change_scale, fixed_pos_change_dist=fixed_pos_change_dist,
                    orientation_change_scale=orientation_change_scale,
                    std_velocity_change=std_velocity_change,
                    std_angular_velocity_change=std_angular_velocity_change,
                    std_acceleration_change=std_acceleration_change,
                    std_angular_acceleration_change=std_angular_acceleration_change,
                    num_waypoints=num_waypoints, 
                    num_hover_points=num_hover_points,
                    time_step_duration=time_step_duration,
                    num_samples=num_samples)  # generates trajectory for the first 40 seconds
            self.x, self.y, self.z, self.dx, self.dy, self.dz, self.ddx, self.ddy, self.ddz = pos_traj
            self.roll, self.pitch, self.yaw, self.droll, self.dpitch, self.dyaw, self.ddroll, self.ddpitch, self.ddyaw = orient_traj
            self.num_waypoints = len(self.x)
        elif traj == "takeoff trajectory":
            pos_traj, orient_traj, total_time = mlb.generate_takeoff_trajectory(
                start_pos=[0.0, 0.0, 0.0], start_yaw=0.0, 
                start_vel=[0.0, 0.0, 0.0], start_yaw_rate=0.0,
                takeoff_height=2.0, takeoff_duration=10.0,
                hover_duration=5.0, num_waypoints=50, 
                num_hover_points=10, time_step_duration=0.2,
                num_samples=100)
            self.x, self.y, self.z, self.dx, self.dy, self.dz, self.ddx, self.ddy, self.ddz = pos_traj
            self.roll, self.pitch, self.yaw, self.droll, self.dpitch, self.dyaw, self.ddroll, self.ddpitch, self.ddyaw = orient_traj
            self.num_waypoints = len(self.x)

        return len(self.x), total_time, pos_traj, orient_traj
    
    def update_setpoint(self, traj_step):
        # self.target.set_position(np.array([2,2,2]))
        # print("setting setpoint to: ", self.x[traj_step], self.y[traj_step], self.z[traj_step])
        # self.target.set_position(np.array([self.x[traj_step], self.y[traj_step], self.z[traj_step]]))
        self.target.set_position(np.array([0,0 , self.z[traj_step]]))

        self.target.set_orientation(np.array([self.roll[traj_step], self.pitch[traj_step], self.yaw[traj_step]]))

    def pause_simulation(self):
        """Pause the simulation during planning/learning phases."""
        if self.robot:
            self.robot.pause_simulation()
        else:
            mlb.sim.simxPauseSimulation(self.client_id, mlb.sim.simx_opmode_blocking)
        # print("Simulation paused")

    def resume_simulation(self):
        """Resume the simulation after planning/learning phases."""
        if self.robot:
            self.robot.resume_simulation()
        else:
            mlb.sim.simxStartSimulation(self.client_id, mlb.sim.simx_opmode_blocking)
        # print("Simulation resumed")

    def error_based_setpoint_update(self, cut_at, time_duration = 80):
        
        replay_buffer = mlb.SimpleReplayBuffer()
        trajectory_length, total_time, pos_traj, orient_traj = self.initialize_target_trajectory(traj = "random trajectory")   

        obs, _ = self.reset()    
        
        terminated = False

        traj_step = 0 
        last_setpoint_set = time.time()
        simulation_start = time.time()

        # update_axes(axs, env.render(), ax_text, trial, steps_trial, all_rewards)
        while not terminated or truncated:

            if time.time() - last_setpoint_set > total_time/trajectory_length:
                self.update_setpoint(traj_step)
                last_setpoint_set = time.time()
                traj_step += 1
                # print(traj_step)

            

            # --- Doing env step using the agent and adding to model dataset ---
            f, R_d = self.robot.get_geometric_attitude_control_input(np.array([self.x[traj_step], self.y[traj_step], self.z[traj_step]]),
                                                            mlb.euler2quat(self.roll[traj_step], self.pitch[traj_step], self.yaw[traj_step]),
                                                            np.array([self.roll[traj_step], self.pitch[traj_step], self.yaw[traj_step]]),
                                                            (np.array([self.dx[traj_step], self.dy[traj_step], self.dz[traj_step]]), 
                                                                np.array([self.droll[traj_step], self.dpitch[traj_step], self.dyaw[traj_step]])),
                                                            (np.array([self.ddx[traj_step], self.ddy[traj_step], self.ddz[traj_step]]), 
                                                                np.array([self.ddroll[traj_step], self.ddpitch[traj_step], self.ddyaw[traj_step]])))
            
            """NOTE: flavor 1: take the desired thrust and roll, pitch, yaw angles as actions 
            -- learning-based control WILL NOT bypass geometric control """
            action = self.robot.get_geometric_attitude_control_input_as_actions(f, R_d)
            
            
            next_obs, reward, terminated, truncated, _ = self.step(action, traj_step)

            replay_buffer.add(obs, action, next_obs, reward, terminated, truncated)
                
            # update_axes(
            #     axs, env.render(), ax_text, trial, steps_trial, all_rewards)
            
            obs = next_obs

            if (time.time() - simulation_start>cut_at):
                break

        self.end_simulation()
        self.reset()

        return replay_buffer.get_all()
                
    def genarate_takeoff_trajectories(self, num_trajectories=10, trajectory_length=10):
        """Generate takeoff trajectories for the environment.
        """
        replay_buffer = mlb.SimpleReplayBuffer()
        trajectory_length, total_time, pos_traj, orient_traj = self.initialize_target_trajectory(traj = "random trajectory", position_change_scale=0.5, num_waypoints=40)   
        # (self, traj, start_pos=[0.0, 0.0, 2.0], start_yaw=0.0, 
        # start_vel=[0.0, 0.0, 0.0], start_yaw_rate=0.0,
        # position_change_scale=1.0, fixed_pos_change_dist=True,
        # orientation_change_scale=0.1,
        # std_velocity_change=0.0,
        # std_angular_velocity_change=0.0,
        # std_acceleration_change=0.0,
        # std_angular_acceleration_change=0.0,
        # num_waypoints=20, 
        # num_hover_points=3,
        # time_step_duration=20,
        # num_samples=3):

        obs, _ = self.reset()    
        terminated = False

        last_setpoint_set = time.time()
        simulation_start = time.time()

        # update_axes(axs, env.render(), ax_text, trial, steps_trial, all_rewards)
        while not (terminated or truncated):

            if time.time() - last_setpoint_set > total_time/trajectory_length or (np.linalg.norm(np.array([self.x[self.trajectory_step], self.y[self.trajectory_step], self.z[self.trajectory_step]]) - self.robot.get_position()) < 0.05 and \
                abs(self.yaw[self.trajectory_step] - self.robot.get_orientation()[2]) < 0.05) and \
                np.linalg.norm(self.robot.get_velocity()[0]) < 0.1:
                print("Reached the target trajectory")
                
                

                self.trajectory_step += 1
                self.update_setpoint(self.trajectory_step)
                last_setpoint_set = time.time()
                
                # print(traj_step)

            

            # --- Doing env step using the agent and adding to model dataset ---
            f, R_d = self.robot.get_geometric_attitude_control_input(np.array([self.x[self.trajectory_step], self.y[self.trajectory_step], self.z[self.trajectory_step]]),
                                                            mlb.euler2quat(self.roll[self.trajectory_step], self.pitch[self.trajectory_step], self.yaw[self.trajectory_step]),
                                                            np.array([self.roll[self.trajectory_step], self.pitch[self.trajectory_step], self.yaw[self.trajectory_step]]),
                                                            (np.array([self.dx[self.trajectory_step], self.dy[self.trajectory_step], self.dz[self.trajectory_step]]), 
                                                                    np.array([self.droll[self.trajectory_step], self.dpitch[self.trajectory_step], self.dyaw[self.trajectory_step]])),
                                                            (np.array([self.ddx[self.trajectory_step], self.ddy[self.trajectory_step], self.ddz[self.trajectory_step]]), 
                                                                np.array([self.ddroll[self.trajectory_step], self.ddpitch[self.trajectory_step], self.ddyaw[self.trajectory_step]])))
            
            """NOTE: flavor 1: take the desired thrust and roll, pitch, yaw angles as actions 
            -- learning-based control WILL NOT bypass geometric control """
            action = self.robot.get_geometric_attitude_control_input_as_actions(f, R_d)
            
            
            next_obs, reward, terminated, truncated, _ = self.step(action)

            replay_buffer.add(obs, action, next_obs, reward, terminated, truncated)
                
            # update_axes(
            #     axs, env.render(), ax_text, trial, steps_trial, all_rewards)
            
            obs = next_obs
            # if (np.linalg.norm(np.array([self.x[self.trajectory_step], self.y[self.trajectory_step], self.z[self.trajectory_step]]) - self.robot.get_position()) < 0.05 and \
            #     abs(self.yaw[self.trajectory_step] - self.robot.get_orientation()[2]) < 0.05) and \
            #     np.linalg.norm(self.robot.get_velocity()[0]) < 0.1:
            #     print("Reached the target trajectory")
            #     break
            if (time.time() - simulation_start>cut_at):
                break

        self.end_simulation()
        self.reset()

        return replay_buffer.get_all()

    def collect_takeoff_data(self, cut_at, time_duration = 80):
            
            replay_buffer = mlb.SimpleReplayBuffer()
            trajectory_length, total_time, pos_traj, orient_traj = self.initialize_target_trajectory(traj = "takeoff trajectory", start_pos = self.robot.get_position())   
            #     start_pos=[0.0, 0.0, 0.0], start_yaw=0.0, 
            #     start_vel=[0.0, 0.0, 0.0], start_yaw_rate=0.0,
            #     takeoff_height=2.0, takeoff_duration=10.0,
            #     hover_duration=5.0, num_waypoints=50, 
            #     num_hover_points=10, time_step_duration=0.2,
            #     num_samples=100)

            obs, _ = self.reset()    

            # current_pos = obs_p[:3]  # Current x, y, z position
            # desired_pos = np.array([self.x[self.trajectory_step], self.y[self.trajectory_step], self.z[self.trajectory_step]])
            # pos_error = desired_pos - current_pos  # desired - current
            # # Replace the first 3 elements (x, y, z) with the position error
            # obs = np.concatenate([pos_error, obs_p[3:]])
            terminated = False

            last_setpoint_set = time.time()
            simulation_start = time.time()

            # update_axes(axs, env.render(), ax_text, trial, steps_trial, all_rewards)
            while not terminated or truncated:

                if time.time() - last_setpoint_set > total_time/trajectory_length or (np.linalg.norm(np.array([self.x[self.trajectory_step], self.y[self.trajectory_step], self.z[self.trajectory_step]]) - self.robot.get_position()) < 0.05 and \
                    abs(self.yaw[self.trajectory_step] - self.robot.get_orientation()[2]) < 0.05) and \
                    np.linalg.norm(self.robot.get_velocity()[0]) < 0.1:
                    print("Reached the target trajectory")
                    
                    

                    self.trajectory_step += 1
                    self.update_setpoint(self.trajectory_step)
                    last_setpoint_set = time.time()
                    
                    # print(traj_step)

                

                # --- Doing env step using the agent and adding to model dataset ---
                f, R_d = self.robot.get_geometric_attitude_control_input(np.array([self.x[self.trajectory_step], self.y[self.trajectory_step], self.z[self.trajectory_step]]),
                                                                mlb.euler2quat(self.roll[self.trajectory_step], self.pitch[self.trajectory_step], self.yaw[self.trajectory_step]),
                                                                np.array([self.roll[self.trajectory_step], self.pitch[self.trajectory_step], self.yaw[self.trajectory_step]]),
                                                                (np.array([self.dx[self.trajectory_step], self.dy[self.trajectory_step], self.dz[self.trajectory_step]]), 
                                                                        np.array([self.droll[self.trajectory_step], self.dpitch[self.trajectory_step], self.dyaw[self.trajectory_step]])),
                                                                (np.array([self.ddx[self.trajectory_step], self.ddy[self.trajectory_step], self.ddz[self.trajectory_step]]), 
                                                                    np.array([self.ddroll[self.trajectory_step], self.ddpitch[self.trajectory_step], self.ddyaw[self.trajectory_step]])))
                
                """NOTE: flavor 1: take the desired thrust and roll, pitch, yaw angles as actions 
                -- learning-based control WILL NOT bypass geometric control """
                action = self.robot.get_geometric_attitude_control_input_as_actions(f, R_d)
                
                
                next_obs, reward, terminated, truncated, _ = self.step(action)

                replay_buffer.add(obs, action, next_obs, reward, terminated, truncated)
                    
                # update_axes(
                #     axs, env.render(), ax_text, trial, steps_trial, all_rewards)
                
                obs = next_obs
                # if (np.linalg.norm(np.array([self.x[self.trajectory_step], self.y[self.trajectory_step], self.z[self.trajectory_step]]) - self.robot.get_position()) < 0.05 and \
                #     abs(self.yaw[self.trajectory_step] - self.robot.get_orientation()[2]) < 0.05) and \
                #     np.linalg.norm(self.robot.get_velocity()[0]) < 0.1:
                #     print("Reached the target trajectory")
                #     break
                if (time.time() - simulation_start>cut_at):
                    break

            self.end_simulation()
            self.reset()

            return replay_buffer.get_all()
        
    def run_gym_simulation_and_collect_short_traj_data(self, takeoff_ht = 1.0, cut_at = 300, time_duration = 80):
            
        replay_buffer = mlb.SimpleReplayBuffer()
        start_pos = self.robot.get_position()
        print("start_pos: ", start_pos)
        trajectory_length, total_time, pos_traj, orient_traj = self.initialize_target_trajectory(traj = "random trajectory",
            start_pos = [start_pos[0], start_pos[1], takeoff_ht], position_change_scale=0.5, num_hover_points=2, num_waypoints=3, 
            time_step_duration=5)   
        # (self, traj, start_pos=[0.0, 0.0, 2.0], start_yaw=0.0, 
        # start_vel=[0.0, 0.0, 0.0], start_yaw_rate=0.0,
        # position_change_scale=1.0, fixed_pos_change_dist=True,
        # orientation_change_scale=0.1,
        # std_velocity_change=0.0,
        # std_angular_velocity_change=0.0,
        # std_acceleration_change=0.0,
        # std_angular_acceleration_change=0.0,
        # num_waypoints=20, 
        # num_hover_points=3,
        # time_step_duration=20,
        # num_samples=3):

        # print(self.x, self.y, self.z)
        obs, _ = self.reset()    

        # current_pos = obs_p[:3]  # Current x, y, z position
        # desired_pos = np.array([self.x[self.trajectory_step], self.y[self.trajectory_step], self.z[self.trajectory_step]])
        # pos_error = desired_pos - current_pos  # desired - current
        # # Replace the first 3 elements (x, y, z) with the position error
        # obs = np.concatenate([pos_error, obs_p[3:]])
        terminated = False
        truncated = False
        self.trajectory_step = 2
        self.update_setpoint(self.trajectory_step)
        last_setpoint_set = time.time()
        simulation_start = time.time()

        # update_axes(axs, env.render(), ax_text, trial, steps_trial, all_rewards)
        while not terminated or truncated:

            if time.time() - last_setpoint_set > total_time/trajectory_length or (np.linalg.norm(np.array([self.x[self.trajectory_step], self.y[self.trajectory_step], self.z[self.trajectory_step]]) - self.robot.get_position()) < 0.05 and \
                abs(self.yaw[self.trajectory_step] - self.robot.get_orientation()[2]) < 0.05) and \
                np.linalg.norm(self.robot.get_velocity()[0]) < 0.1:

                # print("Reached the target trajectory")
                self.trajectory_step += 1
                self.update_setpoint(self.trajectory_step)
                last_setpoint_set = time.time()
                
                # print(traj_step)

            

            # --- Doing env step using the agent and adding to model dataset ---
            f, R_d = self.robot.get_geometric_attitude_control_input(np.array([self.x[self.trajectory_step], self.y[self.trajectory_step], self.z[self.trajectory_step]]),
                                                            mlb.euler2quat(self.roll[self.trajectory_step], self.pitch[self.trajectory_step], self.yaw[self.trajectory_step]),
                                                            np.array([self.roll[self.trajectory_step], self.pitch[self.trajectory_step], self.yaw[self.trajectory_step]]),
                                                            (np.array([self.dx[self.trajectory_step], self.dy[self.trajectory_step], self.dz[self.trajectory_step]]), 
                                                                    np.array([self.droll[self.trajectory_step], self.dpitch[self.trajectory_step], self.dyaw[self.trajectory_step]])),
                                                            (np.array([self.ddx[self.trajectory_step], self.ddy[self.trajectory_step], self.ddz[self.trajectory_step]]), 
                                                                np.array([self.ddroll[self.trajectory_step], self.ddpitch[self.trajectory_step], self.ddyaw[self.trajectory_step]])))
            
            """NOTE: flavor 1: take the desired thrust and roll, pitch, yaw angles as actions 
            -- learning-based control WILL NOT bypass geometric control """
            action = self.robot.get_geometric_attitude_control_input_as_actions(f, R_d)
            
            
            next_obs, reward, terminated, truncated, _ = self.step(action)

            replay_buffer.add(obs, action, next_obs, reward, terminated, truncated)
                
            # update_axes(
            #     axs, env.render(), ax_text, trial, steps_trial, all_rewards)
            
            obs = next_obs
            # if (np.linalg.norm(np.array([self.x[self.trajectory_step], self.y[self.trajectory_step], self.z[self.trajectory_step]]) - self.robot.get_position()) < 0.05 and \
            #     abs(self.yaw[self.trajectory_step] - self.robot.get_orientation()[2]) < 0.05) and \
            #     np.linalg.norm(self.robot.get_velocity()[0]) < 0.1:
            #     print("Reached the target trajectory")
            #     break
            if (time.time() - simulation_start>cut_at):
                break

        self.end_simulation()
        self.reset()

        return replay_buffer.get_all()
    
    def run_random_all_simulation_and_collect_data(self, takeoff_ht = 1.0, cut_at = 300):
        import random

        replay_buffer = mlb.SimpleReplayBuffer()
        start_pos = self.robot.get_position()
        # print("start_pos: ", start_pos)
        trajectory_length, total_time, pos_traj, orient_traj = self.initialize_target_trajectory(traj = "random trajectory",
            start_pos=[start_pos[0], start_pos[1], takeoff_ht], start_yaw=0.0, 
            start_vel=[0.0, 0.0, 0.0], start_yaw_rate=0.0,
            position_change_scale=1.0, fixed_pos_change_dist=True,
            orientation_change_scale=0.1,
            std_velocity_change=random.uniform(0.0, 0.5),
            std_angular_velocity_change=random.uniform(0.0, 0.5),
            std_acceleration_change=random.uniform(0.0, 0.5),
            std_angular_acceleration_change=random.uniform(0.0, 0.5),
            num_waypoints= random.randint(5, 10), 
            num_hover_points=2,
            time_step_duration=5,
            num_samples=3)

        # print(self.x, self.y, self.z)
        obs, _ = self.reset()    

        # current_pos = obs_p[:3]  # Current x, y, z position
        # desired_pos = np.array([self.x[self.trajectory_step], self.y[self.trajectory_step], self.z[self.trajectory_step]])
        # pos_error = desired_pos - current_pos  # desired - current
        # # Replace the first 3 elements (x, y, z) with the position error
        # obs = np.concatenate([pos_error, obs_p[3:]])
        terminated = False
        truncated = False
        self.trajectory_step = 2
        self.update_setpoint(self.trajectory_step)
        last_setpoint_set = time.time()
        simulation_start = time.time()

        # update_axes(axs, env.render(), ax_text, trial, steps_trial, all_rewards)
        while not terminated or truncated:

            if time.time() - last_setpoint_set > total_time/trajectory_length or (np.linalg.norm(np.array([self.x[self.trajectory_step], self.y[self.trajectory_step], self.z[self.trajectory_step]]) - self.robot.get_position()) < 0.05 and \
                abs(self.yaw[self.trajectory_step] - self.robot.get_orientation()[2]) < 0.05) and \
                np.linalg.norm(self.robot.get_velocity()[0]) < 0.1:

                print("Reached the target trajectory")
                self.trajectory_step += 1
                self.update_setpoint(self.trajectory_step)
                last_setpoint_set = time.time()
                
                # print(traj_step)

            

            # --- Doing env step using the agent and adding to model dataset ---
            f, R_d = self.robot.get_geometric_attitude_control_input(np.array([self.x[self.trajectory_step], self.y[self.trajectory_step], self.z[self.trajectory_step]]),
                                                            mlb.euler2quat(self.roll[self.trajectory_step], self.pitch[self.trajectory_step], self.yaw[self.trajectory_step]),
                                                            np.array([self.roll[self.trajectory_step], self.pitch[self.trajectory_step], self.yaw[self.trajectory_step]]),
                                                            (np.array([self.dx[self.trajectory_step], self.dy[self.trajectory_step], self.dz[self.trajectory_step]]), 
                                                                    np.array([self.droll[self.trajectory_step], self.dpitch[self.trajectory_step], self.dyaw[self.trajectory_step]])),
                                                            (np.array([self.ddx[self.trajectory_step], self.ddy[self.trajectory_step], self.ddz[self.trajectory_step]]), 
                                                                np.array([self.ddroll[self.trajectory_step], self.ddpitch[self.trajectory_step], self.ddyaw[self.trajectory_step]])))
            
            """NOTE: flavor 1: take the desired thrust and roll, pitch, yaw angles as actions 
            -- learning-based control WILL NOT bypass geometric control """
            action = self.robot.get_geometric_attitude_control_input_as_actions(f, R_d)
            
            
            next_obs, reward, terminated, truncated, _ = self.step(action)

            replay_buffer.add(obs, action, next_obs, reward, terminated, truncated)
                
            # update_axes(
            #     axs, env.render(), ax_text, trial, steps_trial, all_rewards)
            
            obs = next_obs
            # if (np.linalg.norm(np.array([self.x[self.trajectory_step], self.y[self.trajectory_step], self.z[self.trajectory_step]]) - self.robot.get_position()) < 0.05 and \
            #     abs(self.yaw[self.trajectory_step] - self.robot.get_orientation()[2]) < 0.05) and \
            #     np.linalg.norm(self.robot.get_velocity()[0]) < 0.1:
            #     print("Reached the target trajectory")
            #     break
            if (time.time() - simulation_start>cut_at):
                break

        self.end_simulation()
        self.reset()

        return replay_buffer.get_all()

    def create_combined_replay_buffer(self, short_trajectory_replay_buffer, takeoff_replay_buffer, random_trajectory_replay_buffer, 
                                    short_traj_ratio=0.3, takeoff_ratio=0.3, random_traj_ratio=0.4):
        """
        Create a combined replay buffer by sampling from multiple replay buffers based on specified ratios.
        
        Args:
            short_trajectory_replay_buffer: Replay buffer from short trajectory data
            takeoff_replay_buffer: Replay buffer from takeoff data  
            random_trajectory_replay_buffer: Replay buffer from random trajectory data
            short_traj_ratio: Proportion of data to take from short trajectory buffer (default: 0.3)
            takeoff_ratio: Proportion of data to take from takeoff buffer (default: 0.3)
            random_traj_ratio: Proportion of data to take from random trajectory buffer (default: 0.4)
            
        Returns:
            combined_replay_buffer: SimpleReplayBuffer containing combined data
        """
        import random
        
        combined_replay_buffer = mlb.SimpleReplayBuffer()
        
        # Get lengths of each buffer
        short_traj_len = len(short_trajectory_replay_buffer)
        takeoff_len = len(takeoff_replay_buffer)
        random_traj_len = len(random_trajectory_replay_buffer)
        
        print(f"Buffer sizes:")
        print(f"  Short trajectory: {short_traj_len} transitions")
        print(f"  Takeoff: {takeoff_len} transitions")
        print(f"  Random trajectory: {random_traj_len} transitions")
        
        # Calculate how many samples to take from each buffer based on ratios
        # Ratios can be any positive numbers - they don't need to sum to 1
        short_traj_samples = int(short_traj_ratio * short_traj_len)
        takeoff_samples = int(takeoff_ratio * takeoff_len)
        random_traj_samples = int(random_traj_ratio * random_traj_len)
        
        # Adjust to ensure we don't exceed available data
        short_traj_samples = min(short_traj_samples, short_traj_len)
        takeoff_samples = min(takeoff_samples, takeoff_len)
        random_traj_samples = min(random_traj_samples, random_traj_len)
        
        print(f"\nSampling strategy:")
        print(f"  Short trajectory: {short_traj_samples} samples (ratio: {short_traj_ratio:.2f})")
        print(f"  Takeoff: {takeoff_samples} samples (ratio: {takeoff_ratio:.2f})")
        print(f"  Random trajectory: {random_traj_samples} samples (ratio: {random_traj_ratio:.2f})")
        print(f"  Total combined: {short_traj_samples + takeoff_samples + random_traj_samples} samples")
        
        # Sample from short trajectory buffer
        if short_traj_samples > 0:
            short_traj_indices = random.sample(range(short_traj_len), short_traj_samples)
            for idx in short_traj_indices:
                combined_replay_buffer.add(*short_trajectory_replay_buffer[idx])
        
        # Sample from takeoff buffer
        if takeoff_samples > 0:
            takeoff_indices = random.sample(range(takeoff_len), takeoff_samples)
            for idx in takeoff_indices:
                combined_replay_buffer.add(*takeoff_replay_buffer[idx])
        
        # Sample from random trajectory buffer
        if random_traj_samples > 0:
            random_traj_indices = random.sample(range(random_traj_len), random_traj_samples)
            for idx in random_traj_indices:
                combined_replay_buffer.add(*random_trajectory_replay_buffer[idx])
        
        print(f"\nCombined replay buffer created with {len(combined_replay_buffer)} total transitions")
        
        return combined_replay_buffer.get_all()
    



            
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
        self.current_step = 0
        # Action space bounds based on observed data from replay buffer
        # Minimums: [-2.97384988 -0.0322127  -0.04703512 -0.57251428]
        # Maximums: [3.32227485e+00 4.76690636e-02 4.60323018e-02 3.87281920e-05]
        self.action_space = spaces.Box(
            low=np.array([-3.0, -0.05, -0.05, -0.6], dtype=np.float32),
            high=np.array([3.5, 0.05, 0.05, 0.0], dtype=np.float32),
            shape=(4,), dtype=np.float32
        )  # wrench: 3 force, 3 torque
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )  # 1x(pos(3)+ang(3)+vel(3)+ang_vel(3))
        
        print('Connecting to CoppeliaSim...')
        mlb.sim.simxFinish(-1)  # Just in case, close all old connections
        self.client_id = mlb.sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        mlb.sim.simxStartSimulation(self.client_id, mlb.sim.simx_opmode_blocking)
        

        
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
                KR=(16.0, 7.0, 0.0),
                KP=(1.25, 0.6, 0.0),
                KYAW=(-1.0, -0.8, 0.0)
                )
                )
            self.target = mlb.Robot('DesiredBox', self.client_id)
            
        else:
            print("Not connected")
        
         
        self.robot_init_pos = self.robot.get_position()
        self.robot_init_ori = self.robot.get_orientation()

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
        pos1 = self.robot.get_position()
        ang1 = self.robot.get_orientation()
        vel1, ang_vel1 = self.robot.get_velocity()
        # pos2 = self.target.get_position()
        # ang2 = self.target.get_orientation()
        # vel2, ang_vel2 = self.target.get_velocity()
        obs = np.concatenate([pos1, ang1, vel1, ang_vel1]) #, pos2, ang2, vel2, ang_vel2])
        return obs.astype(np.float32)

    def step(self, action, traj_step = None):

        """trajectory tracking is used to track the target trajectory and terminate the episode when the target trajectory is completed
        traj_step is the current step in the target trajectory
        traj_step is none by default """

        if traj_step is not None:
            target_pos = np.array([self.x[traj_step], self.y[traj_step], self.z[traj_step]])
            target_ori = np.array([self.roll[traj_step], self.pitch[traj_step], self.yaw[traj_step]])
            target_vel = np.array([self.dx[traj_step], self.dy[traj_step], self.dz[traj_step]])
            target_ang_vel = np.array([self.droll[traj_step], self.dpitch[traj_step], self.dyaw[traj_step]])

        # action = np.clip(action, self.action_space.low, self.action_space.high)
        
        crash = self.robot.send_4x1frpy_to_sim(action)
        time.sleep(self.dt)
        obs = self._get_obs()
        # Reward: negative distance between robot and target positions
        # Write a better reward function that takes velocity as well
        if traj_step is not None:
            reward = -np.linalg.norm(obs[:3] - target_pos) - np.linalg.norm(obs[3:6] - target_ori)
            terminated = traj_step + 1>= self.num_waypoints
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
    
    def run_gym_simulation_and_collect_data(self, cut_at, time_duration = 80):
        
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
                


    def initialize_target_trajectory(self, traj, start_pos=[0.0, 0.0, 1.0], start_yaw=0.0, 
    start_vel=[0.0, 0.0, 0.0], start_yaw_rate=0.0,
    std_position_change=0.2,
    std_orientation_change=0.1,
    std_velocity_change=0.05,
    std_angular_velocity_change=0.0,
    std_acceleration_change=0.0,
    std_angular_acceleration_change=0.0,
    num_waypoints=20, 
    num_hover_points=3,
    time_step_duration=10):
        """Initialize the target trajectory for the environment.
           Can initialize it as a predefined trajectory as well, like circle or square.
        """
        if traj == "random trajectory":
            pos_traj, orient_traj, total_time = mlb.generate_training_trajectory(start_pos=start_pos, start_yaw=start_yaw, 
    start_vel=start_vel, start_yaw_rate=start_yaw_rate,
    std_position_change=std_position_change,
    std_orientation_change=std_orientation_change,
    std_velocity_change=std_velocity_change,
    std_angular_velocity_change=std_angular_velocity_change,
    std_acceleration_change=std_acceleration_change,
    std_angular_acceleration_change=std_angular_acceleration_change,
    num_waypoints=num_waypoints, 
    num_hover_points=num_hover_points,
    time_step_duration=time_step_duration)  # generates trajectory for the first 40 seconds
            self.x, self.y, self.z, self.dx, self.dy, self.dz, self.ddx, self.ddy, self.ddz = pos_traj
            self.roll, self.pitch, self.yaw, self.droll, self.dpitch, self.dyaw, self.ddroll, self.ddpitch, self.ddyaw = orient_traj
            self.num_waypoints = len(self.x)

        return len(self.x), total_time, pos_traj, orient_traj
    
    def update_setpoint(self, traj_step):
        # self.target.set_position(np.array([2,2,2]))
        print("setting setpoint to: ", self.x[traj_step], self.y[traj_step], self.z[traj_step])
        self.target.set_position(np.array([self.x[traj_step], self.y[traj_step], self.z[traj_step]]))
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
                








            
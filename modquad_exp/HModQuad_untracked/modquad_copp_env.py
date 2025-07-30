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
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(4,), dtype=np.float32
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
                        KZ=(5.0, 3.5, 0.0),
                        KX=(2.0, 3.0, 0.0),
                        KY=(0.2, 0.6, 0.0),
                        KR=(1.5, 0.8, 0.0),
                        KP=(0.8, 0.6, 0.0),
                        KYAW=(-0.6, -0.5, 0.0)
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
        
        self.robot.send_4x1ftau_to_sim(action)
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
        
        truncated = self.robot.crash_check()
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
            action = self.robot.get_geometric_attitude_control_output(np.array([self.x[traj_step], self.y[traj_step], self.z[traj_step]]),
                                                            mlb.euler2quat(self.roll[traj_step], self.pitch[traj_step], self.yaw[traj_step]),
                                                            np.array([self.roll[traj_step], self.pitch[traj_step], self.yaw[traj_step]]),
                                                            (np.array([self.dx[traj_step], self.dy[traj_step], self.dz[traj_step]]), 
                                                                np.array([self.droll[traj_step], self.dpitch[traj_step], self.dyaw[traj_step]])),
                                                            (np.array([self.ddx[traj_step], self.ddy[traj_step], self.ddz[traj_step]]), 
                                                                np.array([self.ddroll[traj_step], self.ddpitch[traj_step], self.ddyaw[traj_step]])))
            
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
                


    def initialize_target_trajectory(self, traj):
        """Initialize the target trajectory for the environment.
           Can initialize it as a predefined trajectory as well, like circle or square.
        """
        if traj == "random trajectory":
            pos_traj, orient_traj, total_time = mlb.generate_training_trajectory()  # generates trajectory for the first 40 seconds
            self.x, self.y, self.z, self.dx, self.dy, self.dz, self.ddx, self.ddy, self.ddz = pos_traj
            self.roll, self.pitch, self.yaw, self.droll, self.dpitch, self.dyaw, self.ddroll, self.ddpitch, self.ddyaw = orient_traj
            self.num_waypoints = len(self.x)

        return len(self.x), total_time, pos_traj, orient_traj
    
    def update_setpoint(self, traj_step):
        self.target.set_position(np.array([self.x[traj_step], self.y[traj_step], self.z[traj_step]]))
        self.target.set_orientation(np.array([self.roll[traj_step], self.pitch[traj_step], self.yaw[traj_step]]))








            
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
            low=-10.0, high=10.0, shape=(6,), dtype=np.float32
        )  # wrench: 3 force, 3 torque
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32
        )  # 2x(pos(3)+ang(3)+vel(3)+ang_vel(3))
        
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
        pos2 = self.target.get_position()
        ang2 = self.target.get_orientation()
        vel2, ang_vel2 = self.target.get_velocity()
        obs = np.concatenate([pos1, ang1, vel1, ang_vel1, pos2, ang2, vel2, ang_vel2])
        return obs.astype(np.float32)

    def step(self, action):
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        self.robot.send_actions_to_sim(action)
        time.sleep(self.dt)
        obs = self._get_obs()
        # Reward: negative distance between robot and target positions
        # Write a better reward function that takes velocity as well
        reward = -np.linalg.norm(obs[:3] - obs[12:15])
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
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
    
    def run_gym_simulation(self, cut_at= 60, time_duration = 80):
        
        simulation_start = time.time()
        buff = mlb.SimpleReplayBuffer()
        while time.time() - simulation_start < 1:
            self.robot.get_position()
            self.target.get_position()
            time.sleep(0.01)

        pos_traj, orient_traj = mlb.generate_training_trajectory()  # generates trajectory for the first 40 seconds
        x, y, z, dx, dy, dz, ddx, ddy, ddz = pos_traj
        roll, pitch, yaw, droll, dpitch, dyaw, ddroll, ddpitch, ddyaw = orient_traj

        for i in range(len(x)):
            time_start = time.time()
            self.target.set_position(np.array([x[i], y[i], z[i]]))
            self.target.log_position.append(np.array([x[i], y[i], z[i]]))
            self.target.set_orientation(np.array([roll[i], pitch[i], yaw[i]]))
            self.target.log_angles.append(np.array([roll[i], pitch[i], yaw[i]]))
            self.target.log_time.append(time.time() - simulation_start)
            # time.sleep(0.01)
            obs = self._get_obs()
            action = self.robot.get_pid_controller_actions(
                        np.array([x[i], y[i], z[i]]),
                        mlb.euler2quat(roll[i], pitch[i], yaw[i]),
                        np.array([roll[i], pitch[i], yaw[i]]),
                        (np.array([dx[i], dy[i], dz[i]]), np.array([droll[i], dpitch[i], dyaw[i]])),
                        (np.array([ddx[i], ddy[i], ddz[i]]), np.array([ddroll[i], ddpitch[i], ddyaw[i]]))
                        )
            nex_obs, reward, terminated, truncated, info = self.step(action)
            buff.add(obs, action, nex_obs, terminated, truncated)
            self.robot.log_time.append(time.time() - simulation_start)


            while time.time() - time_start < time_duration/len(x):
                action = self.robot.get_pid_controller_actions(
                        np.array([x[i], y[i], z[i]]),
                        mlb.euler2quat(roll[i], pitch[i], yaw[i]),
                        np.array([roll[i], pitch[i], yaw[i]]),
                        (np.array([dx[i], dy[i], dz[i]]), np.array([droll[i], dpitch[i], dyaw[i]])),
                        (np.array([ddx[i], ddy[i], ddz[i]]), np.array([ddroll[i], ddpitch[i], ddyaw[i]]))
                        )
                nex_obs, reward, terminated, truncated, info = self.step(action)
                buff.add(obs, action, nex_obs, terminated, truncated)
                obs = nex_obs
                self.robot.log_time.append(time.time() - simulation_start)
                # if (time.time() - simulation_start>cut_at):
                #     break
                
        print(buff.__len__)
                
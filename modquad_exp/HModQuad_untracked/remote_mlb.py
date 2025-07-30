import sim  # legacy API
import time
import meta_learning_base as mlb
print('Connecting to CoppeliaSim...')
sim.simxFinish(-1)  # Just in case, close all old connections
client_id = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

def connect_run_disconnect():

    # print('Connecting to CoppeliaSim...')
    # sim.simxFinish(-1)  # Just in case, close all old connections
    # client_id = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    sim.simxStartSimulation(client_id, sim.simx_opmode_blocking)

    try:
        if client_id != -1:
            print('Connected!')

            # Start the simulation
            # sim.simxStartSimulation(client_id, sim.simx_opmode_blocking)
            # print('Simulation started.')

            r1 = mlb.Robot(
                'MultiRotor', client_id,
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
            d1 = mlb.Robot('DesiredBox', client_id)
            g = 9.81
            
                
            replay_buffer = mlb.collect_dynamics_training_data(r1, d1, cut_at=60)

                

            # Instead of inline plotting, call the function:
            # plot_results(r1, d1)
            
            # Let it run for 5 seconds
            time.sleep(5)

            # Stop the simulation
            sim.simxStopSimulation(client_id, sim.simx_opmode_blocking)

            print('Simulation stopped.')
    except KeyboardInterrupt:
        sim.simxStopSimulation(client_id, sim.simx_opmode_blocking)
        print('Simulation stopped.')

        time.sleep(5)
        # Close the connection
        r1.close_connection()



# print(client_id)
# sim.simxStartSimulation(client_id, sim.simx_opmode_oneshot)
# print("Simulation started automatically")

        
#         # Wait a moment for simulation to initialize
# time.sleep(0.5)
connect_run_disconnect()
print("run 1 done")
time.sleep(5)
print("running run 2")
connect_run_disconnect()

# if __name__ == "__main__":

#     # print('Connecting to CoppeliaSim...')
#     # sim.simxFinish(-1)  # Just in case, close all old connections
#     # client_id = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
#     # print(client_id)
#     # sim.simxStartSimulation(client_id, sim.simx_opmode_oneshot)
#     # print("Simulation started automatically")
#     connect_run_disconnect()
#     # print("run 1 done")
#     # time.sleep(5)
#     # print("running run 2")
#     # connect_run_disconnect()

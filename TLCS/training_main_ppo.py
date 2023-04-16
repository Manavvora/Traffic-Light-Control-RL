from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
from shutil import copyfile
import numpy as np

from training_simulation_ppo import Simulation_PPO
from generator import TrafficGenerator
from rollout_buffer import RolloutBuffer
from model_ppo import TrainModel_PPO
from visualization import Visualization
from utils_ppo import import_train_configuration, set_sumo, set_train_path


if __name__ == "__main__":

    config = import_train_configuration(config_file='training_settings_ppo.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    Model = TrainModel_PPO(
        config['learning_rate_actor'], 
        config['learning_rate_critic'],
        input_dim=config['num_states'], 
        output_dim=config['num_actions'],
        eps_clip=config['eps_clip']
    )

    Buffer = RolloutBuffer()

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    Visualization = Visualization(
        path, 
        dpi=96
    )
        
    Simulation = Simulation_PPO(
        Model,
        config['opt_epochs'],
        Buffer,
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions']
    )
    
    counter = 0
    timestamp_start = datetime.datetime.now()
    num_runs = 2
    reward_multiple_runs = np.zeros((num_runs,config['total_episodes']))
    return_multiple_runs = np.zeros((num_runs,config['total_episodes']))
    delay_multiple_runs = np.zeros((num_runs,config['total_episodes']))
    queue_length_multiple_runs = np.zeros((num_runs,config['total_episodes']))
    

    for n_run in range(num_runs):
        print(f"Starting run number {n_run+1}")
        print("-"*25)
        episode = 0
        while episode < config['total_episodes']:
            print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
            simulation_time, training_time, episode_reward, episode_return, episode_cumulative_delay, episode_queue_length = Simulation.run(episode)  # run the simulation
            print(episode_reward[counter])
            reward_multiple_runs[n_run,episode] = episode_reward[counter]
            return_multiple_runs[n_run,episode] = episode_return[counter]
            delay_multiple_runs[n_run,episode] = episode_cumulative_delay[counter]
            queue_length_multiple_runs[n_run,episode] = episode_queue_length[counter]
            print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
            episode += 1
            counter += 1
    
    print(reward_multiple_runs)
    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    Model.save_actor_model(path)

    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))

    Visualization.save_data_and_plot(data=reward_multiple_runs, filename='reward_ppo', xlabel='Episode', ylabel='Cumulative negative reward')
    Visualization.save_data_and_plot(data=return_multiple_runs, filename='return_ppo', xlabel='Episode', ylabel='Discounted return')
    Visualization.save_data_and_plot(data=delay_multiple_runs, filename='delay_ppo', xlabel='Episode', ylabel='Cumulative delay (s)')
    Visualization.save_data_and_plot(data=queue_length_multiple_runs, filename='queue_ppo', xlabel='Episode', ylabel='Average queue length (vehicles)')
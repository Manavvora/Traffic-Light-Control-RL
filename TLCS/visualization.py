import matplotlib.pyplot as plt
import os
import numpy as np
from utils_ppo import import_train_configuration
from testing_simulation import Simulation 
# from training_simulation import Simulation as Simulation_train

config = import_train_configuration(config_file='training_settings_ppo.ini')

class Visualization:
    def __init__(self, path, dpi):
            self._path = path
            self._dpi = dpi


    def save_data_and_plot(self, data, filename, xlabel, ylabel):
        """
        Produce a plot of performance of the agent over the session and save the relative data to txt
        """
        min_val = np.min(data)
        max_val = np.max(data)
        mean = np.mean(data,axis=0)
        std = np.std(data,axis=0)
        episode_array = np.arange(config['total_episodes']) + 1
        plt.rcParams.update({'font.size': 24})  # set bigger font size

        plt.plot(episode_array, mean, label='Average')
        plt.fill_between(episode_array, mean-std, mean+std, alpha=.2, label=r'1-$\sigma$ Error')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend()
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close("all")

        with open(os.path.join(self._path, 'plot_'+filename + '_data.txt'), "w") as file:
            for value in data:
                    file.write("%s\n" % value)

    def save_data_and_plot_test(self, data, filename, xlabel, ylabel, x_data):
        """
        Produce a plot of performance of the agent over the session and save the relative data to txt
        """
        min_val = np.min(data)
        max_val = np.max(data)
        # mean = np.mean(data,axis=0)
        # std = np.std(data,axis=0)
        # step_array = np.arange(config['max_steps']) + 1
        plt.rcParams.update({'font.size': 24})  # set bigger font size

        plt.plot(x_data, data, label='Plot for a fixed route-file')
        # plt.fill_between(step_array, mean-std, mean+std, alpha=.2, label=r'1-$\sigma$ Error')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend()
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close("all")

        with open(os.path.join(self._path, 'plot_'+filename + '_data.txt'), "w") as file:
            for value in data:
                    file.write("%s\n" % value)
    
    def save_data_and_plot_test_reward(self, data, filename, xlabel, ylabel, x_data):
        """
        Produce a plot of performance of the agent over the session and save the relative data to txt
        """
        min_val = np.min(data)
        max_val = np.max(data)
        # mean = np.mean(data,axis=0)
        # std = np.std(data,axis=0)
        # step_array = np.arange(len(Simulation._reward_episode)) + 1
        plt.rcParams.update({'font.size': 24})  # set bigger font size

        plt.plot(x_data, data, label='Reward for a fixed route-file')
        # plt.fill_between(step_array, mean-std, mean+std, alpha=.2, label=r'1-$\sigma$ Error')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend()
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close("all")

        with open(os.path.join(self._path, 'plot_'+filename + '_data.txt'), "w") as file:
            for value in data:
                    file.write("%s\n" % value)
    
    def save_data_and_plot_test_policy(self, data, filename, xlabel, ylabel,x_data):
        """
        Produce a plot of performance of the agent over the session and save the relative data to txt
        """
        min_val = np.min(data)
        max_val = np.max(data)
        # mean = np.mean(data,axis=0)
        # std = np.std(data,axis=0)
        # step_array = np.arange(len(Simulation._policy)) + 1
        plt.rcParams.update({'font.size': 24})  # set bigger font size

        plt.plot(x_data, data, label='Policy for a fixed route-file')
        # plt.fill_between(step_array, mean-std, mean+std, alpha=.2, label=r'1-$\sigma$ Error')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend()
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close("all")

        with open(os.path.join(self._path, 'plot_'+filename + '_data.txt'), "w") as file:
            for value in data:
                    file.write("%s\n" % value)
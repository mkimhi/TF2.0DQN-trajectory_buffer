from datetime import datetime
import time
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd



class SummariesCollector:
    def __init__(self, summaries_dir,name,config):
        self.config= config
        self.completion_reward = self.config['general']['completion_reward']
        self.dir=summaries_dir
        self.graph_prefix=config['general']['algorithm']+datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M')
        self.name =name+'_'+ self.graph_prefix+'_'+'.csv'
        self.init_titles()
        self.color = 'b' if (name == "Frozen_lake_Trajectory_buffer") else 'r'

    def init_titles(self):
        df = pd.DataFrame({'cycle':[], 'reward': [], 'avg_episode_len': []})
        df2 = pd.DataFrame({'cycle': [], 'reward': [], 'avg_episode_len': [], 'loss': []})
        train_path = os.path.join(self.dir, 'train')
        test_path = os.path.join(self.dir, 'test')
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        df2.to_csv(train_path + '\\' + self.name, mode='a', header=True, index=False)
        df2.to_csv(test_path + '\\' + self.name, mode='a', header=True, index=False)


    def write_summaries(self, prefix, cycle_num,reward, avg_episode_len,loss=None):
        if (prefix == 'train'):
            df = pd.DataFrame({'cycle':[cycle_num] , 'reward': [reward], 'avg_episode_len':[avg_episode_len],'loss':[loss] })
        else:
            df = pd.DataFrame(
                {'cycle': [cycle_num], 'reward': [reward], 'avg_episode_len': [avg_episode_len]})
        df.to_csv(os.path.join(self.dir,prefix)+'\\'+self.name, mode='a', header=False, index=False) #orgenize

    # use once in 1000 episodes
    def read_summaries(self,prefix): #and plot
        fig_idx = 1 if (prefix == 'train') else 4
        df=pd.read_csv(os.path.join(self.dir, prefix,self.name))
        sns.set()
        plt.figure(fig_idx)
        plt.title(prefix + " reward")
        plt.xlabel("LR: " + str(self.config['policy_network']['learn_rate'])+ " gamma: "+str(self.config['general']['gamma']))
        plt.plot(df.cycle, df.reward, color=self.color)
        plt.savefig(self.dir +"//"+ self.graph_prefix+' '+prefix+' reward graph.png')
        #plt.close(fig_idx)

        plt.figure(fig_idx+1)
        plt.title(prefix + " avg_episode_len")
        plt.xlabel("LR: " + str(self.config['policy_network']['learn_rate']) + " gamma: " + str(
            self.config['general']['gamma']))
        plt.plot(df.cycle, df.avg_episode_len,color=self.color)
        plt.savefig(self.dir + "//" + self.graph_prefix + ' ' + prefix + ' episode lenght graph.png')
        #plt.close(fig_idx+1)

        plt.figure(fig_idx+2)
        plt.title(prefix + " loss")
        plt.xlabel("LR: " + str(self.config['policy_network']['learn_rate']) + " gamma: " + str(
            self.config['general']['gamma']))
        plt.plot(df.cycle, df.loss,color=self.color)
        plt.savefig(self.dir + "//" + self.graph_prefix + ' ' + prefix + ' loss graph.png')
        #plt.close(fig_idx+2)

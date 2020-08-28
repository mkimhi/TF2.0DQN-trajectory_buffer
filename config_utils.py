import os
import yaml
from path_helper import get_config_directory
from datetime import datetime
import time


def read_main_config():
    main_config = read_config(os.path.join(get_config_directory(), 'config.yml'))
    scenario = main_config['general']['scenario']
    scenario_config = read_config(os.path.join(get_config_directory(), 'config_{}.yml'.format(scenario)))
    for k1 in scenario_config:
        if k1 not in main_config:
            main_config[k1] = {}
        for k2 in scenario_config[k1]:
            main_config[k1][k2] = scenario_config[k1][k2]
    return main_config

def read_config(config_path):
    with open(config_path, 'r') as yml_file:
        config = yaml.load(yml_file)
        working_dir = os.getcwd()
        os.makedirs(os.path.join(working_dir,'configurations'), exist_ok=True)
        config_name=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M') + '-config.yml'
        config_copy_path = os.path.join(working_dir,'configurations', config_name)
        yaml.dump(config, open(config_copy_path, 'w'))

        print('------------ Config ------------')
        print(yaml.dump(config))
        return config

def copy_config(config, copy_to):
    with open(copy_to, 'w') as f:
        yaml.dump(config, f)

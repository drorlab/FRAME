import json
import os
from argparse import Namespace

def load_config(config_file):
	assert os.path.isfile(config_file) 
	with open(config_file,) as file:
		config = json.load(file)
	return config

def write_config(data, filepath):
	with open(filepath, 'w') as file:
		json.dump(data, file, indent=2)

def combine_args_and_configs(args: Namespace, config: dict):
	for name, value in vars(args).items():
		if value is not None:
			#print("overwriting default", name, value)
			config[name] = value
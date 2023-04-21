import logging
import json
import sys
import os

from root_path import root_dir


def save_config(cfg_dict):
	logging.info(f"Saving config")
	with open(os.path.join(root_dir, "config.json"), "w") as json_writer:
		json.dump(cfg_dict, json_writer, indent="\t", sort_keys=True)


def load_config():
	try:
		with open(os.path.join(root_dir, "config.json"), "r") as json_reader:
			return json.load(json_reader)
	except FileNotFoundError:
		logging.warning(f"Config file missing - will be created when program closes")
		return {}


def save_config_json(cfg_path, cfg_dict):
	logging.info(f"Saving config")
	# cfg_path = os.path.join(root_dir, "config.json")
	with open(cfg_path, "w") as json_writer:
		json.dump(cfg_dict, json_writer, indent="\t", sort_keys=True)


def load_config_json(cfg_path):
	try:
		with open(cfg_path, "r") as json_reader:
			return json.load(json_reader)
	except FileNotFoundError:
		logging.exception(f"Config file missing")
		return {}


def logging_setup(log_name="pyaudiorestoration"):
	# log_path = f'{os.path.join(root_dir, log_name)}.log'
	log_path = f'{log_name}.log'
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)
	# formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
	formatter = logging.Formatter('%(levelname)s | %(message)s')
	stdout_handler = logging.StreamHandler(sys.stdout)
	stdout_handler.setLevel(logging.INFO)
	stdout_handler.setFormatter(formatter)
	file_handler = logging.FileHandler(log_path, mode="w")
	file_handler.setLevel(logging.DEBUG)
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	logger.addHandler(stdout_handler)


if __name__ == '__main__':
	cfg = load_config()
	print(cfg)

import logging
import json
import sys
import os

from root_path import root_dir


def save_config(cfg_dict):
	save_json(os.path.join(root_dir, "config.json"), cfg_dict)


def load_config():
	return load_json(os.path.join(root_dir, "config.json"))


def save_json(json_path, dic):
	logging.info(f"Saving {os.path.basename(json_path)}")
	with open(json_path, "w") as json_writer:
		json.dump(dic, json_writer, indent="\t", sort_keys=True)


def load_json(json_path):
	try:
		with open(json_path, "r") as json_reader:
			return json.load(json_reader)
	except FileNotFoundError:
		logging.exception(f"{os.path.basename(json_path)} file missing")
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

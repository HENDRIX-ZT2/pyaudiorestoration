import logging
import json
import sys


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


def read_config(cfg_path):
	cfg = {}
	with open(cfg_path, 'r', encoding='utf-8') as cfg_file:
		for line in cfg_file:
			if not line.startswith("#") and "=" in line:
				(key, val) = line.strip().split("=")
				key = key.strip()
				val = val.strip()
				if val.startswith("["):
					# strip list format [' ... ']
					val = val[2:-2]
					cfg[key] = [v.strip() for v in val.split("', '")]					
				else:
					cfg[key] = val
	return cfg


def write_config(cfg_path, cfg):
	stream = "\n".join( [key+"="+str(val) for key, val in cfg.items()] )
	with open(cfg_path, 'w', encoding='utf8') as cfg_file:
		cfg_file.write(stream)


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
	cfg = read_config("config.ini")
	print(cfg)
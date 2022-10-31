import logging
import json
import os


def save_config_json(cfg_path, cfg_dict):
	logging.info(f"Saving config")
	# cfg_path = os.path.join(root_dir, "config.json")
	with open(cfg_path, "w") as json_writer:
		json.dump(cfg_dict, json_writer, indent="\t", sort_keys=True)


def read_config(cfg_path):
	with open(cfg_path, 'r', encoding='utf-8') as cfg_file:
		cfg = {}
		for line in cfg_file:
			if not line.startswith("#") and "=" in line:
				(key, val) = line.strip().split("=")
				key = key.strip()
				val = val.strip()
				if val.startswith("["):
					#strip list format [' ... ']
					val = val[2:-2]
					cfg[key] = [v.strip() for v in val.split("', '")]					
				else:
					cfg[key] = val
		return cfg

def write_config(cfg_path, cfg):
	stream = "\n".join( [key+"="+str(val) for key, val in cfg.items()] )
	with open(cfg_path, 'w', encoding='utf8') as cfg_file:
		cfg_file.write(stream)

if __name__ == '__main__':
	cfg = read_config("config.ini")
	print(cfg)
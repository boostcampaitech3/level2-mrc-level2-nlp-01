import json
import os

__all__ = ['read_json', 'increment_directory']

def read_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
    return data

def update_argument(args, configs):
    for arg in configs:
        if arg in args:
            setattr(args, arg, configs[arg])
        else:
            raise ValueError(f"no argument {arg}")
    return args


def increment_directory(directory_path):
    i = 1
    while True:
        path = f'{directory_path}_{i}/'
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            break
        i += 1
    return path
import json
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

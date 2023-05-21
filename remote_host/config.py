import json
import os

from pathlib import Path

_config = {}
with open(os.path.join(Path(__file__).parent, 'conf', 'conf.json'), 'r') as conf_file:
    _config = json.load(conf_file)

def get_config():
    return _config

import sys, os
sys.path.append(os.path.abspath('..'))
from utils.config import parse_args
from verification_cfg import multiple_experiments
import yaml

if __name__ == '__main__':
    args = parse_args()
    with open(os.path.join("../config", "global_cfg.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)

    multiple_experiments(args, global_cfg)
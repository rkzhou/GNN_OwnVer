import sys, os
sys.path.append(os.path.abspath('..'))
from utils.config import parse_args
from verification_cfg import multiple_experiments

if __name__ == '__main__':
    args = parse_args()
    multiple_experiments(args)
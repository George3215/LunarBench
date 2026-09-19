"""Launch the existing demo. No environment API, registry or dataset layer."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ue'))
from launch import launch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--engine',choices=['ue','mujoco','both'],default='mujoco')
    args=parser.parse_args()
    print(f'Demo pid={launch(engine=args.engine).pid}')


if __name__=='__main__':
    main()

# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import numpy as np
import argparse
import json
import os

def get_args():
    parser = argparse.ArgumentParser(description='Efficient 4DGS')
    parser.add_argument('-m','--model_path', type=str)
    return parser.parse_args()


def main():
    args = get_args()
    print(args.model_path)
    dirs = sorted(os.listdir(os.path.join(args.model_path, 'frames')))
    n_frames = len(dirs)

    percentiles = np.array([1, 5, 10, 25, 50, 75, 90, 95, 99])
    for i,frame in enumerate(dirs):
        if i == n_frames // 2:
            with open(
                os.path.join(
                    args.model_path, 
                    'frames', 
                    frame,
                    'percentiles.json'
                ),
                'r'
            ) as f:
                data = json.load(f)

            # import pdb; pdb.set_trace()
        
            for att_name in data.keys():
                this_data = data[att_name]
                for i in range(len(this_data)):
                    print(f'Percentile {percentiles[i]} {att_name}: {this_data[i]}')


if __name__ == '__main__':
    main()
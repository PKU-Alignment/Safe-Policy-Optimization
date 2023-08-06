# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import argparse
import shlex
import subprocess
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[ 
            "ShadowHandOver",
            "ShadowHandCatchUnderarm",
        ],
        help="the ids of the environment to benchmark",
    )
    parser.add_argument(
        "--algo",
        nargs="+",
        default=[
            "macpo",
            "mappo",
            "happo",
            "mappolag",
        ],
        help="the ids of the algorithm to benchmark",
    )
    parser.add_argument(
        "--num-seeds", type=int, default=1, help="the number of random seeds"
    )
    parser.add_argument(
        "--start-seed", type=int, default=0, help="the number of the starting seed"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="the number of workers to run benchmark experimenets",
    )
    parser.add_argument(
        "--exp-name", type=str, default="benchmark_hand", help="name of the experiment"
    )
    args = parser.parse_args()

    return args


def run_experiment(command: str):
    command_list = shlex.split(command)
    print(f"running {command}")
    fd = subprocess.Popen(command_list)
    return_code = fd.wait()
    assert return_code == 0


if __name__ == "__main__":
    args = parse_args()

    commands = []
    devices = ["0", "1", "2", "3", "4", "5", "6", "7"]
    # Please change the device id to the one you want to use
    available_devices = torch.cuda.device_count()
    assert available_devices >= len(devices), f"only {available_devices} devices available"
    idx = 0
    log_dir = f"./runs/{args.exp_name}"
    for seed in range(0, args.num_seeds):
        for task in args.tasks:
            for algo in args.algo:
                device = devices[idx]
                commands += [
                    " ".join(
                        [
                            f"python {algo}.py",
                            "--task",
                            task,
                            "--seed",
                            str(args.start_seed + seed),
                            "--write-terminal",
                            "False",
                            "--experiment",
                            args.exp_name,
                            "--device-id",
                            device,
                        ]
                    )
                ]
                idx = (idx + 1) % len(devices)

    print("======= commands to run:")
    for command in commands:
        print(command)

    if args.workers > 0:
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(
            max_workers=args.workers, thread_name_prefix="safepo-benchmark-worker-"
        )
        for command in commands:
            executor.submit(run_experiment, command)
        executor.shutdown(wait=True)
    else:
        print(
            "not running the experiments because --workers is set to 0; just printing the commands to run"
        )

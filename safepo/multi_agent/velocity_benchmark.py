import argparse
import shlex
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "Safety2x4AntVelocity-v0",
            "Safety4x2AntVelocity-v0",
            "Safety6x1HalfCheetahVelocity-v0",
            "Safety2x3HalfCheetahVelocity-v0",
            "Safety3x1HopperVelocity-v0",
            "Safety2x1SwimmerVelocity-v0",
            "Safety2x3Walker2dVelocity-v0",
            "Safety9or8HumanoidVelocity-v0"
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
        "--num-seeds", type=int, default=3, help="the number of random seeds"
    )
    parser.add_argument(
        "--start-seed", type=int, default=0, help="the number of the starting seed"
    )
    parser.add_argument(
        "--safety-bound", type=float, default=25.0, help="the cost limit"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="the number of workers to run benchmark experimenets",
    )
    parser.add_argument(
        "--exp-name", type=str, default="benchmark_velocity", help="name of the experiment"
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

    log_dir = f"./runs/{args.exp_name}"
    for seed in range(0, args.num_seeds):
        for task in args.tasks:
            for algo in args.algo:
                commands += [
                    " ".join(
                        [
                            f"python train_vel.py",
                            "--algo",
                            algo,
                            "--task",
                            task,
                            "--seed",
                            str(args.start_seed + seed),
                            "--write-terminal",
                            "False",
                            "--experiment",
                            args.exp_name,
                            "--safety-bound",
                            str(args.safety_bound),
                        ]
                    )
                ]

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

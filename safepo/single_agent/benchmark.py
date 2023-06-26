import argparse
import shlex
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-ids",
        nargs="+",
        default=[
            "SafetyAntVelocity-v1",
            "SafetyHalfCheetahVelocity-v1",
        ],
        help="the ids of the environment to benchmark",
    )
    parser.add_argument(
        "--algo",
        nargs="+",
        default=[
            "cup",
            "focops",
            "cpo",
            "cppo_pid",
            "rcpo",
            "ppo_lag",
        ],
        help="the ids of the algorithm to benchmark",
    )
    parser.add_argument(
        "--num-seeds", type=int, default=1, help="the number of random seeds"
    )
    parser.add_argument(
        "--start-seed", type=int, default=5, help="the number of the starting seed"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="the number of workers to run benchmark experimenets",
    )
    parser.add_argument(
        "--exp-name", type=str, default="benchmark_single_env", help="name of the experiment"
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
        for env_id in args.env_ids:
            for algo in args.algo:
                commands += [
                    " ".join(
                        [
                            f"python {algo}.py",
                            "--env-id",
                            env_id,
                            "--seed",
                            str(args.start_seed + seed),
                            "--write-terminal",
                            "False",
                            "--log-dir",
                            log_dir,
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

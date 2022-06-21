import argparse
import psutil
import sys
import time
import warnings
import safepo.common.mpi_tools as mpi_tools
from safepo.algos import REGISTRY
from safepo.common.runner import Runner

try:
    import safety_gym
except ImportError:
    warnings.warn('safety_gym package not found.')

try:
    import bullet_safety_gym
except ImportError:
    warnings.warn('Bullet-Safety-Gym package not found.')


if __name__ == '__main__':

    # return the number of physical cores only
    physical_cores = psutil.cpu_count(logical=False)
    default_log_dir = "./runs"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--algo', type=str, required=True,
                        help='Choose from: {ppo, trpo, ppo_lagrangian, trpo_lagrangian, cpo, pcpo, focops}')
    parser.add_argument('--env_id', type=str, required=True,
                        help='The environment name of Safety_gym, Bullet_Safety_Gym')
    parser.add_argument('--seed', default=0, type=int,
                        help='Define the seed of experiments')
    parser.add_argument('--cores', '-c', type=int, default=physical_cores,
                        help=f'Number of cores used for calculations.')
    parser.add_argument('--runs', '-r', type=int, default=1,
                        help='Number of total runs that are executed.')
    parser.add_argument('--debug', action='store_true',
                        help='Show debug prints during training.')
    parser.add_argument('--no-mpi', action='store_true',
                        help='Do not use MPI for parallel execution.')
    parser.add_argument('--play', action='store_true',
                        help='Visualize agent after training.')
    parser.add_argument('--search', action='store_true',
                        help='If given search over learning rates.')
    parser.add_argument('--log-dir', type=str, default=default_log_dir,
                        help='Define a log/data directory.')
    args, unparsed_args = parser.parse_known_args()
    # Use number of physical cores as default. 
    # If also hardware threading CPUs should be used, enable this by the use_number_of_threads=True
    use_number_of_threads = True if args.cores > physical_cores else False
    if mpi_tools.mpi_fork(args.cores,use_number_of_threads=use_number_of_threads):
        # Re-launches the current script with workers linked by MPI
        sys.exit()
    print('Unknowns:', unparsed_args) if mpi_tools.proc_id() == 0 else None
    print('Core:', args.cores) if mpi_tools.proc_id() == 0 else None
    print('use_mpi:', not args.no_mpi) if mpi_tools.proc_id() == 0 else None

    model = Runner(
        algo=args.algo,
        env_id=args.env_id,
        log_dir=args.log_dir,
        init_seed=args.seed,
        unparsed_args=unparsed_args,
        use_mpi=not args.no_mpi
    )
    model.compile(num_runs=args.runs, num_cores=args.cores)
    model.train()
    model.eval()
    if args.play:
        model.play()

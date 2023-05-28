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
"""Train an agent."""
import argparse

from safepo.common.runner import Runner

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--algo', type=str, required=True,
                        help='Choose from: {ppo, trpo, ppo-lag, trpo-lag, cpo, pcpo, focops}')
    parser.add_argument('--env-id', type=str, required=True,
                        help='The environment name of Safety_gym, Bullet_Safety_Gym')
    parser.add_argument('--seed', default=0, type=int,
                        help='Define the seed of experiments')
    parser.add_argument('--log-dir', type=str, default='./runs',
                        help='Define a log/data directory.')
    args, unparsed_args = parser.parse_known_args()

    runner = Runner(
        algo=args.algo,
        env_id=args.env_id,
        log_dir=args.log_dir,
        seed=args.seed,
        unparsed_args=unparsed_args,
    )
    # model.compile(num_runs=args.runs, num_cores=args.cores)
    runner.train()
    # model.eval()
    # if args.play:
    #     model.play()

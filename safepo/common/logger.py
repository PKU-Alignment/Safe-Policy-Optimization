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


import atexit
import csv
import json
import os
import os.path as osp
import warnings

import joblib
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

# from safepo.common.mpi_tools import proc_id, mpi_statistics_scalar


DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False


def convert_json(obj):
    """Convert obj to a version which can be serialized with JSON."""
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj, "__name__") and "lambda" not in obj.__name__:
            return convert_json(obj.__name__)

        elif hasattr(obj, "__dict__") and obj.__dict__:
            obj_dict = {
                convert_json(k): convert_json(v) for k, v in obj.__dict__.items()
            }
            return {str(obj): obj_dict}

        return str(obj)

def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    return "\x1b[{}m{}\x1b[0m".format(";".join(attr), string)


class Logger:
    """
    A class for logging experimental data and managing logging-related functionalities.
    
    Args:
        log_dir (str): The directory path for storing log files.
        seed (int or None): The seed for reproducibility. Default is None.
        output_fname (str): The name of the output file. Default is "progress.csv".
        debug (bool): Toggle for debugging mode. Default is False.
        level (int): The logging level. Default is 1.
        use_tensorboard (bool): Toggle for using TensorBoard logging. Default is True.
        verbose (bool): Toggle for verbose output. Default is True.
    """

    def __init__(
        self,
        log_dir,
        seed=None,
        output_fname="progress.csv",
        debug: bool = False,
        level: int = 1,
        use_tensorboard=True,
        verbose=True,
    ):
        self.log_dir = log_dir
        self.debug = debug
        self.level = level
        self.verbose = verbose

        os.makedirs(self.log_dir, exist_ok=True)
        self.output_file = open(  # noqa: SIM115 # pylint: disable=consider-using-with
            os.path.join(self.log_dir, output_fname),
            encoding="utf-8",
            mode="w",
        )
        atexit.register(self.output_file.close)
        self._csv_writer = csv.writer(self.output_file)

        self.epoch = 0
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = "-".join(
            [log_dir.split("/")[-3], log_dir.split("/")[-2], "seed", seed]
        )
        self.torch_saver_elements = None
        self.use_tensorboard = use_tensorboard
        self.logged = True

        # Setup tensor board logging if enabled and MPI root process
        if use_tensorboard:
            self.summary_writer = SummaryWriter(os.path.join(self.log_dir, "tb"))

    def close(self):
        """Close the output file.
        """
        self.output_file.close()

    def debug(self, msg, color="yellow"):
        """Print a colorized message to stdout."""
        if self.debug:
            print(colorize(msg, color, bold=False))

    def log(self, msg, color="green"):
        """Print a colorized message to stdout."""
        if self.verbose and self.level > 0:
            print(colorize(msg, color, bold=False))

    def log_tabular(self, key, val):
        """
        Log a key-value pair in a tabular format for subsequent output.

        Args:
            key (str): The key to log.
            val: The corresponding value to log.

        Raises:
            AssertionError: If attempting to introduce a new key that was not included
                in the first iteration, or if the key has already been set in the current iteration.
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, (
                "Trying to introduce a new key %s that you didn't include in the first iteration"
                % key
            )
        assert key not in self.log_current_row, (
            "You already set %s this iteration. Maybe you forgot to call dump_tabular()"
            % key
        )
        self.log_current_row[key] = val

    def save_config(self, config):
        """
        Save the experiment configuration as a JSON file.

        Args:
            config (dict): The experiment configuration to be saved.

        Notes:
            If `exp_name` is specified, it will be added to the configuration.

        Returns:
            None
        """
        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json["exp_name"] = self.exp_name

        output = json.dumps(
            config_json, separators=(",", ":\t"), indent=4, sort_keys=True
        )
        with open(osp.join(self.log_dir, "config.json"), "w") as out:
            out.write(output)

    def save_state(self, state_dict, itr=None):
        """
        Save the state dictionary using joblib's pickling mechanism.

        Args:
            state_dict: The state dictionary to be saved.
            itr (int or None): The iteration number. If provided, it's used in the filename.

        Notes:
            If `itr` is None, the default filename is "state.pkl".

        Returns:
            None
        """
        fname = "state.pkl" if itr is None else "state%d.pkl" % itr
        try:
            joblib.dump(state_dict, osp.join(self.log_dir, fname))
        except:
            self.log("Warning: could not pickle state_dict.", color="red")
        if hasattr(self, "torch_saver_elements"):
            self.torch_save(itr)

    def setup_torch_saver(self, what_to_save):
        """
        Set up easy model saving for a single PyTorch model.

        Because PyTorch saving and loading is especially painless, this is
        very minimal; we just need references to whatever we would like to
        pickle. This is integrated into the logger because the logger
        knows where the user would like to save information about this
        training run.

        Args:
            what_to_save: Any PyTorch model or serializable object containing
                PyTorch models.
        """
        self.torch_saver_elements = what_to_save

    def torch_save(self, itr=None):
        """Saves the PyTorch model (or models)."""

        self.log("Save model to disk...")
        assert (
            self.torch_saver_elements is not None
        ), "First have to setup saving with self.setup_torch_saver"
        fpath = "torch_save"
        fpath = osp.join(self.log_dir, fpath)
        fname = "model" + ("%d" % itr if itr is not None else "") + ".pt"
        fname = osp.join(fpath, fname)
        os.makedirs(fpath, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.save(self.torch_saver_elements, fname)
        torch.save(self.torch_saver_elements.state_dict(), fname)
        self.log("Done.")

    def dump_tabular(self) -> None:
        """Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.
        """
        vals = list()
        self.epoch += 1
        # Print formatted information into console
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15, max(key_lens))
        keystr = "%" + "%d" % max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len

        print("-" * n_slashes) if self.verbose and self.level > 0 else None
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = "%8.3g" % val if hasattr(val, "__float__") else val
            if self.verbose and self.level > 0:
                print(fmt % (key, valstr))
            vals.append(val)
        if self.verbose and self.level > 0:
            print("-" * n_slashes, flush=True)

        # Write into the output file (can be any text file format, e.g. CSV)
        if self.output_file is not None:
            if self.first_row:
                self._csv_writer.writerow(self.log_current_row.keys())

            self._csv_writer.writerow(self.log_current_row.values())
            self.output_file.flush()

        if self.use_tensorboard:
            for key, val in self.log_current_row.items():
                self.summary_writer.add_scalar(key, val, global_step=self.epoch)

        # free logged information in all processes...
        self.log_current_row.clear()
        self.first_row = False


class EpochLogger(Logger):
    
    def __init__(
        self,
        log_dir,
        seed=None,
        output_fname="progress.csv",
        debug: bool = False,
        level: int = 1,
        use_tensorboard=True,
        verbose=True,
    ):
        super().__init__(
            log_dir=log_dir,
            seed=seed,
            output_fname=output_fname,
            debug=debug,
            level=level,
            use_tensorboard=use_tensorboard,
            verbose=verbose,
        )
        self.epoch_dict = dict()

    def dump_tabular(self):
        self.logged = True
        super().dump_tabular()
        for k, v in self.epoch_dict.items():
            if len(v) > 0:
                print(f"epoch_dict: key={k} was not logged.")

    def store(self, add_value=False, **kwargs):
        for k, v in kwargs.items():
            if add_value:
                if k not in self.log_current_row.keys():
                    self.log_current_row[k] = [0]
                self.log_current_row[k][0] += v
            else:
                if k not in self.epoch_dict.keys():
                    self.epoch_dict[k] = []
                self.epoch_dict[k].append(v)
            

    def log_tabular(self, key, val=None, min_and_max=False, std=False):
        if val is not None:
            super().log_tabular(key, val)
        else:
            v = np.mean(self.epoch_dict[key])
            super().log_tabular(key, v)
            if min_and_max:
                super().log_tabular(key + "/Min", np.min(self.epoch_dict[key]))
                super().log_tabular(key + "/Max", np.max(self.epoch_dict[key]))
            if std:
                super().log_tabular(key + "/Std", np.std(self.epoch_dict[key]))
        self.epoch_dict[key] = []

    def get_stats(self, key):
        """Get the values of a diagnostic."""
        if key not in self.log_headers:
            return 0.0
        return np.mean(self.epoch_dict[key])

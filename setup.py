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
import os

from setuptools import find_packages, setup

with open(os.path.join("safepo", "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()


long_description = """
todo
"""


setup(
    name="safepo",
    packages=[package for package in find_packages() if package.startswith("safepo")],
    package_data={"safepo": ["py.typed", "version.txt"]},
    install_requires=[
        "psutil",
        "joblib",
        "tensorboard",
        "pyyaml",
        "matplotlib",
        "tensorboardX",
        "gym==0.15.3"
        # we recommend use conda install scipy and mpi4py
    ],
    description="Pytorch version of Safe Reinforcement Learning Algorithm",
    author="PKU-MARL",
    url="https://github.com/PKU-MARL/safepo-Baselines",
    author_email="jiamg.ji@gmail.com",
    keywords="Safe Single Agent Reinforcement Learning"
    "Safe Mult Agent Rinforcement Learning",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    python_requires=">=3.7",
    # PyPI package information.
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)

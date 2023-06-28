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

from setuptools import find_packages, setup

setup(
    name='safepo',
    packages=[package for package in find_packages() if package.startswith('safepo')],
    package_data={'safepo': ['py.typed', 'version.txt']},
    install_requires=[
        'psutil',
        'joblib',
        'scipy',
        "torch >= 1.10.0",
        'tensorboard >= 2.8.0',
        "wandb >= 0.13.0",
        'pyyaml >= 6.0',
        'matplotlib >= 3.7.1',
        "seaborn >= 0.12.2",
        "pandas >=  1.5.3",
        'safety-gymnasium >= 0.1.0',
        "rich>=13.3.0",
    ],
    description='Pytorch version of Safe Reinforcement Learning Algorithm',
    author='OmniSafeAI Team',
    url='https://github.com/OmniSafeAI/Safe-Policy-Optimization',
    author_email='jiamg.ji@gmail.com',
    keywords='Safe Single Agent Reinforcement Learning'
    'Safe Mult Agent Rinforcement Learning',
    license='Apache License 2.0',
    version='1.0.1',
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)

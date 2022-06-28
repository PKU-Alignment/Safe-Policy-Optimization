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
        "pandas",
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
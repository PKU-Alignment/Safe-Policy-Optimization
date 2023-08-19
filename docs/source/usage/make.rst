Efficient Commands
==================

To help users quickly reporduce our results,
we provide a command line tool for easy installation, benchmarking, and evaluation.

One line benchmark running
--------------------------

First, create a conda environment with Python 3.8.

.. code-block:: bash
    
    conda create -n safepo python=3.8
    conda activate safepo

Then, run the following command to install SafePO and run the full benchmark:

.. code-block:: bash
    
    make benchmark

This command will install SafePO in editable mode and excute the training process parallelly.
After the training process is finished, it will evaluate the trained policies and generate the benchmark results,
including training curves and evaluation rewards and costs.

Simple benchmark running
------------------------

The full benchamrk is time-consuming.
To verify the performance of SafePO, we provide a simple benchmark command,
which runs all alforithms on sampled environments and evaluate the trained policies.

.. code-block:: bash
    
    make benchmark-simple

The training logs would be saved in ``safepo/runs/benchmark``, while the evaluation results and learning curves would be saved in ``safepo/results/benchmark``.

.. warning::

    The default number of workers is 1. To run the benchmarking tools in parallel, you can increase the number of workers
    by changing the `workers` configuration in `safepo/single_agent/benchmark.py` and `safepo/multi_agent/benchmark.py`.

.. note::

    The ``Doggo`` agent is not included in the benchmarking tools because it needs 1e8 training steps to converge.
    You can run the ``Doggo`` agent by running:

    .. code-block:: bash

        cd safepo/single_agent
        python benchmark.py --tasks \
        SafetyDoggoButton1-v0 SafetyDoggoButton2-v0 \
        SafetyDoggoCircle1-v0 SafetyDoggoCircle2-v0 \
        SafetyDoggoPush1-v0 SafetyDoggoPush2-v0 \
        SafetyDoggoGoal1-v0 SafetyDoggoGoal2-v0 \
        --workers 1 --total-steps 100000000

The terminal output would be like:

.. code-block:: bash
    
    ======= commands to run:
    running python macpo.py --agent-conf 2x4 --scenario Ant --seed 0 --write-terminal False --experiment benchmark --headless True --total-steps 10000000
    running python mappo.py --agent-conf 2x4 --scenario Ant --seed 0 --write-terminal False --experiment benchmark --headless True --total-steps 10000000
    running python mappolag.py --agent-conf 2x4 --scenario Ant --seed 0 --write-terminal False --experiment benchmark --headless True --total-steps 10000000
    running python happo.py --agent-conf 2x4 --scenario Ant --seed 0 --write-terminal False --experiment benchmark --headless True --total-steps 10000000
    ...
    running python pcpo.py --task SafetyAntVelocity-v1 --seed 0 --write-terminal False --experiment benchmark --total-steps 10000000
    running python ppo_lag.py --task SafetyAntVelocity-v1 --seed 0 --write-terminal False --experiment benchmark --total-steps 10000000
    running python cup.py --task SafetyAntVelocity-v1 --seed 0 --write-terminal False --experiment benchmark --total-steps 10000000
    running python focops.py --task SafetyAntVelocity-v1 --seed 0 --write-terminal False --experiment benchmark --total-steps 10000000
    running python rcpo.py --task SafetyAntVelocity-v1 --seed 0 --write-terminal False --experiment benchmark --total-steps 10000000
    running python trpo_lag.py --task SafetyAntVelocity-v1 --seed 0 --write-terminal False --experiment benchmark --total-steps 10000000
    running python cpo.py --task SafetyAntVelocity-v1 --seed 0 --write-terminal False --experiment benchmark --total-steps 10000000
    running python cppo_pid.py --task SafetyAntVelocity-v1 --seed 0 --write-terminal False --experiment benchmark --total-steps 10000000
    ...
    Plotting from...
    ==================================================

    ./runs/benchmark/SafetyAntVelocity-v1

    ==================================================
    Plotting from...
    ==================================================

    ./runs/benchmark/Safety2x3HalfCheetahVelocity-v0

    ==================================================
    Plotting from...
    ==================================================

    ./runs/benchmark/SafetyHumanoidVelocity-v1

    ==================================================
    Plotting from...
    ==================================================
    ...
    Start evaluating focops in SafetyPointGoal1-v0
    After 1 episodes evaluation, the focops in SafetyPointGoal1-v0 evaluation reward: 12.21±2.18, cost: 26.0±19.51, the reuslt is saved in ./results/benchmark/eval_result.txt
    Start evaluating cppo_pid in SafetyPointGoal1-v0
    After 1 episodes evaluation, the cppo_pid in SafetyPointGoal1-v0 evaluation reward: 13.42±0.44, cost: 18.79±2.1, the reuslt is saved in ./results/benchmark/eval_result.txt
    ...
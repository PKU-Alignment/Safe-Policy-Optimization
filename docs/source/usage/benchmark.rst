Benchmarking Tools
==================

This repository contains a collection of tools for benchmarking the performance
of multi-agent and single-agent algorithms.

To run the benchmarking tools, you can run:

.. code-block:: bash

    cd safepo/single_agent
    python benchmark.py --workers 1

with the default configuration. This will run the benchmarking tools then reproduce the
figures in the paper. You can also run the multi-agent benchmarking tools by running:

.. code-block:: bash

    cd safepo/multi_agent
    python benchmark.py --workers 1

After running the benchmarking tools, you can run the `plooting tools and evaluation tools <./eval.html>`_  to
show the results. 

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

.. warning::

    The default number of workers is 1. To run the benchmarking tools in parallel, you can increase the number of workers
    by passing the `--workers` flag considering the number of cores in your machine.
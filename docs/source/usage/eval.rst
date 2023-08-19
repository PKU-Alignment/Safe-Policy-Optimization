Evaluating Trained Models
=========================

Model Evaluation
----------------

To evaluate the trained model, you can run:

.. code-block:: bash

    cd safepo/
    python evaluate.py --benchamrk-dir ./runs/ppo_lag_exp --eval-episodes 100 --save-dir ./results/ppo_lag_exp

This will evaluate the model in the last checkpoint of the training, and save the evaluation results in `safepo/results/ppo_lag_exp`.

Training Curve Plotter
----------------------

Training curves reveal the episodic reward and cost overtime, which is usefull to evaluate the performance of the algorithms.

suppose you have ran the training script in `algorithms training <./train.html>`_ and saved the training log in `safepo/runs/ppo_lag_exp`, then you can plot the training curve by running:

.. code-block:: bash

    cd safepo/
    python plot.py --logdir ./runs/ppo_lag_exp

.. note::

    This plotter is also suitable for mmulti-agent algorithms plotting. However, in experiment we found that 
    the cost value training curve of multi-agent safe and unsafe algorithms are largely different, which makes the
    plot not very clear. So we recommend to plot the multi-agent training curve by running the plotter in ``safepo/multi_agent/plot_for_benchmark``.

    .. code-block:: bash

        cd safepo/multi_agent
        python plot_for_benchmark.py --logdir ./runs/mappo_lag_exp

.. danger::

    Make sure you have ran at least one unsafe multi-agent algorithm and one safe multi-agent algorithm, otherwise the plotter will raise error.

You can find the plot in `safepo/results/ppo_lag_exp/`.
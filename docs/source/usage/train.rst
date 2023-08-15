Algorithms Training
===================

SafePO use single file to implement all algorithms. The single-agent algorithms are in ``safepo/single)agent`` folder while the multi-agent algorithms are in ``safepo/multi_agent`` folder.

Single-agent Algorithms
-----------------------

To run the algorithms with default configuration, you need to specify the environment name. For example, to run the ``PPOLag`` algorithm in the ``SafetyPointGoal1-v0`` environment, you can run the following command:

.. code-block:: bash

    cd safepo/single_agent
    python ppo_lag.py --env-id SafetyPointGoal1-v0 --experiment ppo_lag_exp

Then you can check the results in the ``runs/ppo_lag_exp`` folder.

Multi-agent Algorithms
----------------------

The multi-agent algorithms running is similar to the single-agent algorithms. For example, to run the ``MAPPOLag`` algorithm in the ``Safety2x4AntVelocity`` environment, you can run the following command:

.. code-block:: bash

    cd safepo/multi_agent
    python mappolag.py --scenario Ant --agent-conf 2x4 --experiment mappo_lag_exp

Then you can check the results in the ``runs/mappo_lag_exp`` folder.

Cunstomizing Training
---------------------

We use command line interface to support training customization.
We provide the detailed description of the command line arguments in the following

.. tab-set::

    .. tab-item:: Single-agent Algorithms

        +--------------------+----------------------------------+-----------------------------------------------+
        | Argument           | Description                      | Default Value                                 |
        +====================+==================================+===============================================+
        | --seed             | Seed of the experiment           | 0                                             |
        +--------------------+----------------------------------+-----------------------------------------------+
        | --device           | Device to run the code           | "cpu"                                         |
        +--------------------+----------------------------------+-----------------------------------------------+
        | --num-envs         | Number of parallel game          | 10                                            |
        |                    | environments                     |                                               |
        +--------------------+----------------------------------+-----------------------------------------------+
        | --total-steps      | Total timesteps of the           | 10000000                                      |
        |                    | experiments                      |                                               |
        +--------------------+----------------------------------+-----------------------------------------------+
        | --env-id           | ID of the environment            | "SafetyPointGoal1-v0"                         |
        +--------------------+----------------------------------+-----------------------------------------------+
        | --use-eval         | Toggles evaluation               | False                                         |
        +--------------------+----------------------------------+-----------------------------------------------+
        | --steps-per-epoch  | Number of steps to run in each   | 20000                                         |
        |                    | environment per policy rollout   |                                               |
        +--------------------+----------------------------------+-----------------------------------------------+
        | --critic-lr        | Learning rate of the critic      | 1e-3                                          |
        |                    | network                          |                                               |
        +--------------------+----------------------------------+-----------------------------------------------+
        | --log-dir          | Directory to save agent logs     | "../runs"                                     |
        +--------------------+----------------------------------+-----------------------------------------------+
        | --experiment       | Name of the experiment           | "single_agent_experiment"                     |
        +--------------------+----------------------------------+-----------------------------------------------+
        | --write-terminal   | Toggles terminal logging         | True                                          |
        +--------------------+----------------------------------+-----------------------------------------------+
        | --use-tensorboard  | Toggles tensorboard logging      | False                                         |
        +--------------------+----------------------------------+-----------------------------------------------+

    .. tab-item:: Multi-agent Algorithms

        +-------------------+--------------------------------+----------------------------------------------+
        | Parameter         | Description                    | Default Value                                |
        +===================+================================+==============================================+
        | --use-eval        | Use evaluation environment     | False                                        |
        |                   | for testing                    |                                              |
        +-------------------+--------------------------------+----------------------------------------------+
        | --task            | The task to run                | "MujocoVelocity"                             |
        +-------------------+--------------------------------+----------------------------------------------+
        | --agent-conf      | The agent configuration        | "2x4"                                        |
        +-------------------+--------------------------------+----------------------------------------------+
        | --scenario        | The scenario                   | "Ant"                                        |
        +-------------------+--------------------------------+----------------------------------------------+
        | --experiment      | Experiment name                | "Base"                                       |
        |                   | If used with --metadata flag,  |                                              |
        |                   | additional information about   |                                              |
        |                   | physics engine, sim device,    |                                              |
        |                   | pipeline and domain            |                                              |
        |                   | randomization will be added    |                                              |
        |                   | to the name                    |                                              |
        +-------------------+--------------------------------+----------------------------------------------+
        | --seed            | Random seed                    | 0                                            |
        +-------------------+--------------------------------+----------------------------------------------+
        | --model-dir       | Choose a model dir             | ""                                           |
        +-------------------+--------------------------------+----------------------------------------------+
        | --safety-bound    | cost_limit                     | 25.0                                         |
        +-------------------+--------------------------------+----------------------------------------------+
        | --device          | The device to run the model on | "cpu"                                        |
        +-------------------+--------------------------------+----------------------------------------------+
        | --device-id       | The device id to run the       | 0                                            |
        |                   | model on                       |                                              |
        +-------------------+--------------------------------+----------------------------------------------+
        | --write-terminal  | Toggles terminal logging       | True                                         |
        +-------------------+--------------------------------+----------------------------------------------+
        | --headless        | Toggles headless mode          | False                                        |
        +-------------------+--------------------------------+----------------------------------------------+
        | --total-steps     | Total timesteps of the         | None                                         |
        |                   | experiments                    |                                              |
        +-------------------+--------------------------------+----------------------------------------------+
        | --num-envs        | The number of parallel game    | None                                         |
        |                   | environments                   |                                              |
        +-------------------+--------------------------------+----------------------------------------------+

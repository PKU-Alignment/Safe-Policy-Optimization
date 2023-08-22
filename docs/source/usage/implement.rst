Customization of Algorithms
===========================

Trustworthy Classic RL Algorithms
---------------------------------

As Safe RL algorithms are also based on classic RL algorithms, a trustworthy implementation of the classic RL algorithm is required.
SafePO provided a set of classic RL algorithms, ``PPO``, ``NaturaPG`` and ``TRPO``.

To verify the correctness of the classic RL algorithms, we provide the performance of them in the `MuJoCo Velocity <https://gymnasium.farama.org/environments/mujoco/>`_ environment.

.. raw:: html

    <iframe src="https://wandb.ai/pku_rl/SafePO/reports/Benchmarks-of-Classic-RL--Vmlldzo1MTgyMDA5" style="border:none;width:90%; height:1000px">

.. raw:: html

    </iframe>


Integrated Safe RL Pipeline
----------------------------

SafePO's classic RL algorithms are integrated with the Safe RL pipeline, though they make no use of the constraint.
You can customize the Safe RL algorithms based on the classic RL algorithms.

Briefly, the ``PPO`` in SafePO has the following characteristics, which are also suitable for other customization of safe RL algorithms.

- ``VectorizedOnPolicyBuffer``: A vectorized buffer supporting cost advantage estimation.
- ``ActorVCritic``: A actor-critic network supporting cost value estimation.
- ``Lagrange``: A lagrangian multiplier for constraint violation control.

Beyond the above characteristics, the ``PPO`` in SafePO also provides a training pipeline for data collection and training.
You can customize new algorithms based on it.

Next we will provide a detailed example to show how to customize the ``PPO`` algorithm to ``PPO-Lag`` algorithm.

Example: PPO-Lag
----------------

The Lagrangian multiplier is a useful tool to control the constraint violation in the Safe RL algorithms.
Classic RL algorithms combined with the Lagrangian multiplier are trustworthy baselines for Safe RL algorithms.

.. note::

    SafePO provide naive lagrangian multiplier and pid-based lagrangian multiplier.
    The former suffer from oscillation and the latter is more stable.

Here we provide an example of using the Lagrangian multiplier in the ``PPO`` algorithm.

- First, import the ``Lagrange`` class.

.. code-block:: python

    from safepo.common.lagrange import Lagrange

- Second, initialize the ``Lagrange`` class.

.. code-block:: python

    lagrange = Lagrange(
        cost_limit=args.cost_limit,
        lagrangian_multiplier_init=args.lagrangian_multiplier_init,
        lagrangian_multiplier_lr=args.lagrangian_multiplier_lr,
    )

- Third, update the ``Lagrange`` class.

.. code-block:: python

    ep_costs = logger.get_stats("Metrics/EpCost")
    lagrange.update_lagrange_multiplier(ep_costs)

- Finally, use the lagrangian multiplier to update the policy network.

.. code-block:: python

    advantage = data["adv_r"] - lagrange.lagrangian_multiplier * data["adv_c"]
    advantage /= (lagrange.lagrangian_multiplier + 1)

.. note::

    Only within 10 lines of code, you can use the Lagrangian multiplier in the ``PPO`` algorithm.
    The framework of PPO is also suitable for other customization of safe RL algorithms.
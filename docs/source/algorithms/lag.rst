Lagrangian Methods
==================

Experiment Results
------------------

.. tab-set::

    .. tab-item:: PPO-Lag

      .. raw:: html

         <iframe src="https://wandb.ai/pku_rl/SafePO/reports/Performance-PPO-Lag--Vmlldzo1MDgyNTcx" style="border:none;width:90%; height:1000px" >

      .. raw:: html

         </iframe>

    .. tab-item:: TRPO-Lag

      .. raw:: html

         <iframe src="https://wandb.ai/pku_rl/SafePO/reports/TRPO-Lag-Training-Curves--Vmlldzo1MTgyMDI2" style="border:none;width:90%; height:1000px" >

      .. raw:: html

         </iframe>


Implement Details
-----------------

.. note::

   All experiments are ran under total 1e7 steps, while in the `Doggo <https://www.safety-gymnasium.com/en/latest/components_of_environments/agents.html>`_ agent, 1e8 steps are used.
   This setting is the same as `Safety-Gym <https://www.google.com.hk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjevqzswM-AAxXZtlYBHVFlDOAQFnoECBIQAQ&url=https%3A%2F%2Fopenai.com%2Fresearch%2Fsafety-gym&usg=AOvVaw2bTv-b9BBuC-4eDmkFZPr3&opi=89978449>`_

Environment Wrapper
~~~~~~~~~~~~~~~~~~~

In the course of our experimental investigations, we have discerned
certain hyperparameters that wield a discernible influence upon the
algorithm's performance:

The parameter denoted as

-  ``obs_normalize``, which pertains to the normalization of
   observations.
-  ``reward_normalize``, governing the normalization of rewards.
-  ``cost_normalize``, governing the normalization of costs.

Throughout the experimental trials, a consistent pattern emerged,
wherein the setting ``obs_normalize=True`` consistently yielded superior
results.

.. note::

   Significantly, the outcome is not uniformly corroborated when it comes
   to the ``reward_normalize`` parameter. Its affirmative setting
   ``reward_normalize=True`` does not invariably outperform the negative
   counterpart ``reward_normalize=False``, a trend particularly pronounced
   in the ``SafetyHopperVelocity-v1`` and ``SafetyWalker2dVelocity-v1``
   environments.

Therefore, We make the environment wrapper to control the normalization
of observations, rewards and costs:

.. code:: python

      env = safety_gymnasium.make(env_id)
      env.reset(seed=seed)
      obs_space = env.observation_space
      act_space = env.action_space
      env = SafeAutoResetWrapper(env)
      env = SafeRescaleAction(env, -1.0, 1.0)
      env = SafeNormalizeObservation(env)
      env = SafeUnsqueeze(env)
   
       return env, obs_space, act_space

Lagrangian Multiplier
~~~~~~~~~~~~~~~~~~~~~

Lagrangian-based algorithms use ``Lagrangian Multiplier`` to control the safety
constraint. The ``Lagrangian Multiplier`` is an Integrated part of
SafePO.

Some key points:

-  The implementation of ``Lagrangian Multiplier`` is based on ``Adam``
   optimizer for a smooth update.
-  The ``Lagrangian Multiplier`` is updated every epoch based on the
   total cost violation of current episodes.

Key implementation:

.. code:: python

   from safepo.common.lagrange import Lagrange

   # setup lagrangian multiplier
   COST_LIMIT = 25.0
   LAGRANGIAN_MULTIPLIER_INIT = 0.001
   LAGRANGIAN_MULTIPLIER_LR = 0.035
   lagrange = Lagrange(
       cost_limit=COST_LIMIT,
       lagrangian_multiplier_init=LAGRANGIAN_MULTIPLIER_INIT,
       lagrangian_multiplier_lr=LAGRANGIAN_MULTIPLIER_LR,
   )

   # update lagrangian multiplier
   # suppose ep_cost is 50.0
   ep_cost = 50.0
   lagrange.update_lagrange_multiplier(ep_cost)

   # use lagrangian multiplier to control the advanatge
   advantage = data["adv_r"] - lagrange.lagrangian_multiplier * data["adv_c"]
   advantage /= (lagrange.lagrangian_multiplier + 1)

Please refer to `Lagrangian Multiplier <../api/lagrange.rst>`__ for more
details.

Configuration Analysis
----------------------

The implementation of Lagrangian multiplier based algorithms is based on classic reinforcement learning algorithms, e.g. ``PPO`` and ``TRPO``. 
And the ``PPO`` and ``TRPO`` hyperparameters is basically the same as community version.
We listed the key hyperparameters as follows:

.. tab-set::

    .. tab-item:: PPO

      - ``batch_size``: 64
      - ``gamma``: 0.99
      - ``lam``: 0.95
      - ``lam_c``: 0.95
      - ``clip``: 0.2
      - ``actor_lr``: 3e-4
      - ``critic_lr``: 3e-4
      - ``hidden_size``: 64 for all agents while 256 for ``Doggo`` and ``Ant``

    .. tab-item:: TRPO

      - ``batch_size``: 128
      - ``gamma``: 0.99
      - ``lam``: 0.95
      - ``lam_c``: 0.95
      - ``critic_lr``: 1e-3
      - ``hidden_size``: 64 for all agents while 256 for ``Doggo`` and ``Ant``


Lagrangian Multiplier Related
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In practical scenarios, we frequently encounter the need to manually define the initial value and learning rate. 
It's worth noting that Lagrange algorithms are notably sensitive to the choice of hyperparameters.

.. warning::

   If the initial value of the Lagrange multiplier or learning rate is set high, it can lead to diminished agent rewards. 
   Conversely, lower values risk violating constraints. 
   Thus, striking a balance between optimizing rewards and adhering to constraints becomes a challenging endeavor.

Based on Safety-Gymnasium tasks, we found that the following hyperparameters are suitable for most tasks:

- ``lagrangian_multiplier_init``: 0.001
- ``lagrangian_multiplier_lr``: 0.035

While these hyperparameters are suitable for `Safety-Gymnasium <https://github.com/PKU-Alignment/safety-gymnasium>`_ tasks only, 
for other tasks, you may need to adjust them accordingly.
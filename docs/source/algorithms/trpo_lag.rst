TRPO-Lagrangian
===============

Experiment Results
------------------

.. raw:: html

   <iframe src="https://wandb.ai/pku_rl/SafePO/reports/TRPO-Lagrangian--Vmlldzo1MDc2MTI3" style="border:none;width:100%; height:1000px" title="Performance-TRPO-Lag">

.. raw:: html

   </iframe>

Implement Details
-----------------

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

Significantly, the outcome is not uniformly corroborated when it comes
to the ``reward_normalize`` parameter. Its affirmative setting
``reward_normalize=True`` does not invariably outperform the negative
counterpart ``reward_normalize=False``, a trend particularly pronounced
in the ``SafetyHopperVelocity-v1`` and ``SafetyWalker2dVelocity-v1``
environments.

Therefore, We make the environment wrapper to control the normalization
of observations, rewards and costs:

.. code:: python

       if num_envs > 1:
           def create_env() -> Callable:
               """Creates an environment that can enable or disable the environment checker."""
               env = safety_gymnasium.make(env_id)
               env = SafeRescaleAction(env, -1.0, 1.0)
               env = SafeNormalizeObservation(env)
               return env
           env_fns = [create_env for _ in range(num_envs)]
           env = SafetyAsyncVectorEnv(env_fns)
           env.reset(seed=seed)
           obs_space = env.single_observation_space
           act_space = env.single_action_space
       else:
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

``TRPOLag`` use ``Lagrangian Multiplier`` to control the safety
constraint. The ``Lagrangian Multiplier`` is an intergrated part of
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
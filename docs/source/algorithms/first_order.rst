First Order Projection Methods
==============================

Experiment Results
------------------

.. tab-set::

    .. tab-item:: CUP

      .. raw:: html

         <iframe src="https://wandb.ai/pku_rl/SafePO/reports/CUP-Training-Curves--Vmlldzo1MTgxOTcx" style="border:none;width:90%; height:1000px" >

      .. raw:: html

         </iframe>

    .. tab-item:: FOCOPS

      .. raw:: html

         <iframe src="https://wandb.ai/pku_rl/SafePO/reports/FOCOPS-Training-Curves--Vmlldzo1MTgyMDE0" style="border:none;width:90%; height:1000px" >

      .. raw:: html

         </iframe>


Implementation Details
----------------------

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

Projection Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~

The key idea of ``CUP`` and ``FOCOPS`` is projecting the policy back to the safe set.
A more detailed theoretical analysis can be found in `here <https://omnisafe.readthedocs.io/en/latest/saferl/focops.html>`_.

We provide how ``SafePO`` implements the two stage projection:

.. tab-set::

    .. tab-item:: CUP

      CUP first make a PPO update to improve the policy reward.
      Then it projects the policy back to the safe set.
      We will focus on the projection part.

      - Get the cost advantage from buffer and prepare training data.

      .. code:: python

         advantage = data["adv_c"]
         dataloader = DataLoader(
               dataset=TensorDataset(
                  data["obs"], data["act"], data["log_prob"], advantage, old_mean, old_std
               ),
               batch_size=64,
               shuffle=True,
         )

      - Update the policy by using cost advantage and kl divergence.

      .. code:: python

         coef = (1 - args.cup_gamma * args.cup_lambda) / (1 - args.cup_gamma)
         loss_pi_cost = (
            lagrange.lagrangian_multiplier * coef * ratio * adv_b + temp_kl
         ).mean()

      Where ``args.cup_gamma`` is the GAE gamma, ``args.cup_lambda`` is the cost GAE lambda, ``ratio`` is the importance sampling ratio, ``adv_b`` is the cost advantage, ``temp_kl`` is the kl divergence.

    .. tab-item:: FOCOPS

      FOCOPS uses a lagrangian multiplier combined with projection to project the policy back to the safe set.
      
      - First, get the data from buffer and finish pre-computation.

      .. code:: python

         old_distribution_b = Normal(loc=old_mean_b, scale=old_std_b)

         distribution = policy.actor(obs_b)
         log_prob = distribution.log_prob(act_b).sum(dim=-1)
         ratio = torch.exp(log_prob - log_prob_b)
         temp_kl = torch.distributions.kl_divergence(
            distribution, old_distribution_b
         ).sum(-1, keepdim=True)

      - Then, update the policy by using cost advantage and kl divergence.

      .. code:: python

            loss_pi = (temp_kl - (1 / args.focops_lam) * ratio * adv_b) * (
               temp_kl.detach() <= args.focops_eta
            ).type(torch.float32)

      Where ``temp_kl`` is the kl divergence, ``ratio`` is the importance sampling ratio, ``adv_b`` is the reward advantage, ``args.focops_lam`` and ``args.focops_eta`` are the hyperparameters of FOCOPS.

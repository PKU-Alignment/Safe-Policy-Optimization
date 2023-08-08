SafePO Logger
=============

Simple usage
------------

.. code-block:: python

   import safety_gymnasium
   from safepo.common.logger import EpochLogger

   algo = 'Random'
   env_name = 'SafetyPointGoal1-v0'
   seed = 0
   exp_name = 'example'

   log_dir = f'./{algo}/{env_name}/{exp_name}'

   logger = EpochLogger(
      log_dir = log_dir,
      seed = str(seed)
   )

   exp_config = {
      'env_name': env_name,
      'seed': seed,
      'algo': algo,
      'exp_name': exp_name,
   }

   env = safety_gymnasium.make(env_name)
   env.reset(seed=seed)
   d = False

   ep_ret, ep_cost = 0, 0
   while not d:
      a = env.action_space.sample()
      o, r, c, te, tr, info = env.step(a)
      d = te or tr
      ep_ret += r
      ep_cost += c
      
   logger.store(
      **{
         "EpRet": ep_ret,
         "EpCost": ep_cost,
      }
   )
   logger.log_tabular("EpRet")
   logger.log_tabular("EpCost")

   logger.dump_tabular()

API Documentation
-----------------

.. currentmodule:: safepo.common.logger

.. autoclass:: Logger
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: EpochLogger
   :members:
   :undoc-members:
   :show-inheritance:
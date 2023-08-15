Trustworthy Implementation
==========================

To ensure that the implementation is trustworthy, we have compared our 
implementation with open source implementations of the same algorithms.
As some of the algorithms can not be found in open source, we selected
``PPOLag``, ``TRPOLag``, ``CPO`` and ``FOCOPS`` for comparison. 

We have compared the following algorithms:

- ``PPOLag``: `OpenAI Baselines: Safety Starter Agents <https://github.com/openai/safety-starter-agents>`_
- ``TRPOLag``: `OpenAI Baselines: Safety Starter Agents <https://github.com/openai/safety-starter-agents>`_, `RL Safety Algorithms <https://github.com/SvenGronauer/RL-Safety-Algorithms>`_
- ``CPO``: `OpenAI Baselines: Safety Starter Agents <https://github.com/openai/safety-starter-agents>`_, `RL Safety Algorithms <https://github.com/SvenGronauer/RL-Safety-Algorithms>`_
- ``FOCOPS``: `Original Implementation:`

We compared those alforithms in 14 tasks from `Safety-Gymnasium <https://github.com/PKU-Alignment/safety-gymnasium>`_,
they are:

- ``SafetyPointButton1-v0``
- ``SafetyPointCircle1-v0``
- ``SafetyPointGoal1-v0``
- ``SafetyPointPush1-v0``
- ``SafetyCarButton1-v0``
- ``SafetyCarCircle1-v0``
- ``SafetyCarGoal1-v0``
- ``SafetyCarPush1-v0``
- ``SafetyAntVelocity-v1``
- ``SafetyHalfCheetahVelocity-v1``
- ``SafetyHopperVelocity-v1``
- ``SafetyHumanoidVelocity-v1``
- ``SafetyWalker2dVelocity-v1``
- ``SafetySwimmerVelocity-v1``

The results are shown as follows.

.. tab-set::

    .. tab-item:: PPOLag

      .. raw:: html

         <iframe src="https://wandb.ai/pku_rl/SafePO5/reports/PPOLag-Comparison--Vmlldzo1MTM3NDY4" style="border:none;width:90%; height:1000px" title="Performance-PPO-Lag">

      .. raw:: html

         </iframe>

    .. tab-item:: TRPOLag

      .. raw:: html

         <iframe src="https://wandb.ai/pku_rl/SafePO5/reports/TRPOLag-Comparison--Vmlldzo1MTM2NTE1" style="border:none;width:90%; height:1000px" title="Performance-PPO-Lag">

      .. raw:: html

         </iframe>

    .. tab-item:: CPO

      .. raw:: html

         <iframe src="https://wandb.ai/pku_rl/SafePO5/reports/CPO-Comparison--Vmlldzo1MTMzNjIx" style="border:none;width:90%; height:1000px" title="Performance-PPO-Lag">

      .. raw:: html

         </iframe>

    .. tab-item:: FOCOPS

      .. raw:: html

         <iframe src="https://wandb.ai/pku_rl/SafePO5/reports/-FOCOPS-Comparison--Vmlldzo1MTI4OTI0" style="border:none;width:90%; height:1000px" title="Performance-PPO-Lag">

      .. raw:: html

         </iframe>
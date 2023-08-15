General Performance
===================

Safe reinforcement learning algorithms are designed to achieve high reward while satisfying the safety constraint.
In this section, we evaluate the performance of SafePO algorithms on the various environments in `Safety-Gymnasium <https://github.com/PKU-Alignment/safety-gymnasium>`_.

As `Safety-Gymnasium <https://github.com/PKU-Alignment/safety-gymnasium>`_ varies not only the agents but also the tasks.
So we analyse the performance of SafePO algorithms on different tasks vision, while in each task we also analyse different agents.


.. note::

    In this section we only provide the scatter plot of the performance of SafePO algorithms on different tasks,
    in order to reveal the difference between SafePO implemented algorithms.
    For a detailed training curve and detailed analysis,
    please refer to the corresponding section.

Single Agent
------------

Button
~~~~~~

Tha `Button <https://www.safety-gymnasium.com/en/latest/environments/safe_navigation/button.html>`_ task demands the agent to reach the button while avoiding the obstacles.

.. tab-set::

    .. tab-item:: Ant

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/AntButton--Vmlldzo1MDk0MjYx" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">
        
        .. raw:: html

            </iframe>

    .. tab-item:: Car

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/CarButton--Vmlldzo1MDk0MzQy" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">

        .. raw:: html

            </iframe>

    .. tab-item:: Doggo

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/DoggoButton--Vmlldzo1MDkzNjEw" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">

        .. raw:: html

            </iframe>

    .. tab-item:: Point

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/PointButton--Vmlldzo1MDk0Mzgw" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">

        .. raw:: html

            </iframe>

    .. tab-item:: Racecar

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/RacecarButton--Vmlldzo1MDk0MTY1" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">

        .. raw:: html

            </iframe>

Circle
~~~~~~

Tha `Circle <https://www.safety-gymnasium.com/en/latest/environments/safe_navigation/circle.html>`_ task demands the agent to circle around the center of the circle area while avoiding going outside the boundaries. 

.. tab-set::

    .. tab-item:: Ant

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/AntCircle--Vmlldzo1MDk0Mjcw" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">
        
        .. raw:: html

            </iframe>

    .. tab-item:: Car

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/CarCircle--Vmlldzo1MDk0MzQ1" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">

        .. raw:: html

            </iframe>

    .. tab-item:: Doggo

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/DoggoCircle--Vmlldzo1MDkzNTcz" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">

        .. raw:: html

            </iframe>

    .. tab-item:: Point

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/PointCircle--Vmlldzo1MDk0Mzg5" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">

        .. raw:: html

            </iframe>

    .. tab-item:: Racecar

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/RacecarCircle--Vmlldzo1MDkzNTQz" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">

        .. raw:: html

            </iframe>


Goal
~~~~

Tha `Goal <https://www.safety-gymnasium.com/en/latest/environments/safe_navigation/goal.html>`_ task demands the agent to reach the goal while avoiding the obstacles.

.. tab-set::

    .. tab-item:: Ant

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/AntGoal--Vmlldzo1MDkzMjAy" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">
        
        .. raw:: html

            </iframe>

    .. tab-item:: Car

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/CarGoal--Vmlldzo1MDkzMTk4" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">

        .. raw:: html

            </iframe>

    .. tab-item:: Doggo

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/DoggoGoal--Vmlldzo1MDkzMjQ2" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">

        .. raw:: html

            </iframe>

    .. tab-item:: Point

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/PointGoal--Vmlldzo1MDkzMTYz" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">

        .. raw:: html

            </iframe>

    .. tab-item:: Racecar

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/RacecarGoal--Vmlldzo1MDkzMjIw" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">

        .. raw:: html

            </iframe>

Push
~~~~

Tha `Push <https://www.safety-gymnasium.com/en/latest/environments/safe_navigation/push.html>`_ task demands the agent navigate to the goal's location while circumventing hazards.

.. tab-set::

    .. tab-item:: Ant

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/AntPush--Vmlldzo1MDk0Mjcy" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">
        
        .. raw:: html

            </iframe>

    .. tab-item:: Car

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/CarPush--Vmlldzo1MDk0MzUw" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">

        .. raw:: html

            </iframe>

    .. tab-item:: Doggo

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/DoggoPush--Vmlldzo1MDkzNjIx" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">

        .. raw:: html

            </iframe>

    .. tab-item:: Point

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/PointPush--Vmlldzo1MDk0Mzk2" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">

        .. raw:: html

            </iframe>

    .. tab-item:: Racecar

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/RacecarPush--Vmlldzo1MDk0MTc4" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">

        .. raw:: html

            </iframe>

Velocity
~~~~~~~~

Tha Velocity task demands the agent run `MuJoCo Robot <https://gymnasium.farama.org/environments/mujoco/>`_ while avoiding too large angular velocity.

.. tab-set::

    .. tab-item:: Ant

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/AntVelocity--Vmlldzo1MDk2MTMy" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">
        
        .. raw:: html

            </iframe>

    .. tab-item:: HalfCheetah

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/HalfCheetahVelocity--Vmlldzo1MDk2MTQ0" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">

        .. raw:: html

            </iframe>

    .. tab-item:: Hopper

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/HopperVelocity--Vmlldzo1MDk2MTUw" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">
        
        .. raw:: html

            </iframe>

    .. tab-item:: Humanoid

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/HumanoidVelocity--Vmlldzo1MDk2MTYz" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">
        
        .. raw:: html

            </iframe>

    .. tab-item:: Swimmer

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/SwimmerVelocity--Vmlldzo1MDk2MTcz" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">
        
        .. raw:: html

            </iframe>

    .. tab-item:: Walker2d

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/Walker2dVelocity--Vmlldzo1MDk2MTY2" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">
        
        .. raw:: html

            </iframe>

Multi-Agent
-----------

Velocity
~~~~~~~~

.. tab-set::

    .. tab-item:: 2x4Ant

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/AntVelocity--Vmlldzo1MDk2MTMy" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">
        
        .. raw:: html

            </iframe>

    .. tab-item:: 4x2Ant

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/AntVelocity--Vmlldzo1MDk2MTMy" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">
        
        .. raw:: html

            </iframe>

    .. tab-item:: 2x3HalfCheetah

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/2x3HalfCheetahVelocity--Vmlldzo1MDk2MzY2" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">
        
        .. raw:: html

            </iframe>
    
    .. tab-item:: 6x1HalfCheetah

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/6x1HalfCheetahVelocity--Vmlldzo1MDk2NDk1" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">
        
        .. raw:: html

            </iframe>

    .. tab-item:: 3x1Hopper

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/3x1HopperVelocity--Vmlldzo1MDk2NDk5" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">
        
        .. raw:: html

            </iframe>

    .. tab-item:: 9|8Humanoid

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/9-8HumanoidVelocity--Vmlldzo1MDk2Mzky" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">
        
        .. raw:: html

            </iframe>

    .. tab-item:: 2x3Walker2d

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/2x3Walker2dVelocity--Vmlldzo1MDk2NDQ5" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">
        
        .. raw:: html

            </iframe>

    .. tab-item:: 2x1Swimmer

        .. raw:: html

            <iframe src="https://wandb.ai/pku_rl/SafePO/reports/2x1SwimmerVelocity--Vmlldzo1MDk2NDYx" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">
        
        .. raw:: html

            </iframe>


    .. tab-item:: ShadowHand

        .. tab-set::

            .. tab-item:: Over

                .. raw:: html

                    <iframe src="https://wandb.ai/pku_rl/SafePO/reports/ShadowHandOver--Vmlldzo1MDk2NDYz" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">
                
                .. raw:: html

                    </iframe>

            .. tab-item:: CatchUnderarm

                .. raw:: html

                    <iframe src="https://wandb.ai/pku_rl/SafePO/reports/ShadowHandCatchUnderarm--Vmlldzo1MDk2NDcz" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">
                
                .. raw:: html

                    </iframe>
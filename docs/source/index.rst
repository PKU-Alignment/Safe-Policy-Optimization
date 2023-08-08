Welcome to Safe Policy Optimization's documentation!
====================================================

Safe Policy Optimization is a comprehensive algorithm benchmark for Safe Reinforcement Learning (Safe RL).

.. image:: _static/images/logo.png
   :alt: logo
   :width: 700
   :align: center

Overall Algorithms Performance Analysis
---------------------------------------

.. raw:: html

   <iframe src="https://wandb.ai/pku_rl/SafePO5/reports/Overview--Vmlldzo1MDgyNTU3" style="border:none;width:100%; height:500px" title="Performance-PPO-Lag">

.. raw:: html

   </iframe>

This illustration delineates the algorithms within the safepo framework across diverse environmental conditions and tasks, 
while also encompassing a comparative analysis of the distribution of ``EpCost`` throughout the entirety of the training process. 
The area under consideration signifies the degree of concentration exhibited by ``EpCost`` during the course of training.
Upon scrutiny of this graphical representation, several observations emerge: 

.. hint::

   - ``CPO`` exhibits superior stability in contrast to the Lagrangian approach, resulting in a comparatively more concentrated distribution of ``EpCost``; however, it is noteworthy that instances of constraint violation occur with heightened frequency. 
   - The PID Lagrangian method ``CPPOPID`` displays enhanced stability when juxtaposed with the conventional Lagrangian approach. 
   - ``PPOLag``, though marked by pronounced oscillations, demonstrates heightened aptitude in adhering to constraints, as evidenced by a relatively lower overall ``EpCost`` value. 
   - ``PCPO`` closely parallels the characteristics of ``CPO``, while ``FOCOPS`` and CUP can be conceptualized as striking a balance between the ``PPOLag`` method and ``CPO``.

Easy Start
----------

Installation
~~~~~~~~~~~~

To install ``safepo`` from source, run:

.. code-block:: bash

   git clone https://github.com/PKU-Alignment/Safe-Policy-Optimization.git
   cd Safe-Policy-Optimization
   pip install -e .

Training
~~~~~~~~

To train single agent algorithms, e.g. ``PPOLag`` in ``SafetyPointGoal1-v0``, run:

.. code-block:: bash
   
   cd safepo/single_agent
   python ppo_lag.py --env-id SafetyPointGoal1-v0

To train multi-agent alforithms, e.g. ``MAPPOLag`` in ``Safety2x4AntVelocity-v0``, run:

.. code-block:: bash

   cd safepo/multi_agent
   python mappolag.py --scenario Ant --agent-conf 2x4

.. warning::

   We also support isaacgym environment. Please install isaacgym manually before running the code.

   .. code-block:: bash

      cd safepo/multi_agent
      python mappolag.py --task ShadowHandOver

Benchmark running
~~~~~~~~~~~~~~~~~

To run a single agent benchmark based on default configuration, run:

.. code-block:: bash

   cd safepo/single_agent
   python benchmark.py

Similarly, to run a multi-agent benchmark based on default configuration, run:

.. code-block:: bash

   cd safepo/multi_agent
   python benchmark.py

You can also customize the benchmark configuration by modifying the python file in ``benchamrk.py``.

Plot training curves
~~~~~~~~~~~~~~~~~~~~

Each experiemnt would be assigned an experiment name, default to be ``Base`` and stored in ``safepo/runs``.

you can plot the training curves by running:

.. code-block:: bash

      cd safepo/single_agent
      python plot.py --logdir ../runs/Base

If you want to plot the training curves of multi-agent algorithms, run:

.. code-block:: bash
   
      cd safepo/multi_agent
      python plot.py --logdir ../runs/Base

.. toctree::
   :hidden:
   :caption: API

   api/logger
   api/buffer
   api/model
   api/lagrange
   api/env

.. toctree::
   :hidden:
   :caption: ALGORITHM

   algorithms/ppo_lag
   algorithms/trpo_lag

`Github <https://github.com/PKU-Alignment/Safe-Policy-Optimization>`__


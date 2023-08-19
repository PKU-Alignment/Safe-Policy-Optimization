Welcome to SafePO's documentation!
==================================

Safe Policy Optimization is a comprehensive algorithm benchmark for Safe Reinforcement Learning (Safe RL).

.. image:: _static/images/logo.png
   :alt: logo
   :width: 700
   :align: center

Overall Algorithms Performance Analysis
---------------------------------------

.. raw:: html

   <iframe src="https://wandb.ai/pku_rl/SafePO/reports/Overview-of-SafePO-Implementation--Vmlldzo1MTgyMDMy" style="border:none;width:100%; height:500px">

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

One line to run SafePO benchmark:

.. code-block:: bash

   make benchmark

Then you can check the runs in ``safepo/runs``. After that, you can check the 
results (eavluation outcomes, training curves) in ``safepo/results``.


.. toctree::
   :hidden:
   :caption: Usage
   
   usage/train
   usage/eval
   usage/benchmark
   usage/implement
   usage/make

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

   algorithms/curve
   algorithms/lag
   algorithms/first_order
   algorithms/comparision

Related Projects
-----------------

`Github <https://github.com/PKU-Alignment/Safe-Policy-Optimization>`_

`Safety Gymnasisum <https://github.com/PKU-Alignment/safety-gymnasium>`_

`OmniSafe <https://github.com/PKU-Alignment/omnisafe>`_

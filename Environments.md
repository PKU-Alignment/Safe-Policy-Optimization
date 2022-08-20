
### Velocity-Constraint

#### Task Defination

The goal is for an agent to move along a straight line or a two dimensional plane, but the speed of the robot is constrained for safety purposes.

#### reward

The reward consists of three parts:

- **alive bonus**: Every timestep that the walker is alive, it gets a reward of 1,
- **reward_forward**: A reward of walking forward which is measured as (x-coordinate before action - x-coordinate after action)/dt. *dt* is the time between actions and is dependent on the frame_skip parameter (default is 4), where the *dt* for one frame is 0.002 - making the default dt = 4 \* 0.002 = 0.008. This reward would be positive if the walker walks forward (right) desired.
- **reward_control**: A negative reward for penalising the walker if it takes actions that are too large. It is measured as -coefficient **x** sum(action2) where coefficient is a parameter set for the control and has a default value of 0.001

The total reward returned is **reward** *=* alive bonus + reward_forward + reward_control

#### cost

We obtain the velocity information as follows:

```python
#next_obs, rew, done, info = env.step(act)
if 'y_velocity' not in info:
	cost = np.abs(info['x_velocity'])
else:
	cost = np.sqrt(info['x_velocity'] ** 2 + info['y_velocity'] ** 2)
```


### Safety Gym

|                          PointGoal                           |                         PointButton                          |                          PointPush                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="assets/envs/safety_gym/pointgoal.gif" align="middle" width="200" border="1"/> | <img src="assets/envs/safety_gym/pointbutton.gif" align="middle" width="200" border="1"/> | <img src="assets/envs/safety_gym/pointpush.gif" align="middle" width="200" border="1"/> |
|                         **CarGoal**                          |                        **CarButton**                         |                         **CarPush**                          |
| <img src="assets/envs/safety_gym/cargoal.gif" align="middle" width="200" border="1"/> | <img src="assets/envs/safety_gym/carbutton.gif" align="middle" width="200" border="1"/> | <img src="assets/envs/safety_gym/carpush.gif" align="middle" width="200" border="1"/> |
|                        **DoggoGoal**                         |                       **DoggoButton**                        |                        **DoggoPush**                         |
| <img src="assets/envs/safety_gym/doggogoal.gif" align="middle" width="200" border="1"/> | <img src="assets/envs/safety_gym/doggobutton.gif" align="middle" width="200" border="1"/> | <img src="assets/envs/safety_gym/doggopush.gif" align="middle" width="200" border="1"/> |

#### More details

In each task:Goal, Button, Push, there are three levels of difficulty(with higher levels having more challenging constraints).

- `Safexp-{Robot}Goal0-v0`: A robot must navigate to a goal.

- `Safexp-{Robot}Goal1-v0`: A robot must navigate to a goal while avoiding hazards. One vase is present in the scene, but the agent is not penalized for hitting it.

- `Safexp-{Robot}Goal2-v0`: A robot must navigate to a goal while avoiding more hazards and vases.

- `Safexp-{Robot}Button0-v0`: A robot must press a goal button.

- `Safexp-{Robot}Button1-v0`: A robot must press a goal button while avoiding hazards and gremlins, and while not pressing any of the wrong buttons.

- `Safexp-{Robot}Button2-v0`: A robot must press a goal button while avoiding more hazards and gremlins, and while not pressing any of the wrong buttons.

- `Safexp-{Robot}Push0-v0`: A robot must push a box to a goal.

- `Safexp-{Robot}Push1-v0`: A robot must push a box to a goal while avoiding hazards. One pillar is present in the scene, but the agent is not penalized for hitting it.

- `Safexp-{Robot}Push2-v0`: A robot must push a box to a goal while avoiding more hazards and pillars.

  (To make one of the above, make sure to substitute `{Robot}` for one of `Point`, `Car`, or `Doggo`.) If you want find more information about Safety Gym, you can check [this](https://github.com/openai/safety-gym).

### Safe Bullet Gym

#### Agent

|                             Ball                             |                             Car                              |                            Drone                             |                             Ant                              |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="assets/envs/bullet_gym/ball.png" align="middle" width="200" border="1"/> | <img src="assets/envs/bullet_gym/car.png" align="middle" width="200" border="1"/> | <img src="assets/envs/bullet_gym/drone.png" align="middle" width="200" border="1"/> | <img src="assets/envs/bullet_gym/ant.png" align="middle" width="200" border="1"/> |

#### Task

| Circle                                                       | Gather                                                       | Reach                                                        | Run                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="assets/envs/bullet_gym/circle.png" align="middle" width="200" border="1"/> | <img src="assets/envs/bullet_gym/gather.png" align="middle" width="200" border="1"/> | <img src="assets/envs/bullet_gym/reach.png" align="middle" width="200" border="1"/> | <img src="assets/envs/bullet_gym/run.png" align="middle" width="200" border="1"/> |


#### More Description

- **Ball**: A spherical shaped agent which can freely move on the xy-plane.

- **Car**: A four-wheeled agent based on MIT's Racecar.

- **Drone**: An air vehicle based on the AscTec Hummingbird quadrotor.

- **Ant**: A four-legged animal with a spherical torso.

- **Circle**: Agents are expected to move on a circle in clock-wise direction (as proposed by Achiam et al. (2017)). The reward is dense and increases by the agent's velocity and by the proximity towards the boundary of the circle. Costs are received when agent leaves the safety zone defined by the two yellow boundaries.

- **Gather** Agents are expected to navigate and collect as many green apples as possible while avoiding red bombs (Duan et al. 2016). In contrast to the other tasks, agents in the gather tasks receive only sparse rewards when reaching apples. Costs are also sparse and received when touching bombs (Achiam et al. 2017).

- **Reach**: Agents are supposed to move towards a goal (Ray et al. 2019). As soon the agents enters the goal zone, the goal is re-spawned such that the agent has to reach a series of goals. Obstacles are placed to hinder the agent from trivial solutions. We implemented obstacles with a physical body, into which agents can collide and receive costs, and ones without collision shape that produce costs for traversing. Rewards are dense and increase for moving closer to the goal and a sparse component is obtained when entering the goal zone.

- **Run**: Agents are rewarded for running through an avenue between two safety boundaries (Chow et al. 2019). The boundaries are non-physical bodies which can be penetrated without collision but provide costs. Additional costs are received when exceeding an agent-specific velocity threshold.

  If you want to find more information, you can check [this](https://github.com/SvenGronauer/Bullet-Safety-Gym).

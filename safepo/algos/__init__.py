from safepo.algos.policy_graident import PG
from safepo.algos.natural_pg import NPG
from safepo.algos.trpo import TRPO
from safepo.algos.ppo import PPO
from safepo.algos.trpo_lagrangian import TRPO_Lagrangian
from safepo.algos.ppo_lagrangian import PPO_Lagrangian
from safepo.algos.cpo import CPO
from safepo.algos.pcpo import PCPO
from safepo.algos.focops import FOCOPS
from safepo.algos.p3o import P3O

REGISTRY = {
    'pg': PG,
    'npg': NPG,
    'trpo': TRPO,
    'ppo': PPO,
    'trpo_lagrangian':TRPO_Lagrangian,
    'ppo_lagrangian': PPO_Lagrangian,
    'cpo': CPO,
    'pcpo': PCPO,
    'focops': FOCOPS,
    'p3o': P3O
}
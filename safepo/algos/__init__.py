from safepo.algos.policy_gradient import PG
from safepo.algos.natural_pg import NPG
from safepo.algos.trpo import TRPO
from safepo.algos.ppo import PPO
from safepo.algos.trpo_lag import TRPO_Lag
from safepo.algos.ppo_lag import PPO_Lag
from safepo.algos.cpo import CPO
from safepo.algos.pcpo import PCPO
from safepo.algos.focops import FOCOPS
from safepo.algos.p3o import P3O
from safepo.algos.ipo import IPO
from safepo.algos.cppo_pid import CPPOPid
REGISTRY = {
    'pg': PG,
    'npg': NPG,
    'trpo': TRPO,
    'ppo': PPO,
    'trpo-lag':TRPO_Lag,
    'ppo-lag': PPO_Lag,
    'cpo': CPO,
    'pcpo': PCPO,
    'focops': FOCOPS,
    'p3o': P3O,
    'ipo': IPO,
    'cppo-pid': CPPOPid
}
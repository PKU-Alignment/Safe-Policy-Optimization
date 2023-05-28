# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from safepo.algorithms.cpo import CPO
from safepo.algorithms.cppo_pid import CPPOPid
from safepo.algorithms.focops import FOCOPS
from safepo.algorithms.ipo import IPO
from safepo.algorithms.natural_pg import NPG
from safepo.algorithms.p3o import P3O
from safepo.algorithms.pcpo import PCPO
from safepo.algorithms.policy_gradient import PG
from safepo.algorithms.ppo import PPO
from safepo.algorithms.ppo_lag import PPO_Lag
from safepo.algorithms.trpo import TRPO
from safepo.algorithms.trpo_lag import TRPO_Lag

REGISTRY = {
    "pg": PG,
    "npg": NPG,
    "trpo": TRPO,
    "ppo": PPO,
    "trpo-lag": TRPO_Lag,
    "ppo-lag": PPO_Lag,
    "cpo": CPO,
    "pcpo": PCPO,
    "focops": FOCOPS,
    "p3o": P3O,
    "ipo": IPO,
    "cppo-pid": CPPOPid,
}

import subprocess

def test_ppo_lag():
    subprocess.run(
        "python safepo/single_agent/ppo_lag.py --total-steps 2000 --num-envs 2 --steps-per-epoch 2000",
        shell=True,
        check=True,
    )

def test_cpo():
    subprocess.run(
        "python safepo/single_agent/cpo.py --total-steps 1000 --num-envs 1 --steps-per-epoch 1000",
        shell=True,
        check=True,
    )

def test_cup():
    subprocess.run(
        "python safepo/single_agent/cup.py --total-steps 1000 --num-envs 1 --steps-per-epoch 1000",
        shell=True,
        check=True,
    )

def test_focops():
    subprocess.run(
        "python safepo/single_agent/focops.py --total-steps 1000 --num-envs 1 --steps-per-epoch 1000",
        shell=True,
        check=True,
    )

def test_trpo_lag():
    subprocess.run(
        "python safepo/single_agent/trpo_lag.py --total-steps 1000 --num-envs 1 --steps-per-epoch 1000",
        shell=True,
        check=True,
    )

def test_rcpo():
    subprocess.run(
        "python safepo/single_agent/rcpo.py --total-steps 1000 --num-envs 1 --steps-per-epoch 1000",
        shell=True,
        check=True,
    )

def test_pcpo():
    subprocess.run(
        "python safepo/single_agent/pcpo.py --total-steps 1000 --num-envs 1 --steps-per-epoch 1000",
        shell=True,
        check=True,
    )

def test_cppo_pid():
    subprocess.run(
        "python safepo/single_agent/cppo_pid.py --total-steps 1000 --num-envs 1 --steps-per-epoch 1000",
        shell=True,
        check=True,
    )

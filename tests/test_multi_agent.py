import subprocess

def test_happo():
    subprocess.run(
        "python safepo/multi_agent/happo.py --total-steps 2000 --num-envs 1",
        shell=True,
        check=True,
    )
    subprocess.run(
        "python safepo/multi_agent/happo.py --total-steps 2000 --num-envs 2",
        shell=True,
        check=True,
    )

def test_mappo():
    subprocess.run(
        "python safepo/multi_agent/mappo.py --total-steps 2000 --num-envs 1",
        shell=True,
        check=True,
    )
    subprocess.run(
        "python safepo/multi_agent/mappo.py --total-steps 2000 --num-envs 2",
        shell=True,
        check=True,
    )

def test_mappolag():
    subprocess.run(
        "python safepo/multi_agent/mappolag.py --total-steps 2000 --num-envs 1",
        shell=True,
        check=True,
    )
    subprocess.run(
        "python safepo/multi_agent/mappolag.py --total-steps 2000 --num-envs 2",
        shell=True,
        check=True,
    )

def test_macpo():
    subprocess.run(
        "python safepo/multi_agent/macpo.py --total-steps 2000 --num-envs 1",
        shell=True,
        check=True,
    )
    subprocess.run(
        "python safepo/multi_agent/macpo.py --total-steps 2000 --num-envs 2",
        shell=True,
        check=True,
    )

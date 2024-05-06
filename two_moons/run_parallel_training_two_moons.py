import multiprocessing
import subprocess
import os

os.chdir('../')

# for final results :

NUM_SIM_FULL = [100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000, 20000]
NUM_SIM_SMALL = [100, 200, 300, 500, 1000, 2000, 3000, 5000]
NUM_TRAINING_FULL = 3
NUM_TRAINING_SMALL = 8
HUGE_SIM = 100_000


# to test script on local :
"""
NUM_SIM_FULL = [10, 100]
NUM_SIM_SMALL = [10]
NUM_TRAINING_FULL = 1
NUM_TRAINING_SMALL = 2
HUGE_SIM = 200
"""

SCRIPT_TYPES = ["two_moons/hp_two_moons_train_two_step.py", "two_moons/hp_two_moons_train_standard.py"]


def run_script(args):
    python_script, params = args
    command = f"python {python_script} {params}"
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    num_processes = NUM_TRAINING_SMALL
    num_sims = NUM_SIM_FULL

    scripts_and_params = []

    for process_id in range(1, num_processes + 1):
        if process_id >= NUM_TRAINING_FULL:
            num_sims = NUM_SIM_SMALL
        for num_sim in num_sims:
            for script_type in SCRIPT_TYPES:
                script_and_params = (script_type, f"{num_sim} {process_id}")
                scripts_and_params.append(script_and_params)

    for script_type in SCRIPT_TYPES:
        script_and_params = (script_type, f"{HUGE_SIM} {1}")
        scripts_and_params.append(script_and_params)

    pool = multiprocessing.Pool(processes=len(scripts_and_params))
    pool.map(run_script, scripts_and_params)




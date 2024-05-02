import multiprocessing
import subprocess
import os


def run_script(args):
    python_script, params = args
    command = f"python {python_script} {params}"
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    num_processes = 10
    num_iterations = [100, 200, 300, 500, 1000, 2000, 3000, 5000]
    script_types = [path+"/hp_two_moons_train_two_step.py", path+"/hp_two_moons_train_standard.py"]

    scripts_and_params = []

    for process_id in range(1, num_processes + 1):
        if process_id >= 3:
            num_iterations = [100, 200, 300, 500]
        for iteration in num_iterations:
            for script_type in script_types:
                script_and_params = (script_type, f"{process_id} {iteration}")
                scripts_and_params.append(script_and_params)

    pool = multiprocessing.Pool(processes=len(scripts_and_params))
    pool.map(run_script, scripts_and_params)


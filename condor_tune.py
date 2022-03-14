import ray      # pip install "ray[tune]"
import htcondor # conda install -c conda-forge htcondor

from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch

import os, pathlib, json, time, pickle
from typing import Dict, Any

from misc import *
from config import *

############################################################################################
## DIRECTORY SETUP
############################################################################################

create_base_directories     = True
move_current_trials_to_old  = True

if create_base_directories: pathlib.Path(TRIAL_DIR).mkdir(parents=True, exist_ok=True)
if move_current_trials_to_old: move_trials()

############################################################################################
## DEFINE TRIAL FUNCTION
############################################################################################

def run_trial(params: Dict[str, Any], checkpoint_dir=None) -> None:
    """Submit and wait for a condor job to finish, then report results"""

    ############################################################################################
    ## PREPARE PARAMS JSON
    ############################################################################################

    trial_hash = dict_hash(params)
    params['hash'] = trial_hash
    THIS_TRIAL_DIR = f'{TRIAL_DIR}/{trial_hash}'

    # tune automatically converts ints to floats, so we enforce types as needed here
    params['modeltype']  = int(params['modeltype'])
    params['batch_size'] = int(params['batch_size'])
    params['epochs']     = int(params['epochs'])

    pathlib.Path(THIS_TRIAL_DIR).mkdir(parents=True, exist_ok=True)
    with open(f'{THIS_TRIAL_DIR}/params.json', 'w') as f:
        json.dump(params, f) # we will read this file in the training script

    ############################################################################################
    ## SETUP HTCONDOR SCHEDD
    ############################################################################################

    schedd = htcondor.Schedd() # we use xquery since we only need to check if the result is empty
    def is_running(job_id): return not empty_gen(schedd.xquery(f'ClusterId == {job_id}', projection=["ProcId", "JobStatus"]))
    def remove_job(job_id): schedd.act(htcondor.JobAction.Remove, f'ClusterId == {job_id}')

    ############################################################################################
    ## SUBMIT TRAINING JOB
    ############################################################################################

    train_job = htcondor.Submit(
    # Same syntax as the usual condor_submit file.
    # We use Python variables here to dynamically set command line arguments
        f"""
        universe = vanilla
        getenv = true
        executable = /bin/bash
        arguments = {TUNE_DIR}/train.sh $(Cluster) $(Process) {params['data_path']}

        request_gpus = 1
        request_memory = 8192

        log = {THIS_TRIAL_DIR}/train.log
        output = {THIS_TRIAL_DIR}/train.out
        error = {THIS_TRIAL_DIR}/train.err

        queue
        """
    # If successful, this job will write several files, including:
    # THIS_TRIAL_DIR/training_done: prescence indicates that the training job is done
    # THIS_TRIAL_DIR/run_flow.cmd: HTCondor Submit file for flow job array
    # THIS_TRIAL_DIR/flows/: a directory with one json file for each example in the validation set
    # THIS_TRIAL_DIR/training_results.pkl: results of the training job, including the trained model
    )
    result = schedd.submit(train_job, count=1)
    job_id = result.cluster()

    ############################################################################################
    ## DETECT WHEN TRAINING JOB IS DONE
    ############################################################################################
    job_start_time = time.time()
    job_timeout    = lambda : (time.time() - job_start_time) > JOB_TIMEOUT
    while is_running(job_id) and not job_timeout():
        time.sleep(TIME_BETWEEN_QUERIES)

    if job_timeout() and is_running(job_id):
        remove_job(job_id)
        raise Exception('Training took too long to complete')

    # The training script exits on error, which exits the condor job without writing training_done.
    # If it terminates normally, it will write a file called training_done, then exit the condor job.
    # We use this to detect if training was successful.
    file_start_time = time.time()
    file_timeout    = lambda : (time.time() - file_start_time) > FILE_TIMEOUT
    while not pathlib.Path(f'{THIS_TRIAL_DIR}/training_done').is_file() and not file_timeout():
        time.sleep(TIME_BETWEEN_QUERIES)

    if file_timeout(): raise Exception("Job ended but training failed to complete")

    ############################################################################################
    ## SUBMIT FLOW JOBS
    ############################################################################################

    # Each job reads a file like THIS_TRIAL_DIR/flows/42.json and
    # writes a file like THIS_TRIAL_DIR/flows/results/42.json.

    # NOTE: we use bash, because it's easier to handle than the python interface for job arrays
    flow_output = os.popen(f'cd {THIS_TRIAL_DIR} && condor_submit run_flow.cmd').read()
    flow_jobid = flow_output.split(' ')[-1].split('\n')[0].strip('.')
    # NOTE: above line may change if condor_submit changes
    
    ############################################################################################
    ## DETECT WHEN FLOW JOBS ARE DONE
    ############################################################################################

    flow_start_time = time.time()
    flow_timeout    = lambda : (time.time() - flow_start_time) > FLOW_TIMEOUT
    while is_running(flow_jobid) and not flow_timeout():
        time.sleep(TIME_BETWEEN_QUERIES)

    if flow_timeout() and is_running(flow_jobid):
        remove_job(flow_jobid)
        raise Exception('Flows took too long to complete')

    ############################################################################################
    ## SUBMIT METRICS JOB
    ############################################################################################

    metric_job = htcondor.Submit(
    # This job reads the json files in THIS_TRIAL_DIR/flows/, and computes metrics
        f"""
        universe = vanilla
        getenv = true
        executable = /bin/bash
        arguments = {TUNE_DIR}/metric.sh {THIS_TRIAL_DIR}
        request_memory = 2048

        log = {THIS_TRIAL_DIR}/metric.log
        output = {THIS_TRIAL_DIR}/metric.out
        error = {THIS_TRIAL_DIR}/metric.err

        queue
        """
    # If successful, this job will write several files, including:
    # THIS_TRIAL_DIR/results.pkl: results of the metric job, including all metrics
    )
    metric_result = schedd.submit(metric_job, count=1)
    metric_jobid  = metric_result.cluster()

    ############################################################################################
    ## DETECT WHEN METRICS JOB IS DONE
    ############################################################################################

    # This uses the same logic as the training job above.
    metric_start_time = time.time()
    metric_timeout    = lambda : (time.time() - metric_start_time) > METRIC_TIMEOUT
    while is_running(metric_jobid) and not metric_timeout():
        time.sleep(TIME_BETWEEN_QUERIES)

    if metric_timeout() and is_running(metric_jobid):
        remove_job(metric_jobid)
        raise Exception('Metric calculation took too long to complete')

    file_start_time = time.time()
    file_timeout    = lambda : (time.time() - file_start_time) > FILE_TIMEOUT
    while not pathlib.Path(f'{THIS_TRIAL_DIR}/results.pkl').is_file() and not file_timeout():
        time.sleep(TIME_BETWEEN_QUERIES)

    if file_timeout(): raise Exception("Job ended but metric calculation failed to complete")

    ############################################################################################
    ## REPORT RESULTS
    ############################################################################################

    result_dict = torch_unpickle(f'{THIS_TRIAL_DIR}/results.pkl')
    tune.report( l1_loss=result_dict['l1_loss'],          # based on training
                tot_viol=result_dict['tot_viol'],         # based on training
        flow_convergence=result_dict['flow_convergence'], # based on flow jobs
           flow_obj_diff=result_dict['flow_obj_diff'],    # based on flow jobs
                weighted=result_dict['weighted'],         # based on training and flows
                    time=result_dict['training_time'])    # based on training

############################################################################################
## BEGIN TRIALS
############################################################################################

ray.init(num_cpus=RAY_NUM_CPUS)
analysis = tune.run(run_trial, config=config, name=EXPERIMENT_NAME,
                           search_alg=HyperOptSearch(points_to_evaluate=[initial],
                               metric=METRIC, mode=METRIC_MODE),
                          num_samples=NUM_TRIALS,
                raise_on_failed_trial=False,
                  resources_per_trial={'cpu': RAY_NUM_CPUS/MAX_PARALLEL_TRAILS},)

############################################################################################
## SAVE RESULTS
############################################################################################

with open(RESULTS_FILE, 'wb') as f:
    pickle.dump(analysis.results_df, f)

# condor-tune

> [!WARNING] 
> This project is not actively maintained. It was written with Ray Tune 1.11.0 and HTCondor 9.5.0. Since then, several vulnerabilities have been discovered (and [patched](https://www.anyscale.com/blog/update-on-ray-cves-cve-2023-6019-cve-2023-6020-cve-2023-6021-cve-2023-48022-cve-2023-48023)) in Ray. `requirements.txt` has been updated to grab the latest version of Ray and HTCondor, though Ray's Tune API is likely to have changed.

This example project demonstrates the `condor-tune` framework for
hyperparameter optimization. It comes from an application in ML for
Power Systems research, where we first train a neural network using a
GPU, then run parallel power flows using the trained network on every
example in its respective validation set. We then must compute a weighted
metric incorporating results from all power flows as well as raw model performance.

See [condor_tune.py](condor_tune.py), [Ray Tune](https://docs.ray.io/en/latest/tune/index.html), and [HTCondor](https://htcondor.readthedocs.io/en/feature/apis/python-bindings/index.html) for relevant documentation.

Ray Tune restricts us to run all jobs on a single machine, so we need
to spawn a condor job for each trial and automatically collect results
from the trials. We then have access to Ray Tune to automatically
optimize hyperparameters. Due to this structure, we can spawn multiple
condor jobs per trial, even spawning job arrays if needed. This allows
us to leverage the resources of the entire cluster rather than being
limited to a single machine with a limited number of CPUs/GPUs. We can
specify hardware requirements for each stage of the trial, freeing up
resources for other trials and other cluster users.

The submitting process for a job array is a bit more involved. We have to prepare the necessary submit file and data files during the first (training) job:
1. At the end of the training job, we write an HTCondor submit file defining a job array to
   `THIS_TRIAL_DIR/run_flow.cmd` (i.e. `queue args from file.txt`)
2. Suppose every job in this array reads a json file like `THIS_TRIAL_DIR/flows/42.json`
   and runs a job with the required parameters. We can then create the `flows`
   directory and write the json files at the end of the training job.
3. Once we are done preparing the `run_flow.cmd` file and the `flows` directory, we create
   the flag file `THIS_TRIAL_DIR/training_done` and exit the training job.
4. Finally, this script submits the job array via the bash interface.
   
The scripts `train.sh` and `metric.sh` are used to run the neural
network training and metric calculations respectively. They are
located in `TUNE_DIR`. They look something like:

    eval "$(conda shell.bash hook)"
    conda activate project_env
    python /home/klamike/condor-tune-project/train.py "$@"

The full folder structure is outlined at the bottom of [misc.py](misc.py).
We omit the actual training, flow, and metric calculation scripts as they are out of scope for this project.

## TODO: 
- Add checkpointing to support early stopping/restarting
- Quantify maximum number of trials per CPU (tested up to 12 jobs with 2 CPUs)
- Graceful exit on keyboard interrupt, optionally killing all condor jobs spawned


Michael Klamkin 2022  
Georgia Institute of Technology  
H. Milton Stewart School of Industrial Engineering  
Pascal Van Hentenryck Lab, Risk-Aware Market Clearing (RAMC) Group

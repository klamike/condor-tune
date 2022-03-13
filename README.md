# condor_tune

This example project demonstrates the `condor_tune` framework for
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

The submitting process for a job array is a bit more involved:
1. At the end of the training job, we write an HTCondor submit file defining a job array to
   `THIS_TRIAL_DIR/run_flow.cmd`.
2. Suppose every job in this array reads a json file like `THIS_TRIAL_DIR/flows/42.json`
   and runs a job with the required parameters. We can then create the `flows`
   directory and write the json files at the end of the training job.
3. Once we are done preparing run_flow.cmd and the flows directory, we create
   the flag file `THIS_TRIAL_DIR/training_done` and exit.
4. The HTCondor Job then ends, and the job array is submitted to
   HTCondor by this script via the bash interface.
   
The scripts `train.sh` and `metric.sh` are used to run the neural
network training and metric calculations respectively. They are
located in `TUNE_DIR`. They look something like:

    eval "$(conda shell.bash hook)"
    conda activate project_env
    python /home/klamike/condor-tune-project/train.py "$@"

We omit the actual training, flow, and metric calculation scripts as they are out of scope for this project.

TODO: 
- Add checkpointing to support early stopping/restarting
- Quantify maximum number of trials per CPU (tested up to 12 jobs with 2 CPUs)
- Graceful exit on keyboard interrupt, optionally killing all condor jobs spawned

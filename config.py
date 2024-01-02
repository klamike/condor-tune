# Copyright 2022 Michael Klamkin

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import datetime
from ray import tune

MAX_PARALLEL_TRAILS = 50
NUM_TRIALS          = 200

METRIC_MODE, METRIC = "min", "weighted"
EXPERIMENT_NAME     = f'tune_{METRIC}_model_batch_lr'

JOB_TIMEOUT    = 60 * 60 * 12  # in seconds (12hr) - this is time spent in the queue + running the training job
FILE_TIMEOUT   = 60 * 5        # in seconds (5min) - this is time after the job to wait for results
FLOW_TIMEOUT   = 60 * 60 * 12  # in seconds (12hr) - this is time spent in the queue + running the flow jobs
METRIC_TIMEOUT = 60 * 60 * 12  # in seconds (12hr) - this is time spent in the queue + running the metric job

TIME_BETWEEN_QUERIES = 5  # in seconds

RAY_NUM_CPUS = 2  # must use at least two - one for manager, one for trials

BASE_DIR  = "/home/klamike/condor-tune-project"
TUNE_DIR  = BASE_DIR + "/tune"
TRIAL_DIR = TUNE_DIR + "/trials"

RESULTS_FILE = BASE_DIR + '/results/condor_tune_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.pkl'

config = { # Specify the hyperparameters, with tuning options
    "data_path":     '/home/klamike/data.tar.gz',
    "epochs":        5000,
    "verbose":       False,

    "optimizer":     'AdamW',
    "activation":    'LeakyReLU',
    "loss_function": 'mse',

    "modeltype":     tune.quniform(2, 6, 1),
    "batch_size":    tune.loguniform(8, 512),
    "learning_rate": tune.choice(['logspace(-4,-7)', 'logspace(-3,-5)',
               'logspace(-3,-7)', 'logspace(-4,-6)', 'logspace(-3,-6)'])
}

initial = { # Specify the initial values for all hyperparameters
    "data_path":     '/home/klamike/data.tar.gz',
    "epochs":        5000,
    "verbose":       False,

    "optimizer":     'AdamW',
    "activation":    'LeakyReLU',
    "loss_function": 'mse',

    "modeltype":     4,
    "batch_size":    64,
    "learning_rate": 'logspace(-4,-7)'
}
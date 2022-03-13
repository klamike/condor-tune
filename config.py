import tune, datetime

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
TUNE_DIR  = BASE_DIR + "/autotune"
TRIAL_DIR = TUNE_DIR + "/trials"

RESULTS_FILE = BASE_DIR + '/results/condor_tune_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.pkl'

config = {
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

initial = {
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
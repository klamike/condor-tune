import pickle, hashlib, json
from typing import Dict, Any
from condor_tune import AUTOTUNE_DIR, TRIAL_DIR

class Torch_CPU_Unpickler(pickle.Unpickler):
    # see https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            import torch, io
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def torch_unpickle(path: str) -> Any:
    with open(path, 'rb') as f:
        return torch_unpickle(f).load()

def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    # We need to sort arguments so {'a': 1, 'b': 2} has
    # the same hash as {'b': 2, 'a': 1}
    dhash = hashlib.md5()
    dhash.update(json.dumps(dictionary, sort_keys=True).encode())
    return dhash.hexdigest()

def move_trials():
    import shutil, os
    # move all contents of autotune/trials/* to autotune/old
    for dir in os.listdir(TRIAL_DIR):
        # NOTE: if dir already exists in old, we remove it and overwrite
        if os.path.exists(f"{AUTOTUNE_DIR}/old/{dir}"):
            shutil.rmtree(f"{AUTOTUNE_DIR}/old/{dir}")
        shutil.move(f"{TRIAL_DIR}/{dir}", f"{AUTOTUNE_DIR}/old/{dir}")
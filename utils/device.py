import torch

def get_device(requested = "auto"): 
    """ 
    Automatically configures the device to be used by torch.
    Args: 
        requested (str): requested device to be used (cuda, mps, or cpu)
            if auto: checks the following hierarchy: cuda -> mps -> cpu"""

    if requested == "auto": 
        if torch.cuda.is_available(): 
            return torch.device("cuda")
        if torch.backends.mps.is_available(): 
            return torch.device("mps")
        return torch.device("cpu")
    
    if requested == "cuda" and torch.cuda.is_availabe(): 
        return torch.device("cuda")

    if requested == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
    
    return torch.device("cpu")

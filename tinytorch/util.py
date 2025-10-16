import dill
import os

def save(obj, filename):
    try:
        with open(filename, "wb") as f:
            dill.dump(obj, f)
    except Exception as e:
        print(f"Error saving object to {filename}: {e}")
        
def load(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No such file: {filename}")
    try:
        with open(filename, "rb") as f:
            obj = dill.load(f)
        return obj
    except Exception as e:
        print(f"Error loading object from {filename}: {e}")
        return None
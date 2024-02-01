import os

def get_update_dict(d, k, v={}):
    try:
        v = d[k]
    except:
        d[k] = v
    return v


def get_dict(d, k, v=-1):
    try:
        v = d[k]
    except:
        pass
    return v

def get_insert_indices_dict(d,k):
    try:
        v = d[k]
    except:
        v = len(d)
        d[k] = v
    return v

def ensuredir(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
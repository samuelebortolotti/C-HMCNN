'''Helper functions.
'''

import pickle

def unpickle(file):
    '''Unpickle the given file
    '''

    with open(file, 'rb') as f:
        res = pickle.load(f, encoding='bytes')
    return res

def read_meta(metafile):
    '''Read the meta file and return the coarse and fine labels.
    '''
    meta_data = unpickle(metafile)
    fine_label_names = [t.decode('utf8') for t in meta_data[b'fine_label_names']]
    coarse_label_names = [t.decode('utf8') for t in meta_data[b'coarse_label_names']]
    return coarse_label_names, fine_label_names

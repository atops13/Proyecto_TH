import os
import torch
import numpy as np
p = 'features/processed_data.pt'
print('file exists:', os.path.exists(p))
if not os.path.exists(p):
    raise SystemExit('features file not found')
features, labels = torch.load(p)
print('features type:', type(features), 'shape:', getattr(features,'shape',None))
print('labels type:', type(labels), 'shape:', getattr(labels,'shape',None))
labels = np.array(labels)
unique, counts = np.unique(labels, return_counts=True)
print('unique labels:', unique.tolist())
print('counts:', dict(zip(unique.tolist(), counts.tolist())))
print('total samples:', labels.shape[0])

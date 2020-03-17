import os
from collections import Counter

buf = []
a = os.listdir(os.path.abspath('/home/aashukha/megares/'))

for l in sorted(a):
    if '.pt' in l:
        l = l.split('-')
        buf += ['-'.join(l[:2])]

for k, v in Counter(buf).items():
    print(k, v)
import os

cmds = [
    'python CLNode_cora_percent.py --percent=2 --backbone=GCN --lam=0.25 --T=50 --scheduler=geom ',
    'python CLNode_cora_percent.py --percent=2 --backbone=GCN --lam=0.25 --T=100 --scheduler=geom ',
    'python CLNode_cora_percent.py --percent=2 --backbone=GCN --lam=0.25 --T=150 --scheduler=geom ',
    'python CLNode_cora_percent.py --percent=2 --backbone=GCN --lam=0.5 --T=50 --scheduler=geom ',
    'python CLNode_cora_percent.py --percent=2 --backbone=GCN --lam=0.5 --T=100 --scheduler=geom ',
    'python CLNode_cora_percent.py --percent=2 --backbone=GCN --lam=0.5 --T=150 --scheduler=geom ',
    'python CLNode_cora_percent.py --percent=2 --backbone=GCN --lam=0.75 --T=50 --scheduler=geom ',
    'python CLNode_cora_percent.py --percent=2 --backbone=GCN --lam=0.75 --T=100 --scheduler=geom ',
    'python CLNode_cora_percent.py --percent=2 --backbone=GCN --lam=0.75 --T=150 --scheduler=geom ',
]

for cmd in cmds:
    for i in range(10):
        os.system(cmd)
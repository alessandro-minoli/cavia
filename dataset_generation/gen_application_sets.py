import os
import random
from collections import defaultdict

SRC = "../dataset/app"
DEST = "../dataset/app_sets2"
LOWER,UPPER = 3,10
MIN_NAPPS, MAX_NAPPS = 5,40
INSTANCES_PER_NAPPS = 3

random.seed(42)

all_apps = []
for name in os.listdir(SRC):
    if name.endswith(".dat") and name[-9:-5] == "_rp_":
        number_of_microservices = int(name.split('_')[2])
        all_apps.append((name,number_of_microservices))

instances = defaultdict(list)
for n_apps in range(MIN_NAPPS,MAX_NAPPS+1):
    target = round(n_apps*(LOWER+UPPER)/2) # mean of n_apps discrete uniform RVs in [lower,upper]
    while len(instances[n_apps]) < INSTANCES_PER_NAPPS:
        s = random.sample(all_apps,n_apps)
        if sum(x[1] for x in s) == target:
            instances[n_apps].append(s)

id = 0
for k,v in instances.items():
    for ins in v:
        with open(os.path.join(DEST,f"id_{id}_{k}_apps.dat"), 'w') as f:
            for app,_ in ins:
                f.write(f"{app}\n")
        id += 1
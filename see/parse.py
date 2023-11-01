import re
import matplotlib.pyplot as plt

path = "/home/clive/proj/see/faiss/build/demos/default_hnsw_sift1m.log"

steps = {}

cnt = 0

with open(path, 'r') as file:
    for line in file:
        entries = line.split(" ")
        for entry in entries:
            if '$' in entry:
                cnt += 1
                level = int(entry.split('$')[1])
                step = int(entry.split('$')[2])
                if level in steps:
                    steps[level].append(step)
                else:
                    steps[level] = [step]
            elif '%' in entry:
                level = 0
                step = int(entry.replace('%', ''))
                if level in steps:
                    steps[level].append(step)
                else:
                    steps[level] = [step]


for l in steps:
    plt.hist(steps[l], edgecolor='k', alpha=0.75)
    plt.savefig('dist_' + str(l)  + '.png')
    plt.clf()

print("total: ", cnt)



import math
import numpy as np

def select(problist):
    rand = np.random.rand(1)
    tgt = 0
    listlen = len(problist)
    probsum = sum(problist)
    if probsum==0:
        tgt = -1
    else:
        culprob = [0.0 for i in range(listlen)]
        for i in range(listlen):
            if i==0:
                culprob[i] = 1.0*problist[i]/probsum
            else:
                culprob[i] = culprob[i-1] + 1.0*problist[i]/probsum
        left = 0
        right = listlen-1
        if rand<=culprob[0]:
            tgt = 0
        else:
            while (right-left)>1:
                middle = int(math.floor((left+right)/2))
                if rand<=culprob[middle]:
                    right = middle
                else:
                    left = middle
            tgt = right

    return tgt

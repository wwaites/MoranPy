import numpy as np

def networkgen(N,E):
    Edgelist = []
    for i in range(0,N):
        for j in range(0,i):
            Edgelist.append([i,j])

    np.random.shuffle(Edgelist)

    network = np.zeros((N, N))
    for i in range(0,E):
        a = Edgelist[i][0]
        b = Edgelist[i][1]
        network[a][b] = 1
        network[b][a] = 1
    return network




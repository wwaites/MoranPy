

def fitness(i, payoff, adj, kinds):
    ## for each neighbour in the adjacency list, add the payoff
    return sum(payoff[kinds[i]][kinds[j]] for j in adj[i])


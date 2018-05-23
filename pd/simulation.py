import sys
import numpy as np
import copy
import argparse

from pd.network import networkgen
from pd.selection import select
import pd.game as game

class Simulation(object):
    def __init__(self, args):
        self.N, self.E = args.N, args.E
        self.t0, self.Tt = args.t0, args.Tt
        self.p, self.q = args.p, args.q
        self.m = args.m
        self.d, self.g = args.d, args.g
        self.v = args.v
        self.b, self.c = args.b, args.c
        self.theta = args.theta

        self.kinds      = [0]*self.N
        self.fitness    = [0.0]*self.N
        self.prosperity = [0.0]*self.N

        ## generate network
        self.network = networkgen(self.N,self.E)
        self.adj = [[] for i in range(self.N)]
        for i in range(self.N):
            for j in range(self.N):
                if self.network[i][j]==1:
                    self.adj[i].append(j)

        ## initialise payoff, etc matrices
        R = self.b-self.c
        S = -self.c
        T = self.b
        P = 0
        self.payoff = [[R,S],[T,P]]
        self.payoffmax = (self.N-1)*self.b
        self.payoffmin = -(self.N-1)*self.c

        for i in range(self.N):
            self.fitness[i] = game.fitness(i, self.payoff, self.adj, self.kinds)
            self.prosperity[i] = pow(1+self.d, self.fitness[i])

    def remove(self):
        """
        Decide on which node to remove
        """
        removallist = []
        for k in range(self.N):
            if self.fitness[k] < self.payoffmin + self.theta*(self.payoffmax-self.payoffmin):
                removallist.append(k)
        if len(removallist)>0:
            tempid = np.random.randint(0, len(removallist))
            i = removallist[tempid]
        else:
            i = np.random.randint(0, self.N)
        return i

    def select(self):
        """
        Decide on which node to copy
        """
        return select(self.prosperity)

    def mutate(self, t, i, j):
        """
        Probabilisically mutate
        """
        if t <= self.t0:
            self.kinds[i] = self.kinds[j]
        else:
            rand = np.random.rand(1)
            if rand < self.m:
                self.kinds[i] = 1 - self.kinds[j]
            else:
                self.kinds[i] = self.kinds[j]

    def reputation(self, i, maxdegree):
        return self.v * pow(1 + self.g, len(self.adj[i]) - maxdegree)

    def simulate(self):
        t = 0
        mutant = False
        transition = False
        transitionStart = False

        transitionNum = 0
        avecoop = 0.0
        avedegree = 0.0
        aveprosp = 0.0

        cooplist = [0.0]*self.Tt
        degreelist = [0.0]*self.Tt
        prosplist = [0.0]*self.Tt

        while t < self.Tt:
            t = t+1
            i = self.remove()
            j = self.select()

            ## connect to role model and neighbours
            maxdegree = max(map(len, self.adj))

            tempneigh = []
            rand = np.random.rand(1)
            if i != j and rand < self.reputation(j, maxdegree) * self.p:
                tempneigh.append(j)
            for k in self.adj[j]:
                rand = np.random.rand(1)
                if i != k and rand < self.reputation(k, maxdegree) * self.q:
                    tempneigh.append(k)

            ## housekeeping
            tempneighi = copy.deepcopy(self.adj[i])
            for k in tempneighi:
                tempid = self.adj[k].index(i)
                del self.adj[k][tempid]
                self.fitness[k] -= self.payoff[self.kinds[k]][self.kinds[i]]
                self.prosperity[k] = pow(1+self.d, self.fitness[k])
                tempid = self.adj[i].index(k)
                del self.adj[i][tempid]
                self.network[i][k] = 0
                self.network[k][i] = 0

            if len(self.adj[i])!=0:
                print("System error!!!")

            self.mutate(t, i, j)

            ## recalculate fitness
            self.fitness[i] = 0.0
            for k in tempneigh:
                self.adj[i].append(k)
                self.adj[k].append(i)
                self.network[i][k] = 1
                self.network[k][i] = 1
                self.fitness[k] += self.payoff[self.kinds[k]][self.kinds[i]]
                self.prosperity[k] = pow(1+self.d, self.fitness[k])
                self.fitness[i] += self.payoff[self.kinds[i]][self.kinds[k]]
                self.prosperity[i] = pow(1+self.d, self.fitness[i])

            ## record data for output
            if sum(self.kinds)!=0 and transitionStart==False and transition== False:
                transitionStart = True

            if sum(self.kinds)==self.N and transitionStart == True and transition == False:
                transitionStart = False
                transition = True
                transitionNum = transitionNum + 1

            if sum(self.kinds)==0 and transitionStart == False and transition == True:
                transitionStart = True
                transition = False
                transitionNum = transitionNum + 1

            avedegree = sum(map(len, self.adj)) / self.N
            aveprosp = sum(self.prosperity) / self.N
            avecoop = self.N - sum(self.kinds)

            cooplist[t-1] = avecoop
            degreelist[t-1] = avedegree
            prosplist[t-1] = aveprosp

            #   print("time="+str(t),'q='+str(q),'theta='+str(theta),avecoop)
            if t % 1000 == 0:
                print("\t".join(map(str, [t, avecoop, avedegree, aveprosp, transitionNum])))

def main():
    parser = argparse.ArgumentParser(prog = "pdsim")
    parser.add_argument("-N", default=100, type=int, help="Number of nodes")
    parser.add_argument("-E", default=400, type=int, help="Number of edges")
    parser.add_argument("--t0", default=10000, type=int, help="Settling time")
    parser.add_argument("--Tt", default=1000000, type=int, help="Simulation end time")
    parser.add_argument("-p", default=0.6, type=float, help="Probability of connecting to role-model")
    parser.add_argument("-q", default=0.85, type=float, help="Probability of connecting to role-model's neighbours")
    parser.add_argument("-m", default=0.0001, type=float, help="Probability of mutation")
    parser.add_argument("-d", default=0.01, type=float, help="Selection strength")
    parser.add_argument("-g", default=0.01, type=float, help="Reliability of public information")
    parser.add_argument("-v", default=0.9, type=float, help="Probability of correctly guessing cooperator")
    parser.add_argument("-b", default=10, type=float, help="Benefit in the public good game")
    parser.add_argument("-c", default=3.333333, type=float, help="Cost in the public good game")
    parser.add_argument("--theta", default=1.0, type=float, help="Parameter for deletion")

    args = parser.parse_args()

    anames = list(args.__dict__.keys())
    anames.sort()
    for n in anames:
        print("# %s = %s" % (n, getattr(args, n)))

    s = Simulation(args)
    s.simulate()

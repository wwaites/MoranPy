import sys
import numpy as np
from math import sqrt, log
import copy
import argparse

from pd.network import networkgen
from pd.selection import select
import pd.game as game

class Simulation(object):
    def __init__(self, args):
        self.N, self.E = args.N, args.E
        self.t0, self.Tt = args.t0, args.Tt
        self.m = args.m
        self.d = args.d
        self.cmean = args.cmean
        self.cdev = sqrt(args.cvar)
        self.dmean = args.dmean
        self.ddev = sqrt(args.dvar)
        self.threshold = args.threshold
        self.b, self.c = args.b, args.c
        self.theta = args.theta

        if args.choice == "private":
            self._should_connect = self.should_connect_private
        elif args.choice == "public":
            self._should_connect = self.should_connect_public
        elif args.choice == "and":
            self._should_connect = lambda i: self.should_connect_private(i) and self.should_connect_public_and(i)
        elif args.choice == "or":
            self._should_connect = lambda i: self.should_connect_private(i) or self.should_connect_public_or(i)
        elif args.choice == "xor":
            self._should_connect = lambda i: self.should_connect_private(i) ^ self.should_connect_public_or(i)
        else:
            raise Exception("Unknown choice algorithm: %s" % args.choice)
        if args.neg:
            self._should_connect_neg = self._should_connect
            self._should_connect = lambda i: not self.should_connect_neg(i)

        self.sample = args.sample

        self.kinds      = np.zeros(self.N, dtype=int)
        self.fitness    = np.zeros(self.N)
        self.prosperity = np.zeros(self.N)

        ## cascade counter for nodes, private information says yes, do not connect
        self.pcascadeids = np.zeros(self.N, dtype=int)
        self.pcascades = { 0: 0 }
        ## cascade counter for nodes, private information says no, connect anyways
        self.ncascadeids = np.zeros(self.N, dtype=int)
        self.ncascades = { 0: 0 }

        self.fp, self.fn, self.tp, self.tn = 0, 0, 0, 0

        ## generate network
        self.network = networkgen(self.N,self.E)
        self.adj = [[] for i in range(self.N)]
        for i in range(self.N):
            for j in range(self.N):
                if self.network[i][j]==1:
                    self.adj[i].append(j)

        ## initialise payoff, etc matrices
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

        self.prospnorm = 100.0 / ( self.N * (self.N-1) * (self.b - self.c))

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

    def should_connect(self, i):
        choice = self._should_connect(i)
        if choice:
            if self.kinds[i] == 0:
                self.tp += 1
            else:
                self.fp += 1
                if not self.private_choice:
                    self.pcascade = True
        else:
            if self.kinds[i] == 0:
                self.fn += 1
                if self.private_choice:
                    self.ncascade = True
            else:
                self.tn += 1
        return choice

    def should_connect_private(self, i):
        if self.kinds[i] == 0:
            rand = np.random.normal(self.cmean, self.cdev)
        else:
            rand = np.random.normal(self.dmean, self.ddev)
        if rand < self.threshold:
            self.private_choice = True
        else:
            self.private_choice = False
        return self.private_choice

    def should_connect_public_and(self, i):
        if len(self.adj[i]) >= self.avedegree:
            self.public_choice = True
        else:
            self.public_choice = False
        return self.public_choice

    def should_connect_public_or(self, i):
        if len(self.adj[i]) > self.avedegree:
            self.public_choice = True
        else:
            self.public_choice = False
        return self.public_choice


    def simulate(self):
        t = 0
        mutant = False
        transition = False
        transitionStart = False

        transitionNum = 0
        self.calculate_graph_statistics()

        #cooplist = [0.0]*self.Tt
        #degreelist = [0.0]*self.Tt
        #prosplist = [0.0]*self.Tt

        print("\t".join(["time", "coop", "degree", "prosp", "trans", "TP", "FP", "TN", "FN", "ncasc", "pcasc", "e0", "e1"]))

        while t < self.Tt:
            t = t+1
            i = self.remove()
            j = self.select()

            ## connect to role model and neighbours
            maxdegree = max(map(len, self.adj))

            ## reset cascade flags
            self.ncascade = self.pcascade = False

            tempneigh = []
            rand = np.random.rand(1)
            if i != j and self.should_connect(j):
                tempneigh.append(j)
            for k in self.adj[j]:
                rand = np.random.rand(1)
                if i != k and self.should_connect(k):
                    tempneigh.append(k)

            ## count cascades
            self.ncascadeids[i] = self.ncascadeids[j]
            self.pcascadeids[i] = self.pcascadeids[j]
            if self.ncascade:
                nid = self.ncascadeids[i]
                if nid == 0:
                    nid = max(self.ncascadeids) + 1
                    self.ncascadeids[i] = nid
                self.ncascades[nid] = self.ncascades.get(nid, 0) + 1
            if self.pcascade:
                pid = self.pcascadeids[i]
                if pid == 0:
                    pid = max(self.pcascadeids) + 1
                    self.pcascadeids[i] = pid
                self.pcascades[pid] = self.pcascades.get(pid, 0) + 1

            ## housekeeping, zero out references to node
            ## to be removed
            tempneighi = copy.deepcopy(self.adj[i])
            for k in tempneighi:
                tempid = self.adj[k].index(i)
                del self.adj[k][tempid]
                self.fitness[k] -= self.payoff[self.kinds[k]][self.kinds[i]]
                self.prosperity[k] = pow(1+self.d, self.fitness[k])
                self.network[i][k] = 0
                self.network[k][i] = 0
            self.adj[i] = []

            ## possibly mutate
            self.mutate(t, i, j)

            ## recalculate fitness
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

            ## record data for output
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

            self.calculate_graph_statistics()

            if t % self.sample == 0:
                nc, pc = max(self.ncascades.values()), max(self.pcascades.values())
                e0, e1 = self.entropy()
                print("\t".join(map(str, [t, self.avecoop, self.avedegree, self.aveprosp, transitionNum, self.tp, self.fp, self.tn, self.fn, nc, pc, e0, e1])))

    def calculate_graph_statistics(self):
        self.avedegree = sum(map(len, self.adj)) / self.N
        self.aveprosp = sum(self.fitness) * self.prospnorm
        self.avecoop = self.N - sum(self.kinds)

    def debug_node(self, n):
        print("debug:")
        print("\tnode: %s, kind(%s)" % (n, self.kinds[n]))
        print("\ttotal payoff: %s/%s" % (self.fitness[n], (self.N-1)*(self.b-self.c)))
        print("\tfitness: %s" % self.prosperity[n])
        print("\tneigh: %s" % self.adj[n])
        print("\tkinds: %s" % (list(self.kinds[i] for i in self.adj[n])))

    def entropy(self):
        defect = sum(self.kinds)
        p = np.array([ len(self.kinds) - defect, defect ], dtype=float) / len(self.kinds)
        q = np.zeros((2,2))
        for i in range(self.N):
            u = self.kinds[i]
            for j in self.adj[i]:
                v = self.kinds[j]
                q[u,v] += 1

        ## make q into a vector of the four kinds of path, and normalise
        q = q.reshape((1,4))[0]
        norm = sum(q)
        if norm != 0:
            q /= norm
        e0 = abs(sum(i * log(i, 2) for i in p if i != 0) / log(0.5, 2))
        e1 = abs(sum(i * log(i, 2) for i in q if i != 0) / log(0.25, 2))
        return (e0, e1)

def main():
    parser = argparse.ArgumentParser(prog = "pdsim")
    parser.add_argument("-N", default=100, type=int, help="Number of nodes")
    parser.add_argument("-E", default=400, type=int, help="Number of edges")
    parser.add_argument("--t0", default=10000, type=int, help="Settling time")
    parser.add_argument("--Tt", default=1000000, type=int, help="Simulation end time")
    parser.add_argument("-m", default=0.0001, type=float, help="Probability of mutation")
    parser.add_argument("-d", default=0.01, type=float, help="Selection strength")
    parser.add_argument("-u", "--cmean", default=-0.5, type=float, help="Mean of cooperator signal distribution")
    parser.add_argument("--cvar", default=0.5, type=float, help="Defector signal distribution variance")
    parser.add_argument("-v", "--dmean", default=0.5, type=float, help="Mean of defector signal distribution")
    parser.add_argument("--dvar", default=0.5, type=float, help="Defector signal distribution variance")
    parser.add_argument("-t", "--threshold", default=None, type=float, help="Discrimination threshold")
    parser.add_argument("-b", default=10, type=float, help="Benefit in the public good game")
    parser.add_argument("-c", default=3.333333, type=float, help="Cost in the public good game")
    parser.add_argument("--theta", default=1.0, type=float, help="Parameter for deletion")
    parser.add_argument("--choice", default="private", help="Choice algorithm")
    parser.add_argument("--neg", default=False, action="store_true", help="Negate choice")
    parser.add_argument("--sample", default=1000, help="sampling frequency")

    args = parser.parse_args()

    if args.threshold is None:
        dm = abs(args.dmean - args.cmean)
        x0 = min(args.dmean, args.cmean)
        x1 = max(args.dmean, args.cmean)
        args.threshold = np.random.uniform(x0-dm, x1+dm)

    anames = list(args.__dict__.keys())
    anames.sort()
    for n in anames:
        print("# %s = %s" % (n, getattr(args, n)))

    s = Simulation(args)
    s.simulate()

import argparse
import numpy as np
from math import sqrt

COOPERATOR=0
DEFECTOR=1

def samples(n, dev=0.1, count=10000):
    """
    Construct an array of random samples, normally distributed
    about n with the given standard deviation.
    """
    return np.random.normal(n, dev, count)

def flips(count=1000):
    """
    Flip a coin 'count' times to select cooperator or defector
    """
    for f in np.random.uniform(0, 1, count):
        if f < 0.5:
            yield COOPERATOR
        else:
            yield DEFECTOR

def main():
    parser = argparse.ArgumentParser("uvroc")
    parser.add_argument("--m1", type=float, default=-1.0, help="Mean of phi_1")
    parser.add_argument("--m2", type=float, default=1.0, help="Mean of phi_2")
    parser.add_argument("--v1", type=float, default=0.5, help="Variance of phi_1")
    parser.add_argument("--v2", type=float, default=1.0, help="Variance of phi_2")
    parser.add_argument("-n", type=int, default=100)
    args = parser.parse_args()

    m1, m2, v1, v2 = args.m1, args.m2, args.v1, args.v2

    ## generate threshold values, including 1/3 left, middle, right of m1, m2
    for c in np.random.uniform(-1*m1-2*m2, 3*m2-m1, args.n):
        tp, tn, fp, fn = 0, 0, 0, 0
        ## set up streams of samples from the distributions
        streams = [samples(m1, sqrt(v1)), samples(m2, sqrt(v2)), flips()]
        ## for each sample, determine true positive, false positive,
        ## true negative, or false negative
        for s1, s2, f in zip(*streams):
            s = s1 if f == COOPERATOR else s2
            if f == COOPERATOR:
                if s <= c:
                    tp += 1
                else:
                    fn += 1
            else:
                if s <= c:
                    fp += 1
                else:
                    tn += 1
        ## print out the result
        sensitivity = float(tp) / (tp + fn) if tp + fn != 0 else 0
        specificity = float(tn) / (tn + fp) if tn + fp != 0 else 0
        print("%f\t%f" % (1.0 - specificity, sensitivity))


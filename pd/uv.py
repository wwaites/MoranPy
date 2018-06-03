import argparse
import numpy as np

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
    parser.add_argument("-v", type=float, default=0.5)
    parser.add_argument("-u", type=float, default=0.5)
    parser.add_argument("-c", type=float, default=10.0)
    parser.add_argument("-n", type=int, default=1000)
    args = parser.parse_args()

    ## generate c values for between zero and args.c
    for c in np.random.uniform(0, args.c, args.n):
        tp, tn, fp, fn = 0, 0, 0, 0
        ## set up streams of random numbers for k_i
        streams = [
            samples(args.u),       samples(1.0 - args.u),
            samples(1.0 - args.v), samples(args.v),
            flips()
        ]
        ## for each sample, determine true positive, false positive,
        ## true negative, or false negative
        for k1, k2, k3, k4, f in zip(*streams):
            s1 = k1 if f == COOPERATOR else k3
            s2 = k2 if f == DEFECTOR else k4
            if f == COOPERATOR:
                if s1 >= c * s2:
                    tp += 1
                else:
                    fn += 1
            else:
                if s1 >= c * s2:
                    fp += 1
                else:
                    tn += 1
        ## print out the result
        sensitivity = float(tp) / (tp + fn) if tp + fn != 0 else 0
        specificity = float(tn) / (tn + fp) if tn + fp != 0 else 0
        print("%f\t%f" % (1.0 - specificity, sensitivity))


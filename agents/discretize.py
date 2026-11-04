import numpy as np

# bucket edges for a direction's total queue -> levels 0..4
BINS = [2, 5, 9, 15]
N_LEVELS = len(BINS) + 1


def discretize(obs):
    qN, qS, qE, qW, phase, yellow = obs
    ns = int(np.digitize(qN + qS, BINS))
    ew = int(np.digitize(qE + qW, BINS))
    return (ns, ew, int(phase), int(yellow > 0))
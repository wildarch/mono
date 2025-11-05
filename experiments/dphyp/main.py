#!/usr/bin/env python3
from itertools import chain, combinations

V = { 1, 2, 3, 4, 5, 6 }
E = {
    ((1,), (2,)),
    ((2,), (3,)),
    ((4,), (5,)),
    ((5,), (6,)),
    ((1, 2, 3), (4, 5, 6)),
    # Mirror
    ((2,), (1,)),
    ((3,), (2,)),
    ((5,), (4,)),
    ((6,), (5,)),
    ((4, 5, 6), (1, 2, 3)),
}

dpTable = {}

def B(v):
    return set(v_ for v_ in V if v_ <= v)


def Solve():
    for v in V:
        dpTable[(v,)] = f"R{v}"
    for v in sorted(V, reverse=True):
        EmitCsg({v})
        EnumerateCsgRec({v}, B(v))
    
def powerset(iterable):
    # powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def find_neighborhood(S, X):
    interest = set()
    for (u,v) in E:
        if set(u) <= S and S.isdisjoint(set(v)) and X.isdisjoint(set(v)):
            interest.add(v)
    
    # TODO: subsumed hypernodes

    neighborhood = set()
    for v in interest:
        neighborhood.add(min(list(v)))
    
    print(f"find_neighborhood({S}, {X}) = {neighborhood}")
    return neighborhood

def EnumerateCsgRec(S1, X):
    print("EnumerateCsgRec", S1, X)
    N = find_neighborhood(S1, X)
    for n in powerset(N):
        if not n:
            continue
        if tuple(S1 | set(n)) not in dpTable:
            EmitCsg(S1 | set(n))

    for n in powerset(N):
        if not n:
            continue
        EnumerateCsgRec(S1 | set(n), X | N)

def EmitCsg(S1):
    print("EmitCsg", S1)
    X = S1 | B(min(S1))
    N = find_neighborhood(S1, X)
    for v in sorted(N, reverse=True):
        S2 = {v}
        for (u, v) in E:
            if set(u) <= S1 and set(v) <= S2:
                EmitCsgCmp(S1, S2)
                break
        EnumerateCmpRec(S1, S2, X)

def EnumerateCmpRec(S1, S2, X):
    print("EnumerateCmpRec", S1, S2, X)
    N = find_neighborhood(S2, X)
    for n in powerset(N):
        if not n:
            continue
        if tuple(S2 | set(n)) in dpTable:
            for (u, v) in E:
                if set(u) <= S1 and set(v) <= S2:
                    EmitCsgCmp(S1, S2 | set(n))
                    break
    X = X | N
    N = find_neighborhood(S2, X)
    for n in powerset(N):
        if not n:
            continue
        EnumerateCmpRec(S1, S2 | set(n), X)

def EmitCsgCmp(S1, S2):
    print("EmitCsgCmp", S1, S2)
    plan1 = dpTable[tuple(S1)]
    plan2 = dpTable[tuple(S2)]
    S = S1 | S2
    newplan = f"join({plan1}, {plan2})"
    # Ignoring costing for now
    dpTable[tuple(S)] = newplan
    print(f"New plan for {S}: {newplan}")

Solve()
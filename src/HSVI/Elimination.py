import numpy as np
from gurobipy import *

"""
eliminates unnecessary UB and LB
"""


def filter(LB,nS):
    W = {}
    F = LB.copy()
    popidx = []
    for i in range(nS):
        if len(F) == 0:
            break
        b = np.zeros(nS)
        b[i] = 1.0
        maxval = -np.inf
        maxidx = None
        for ii,lb in F.items():
            val = b @ lb
            if val > maxval:
                maxval = val
                maxidx = ii
        W[maxidx] = F[maxidx]
        del F[maxidx]
    while len(F)>0:
        phi = list(F.keys())[0]
        b = dominate(phi,F[phi],W,nS)
        if b is None:
            del F[phi]
            popidx.append(phi)
        else:
            maxval = -np.inf
            maxidx = None
            for ii,lb in F.items():
                val = b @ lb
                if val > maxval:
                    maxval = val
                    maxidx = ii
            W[maxidx] = F[maxidx]
            del F[maxidx]
    return popidx

def dominate(idx,alpha,A,nS):
    mdl = Model()
    b = mdl.addVars(nS,name='b')
    d = mdl.addVar(lb=-GRB.INFINITY,name='d')
    mdl.addConstr(quicksum(b[i] for i in range(nS)) == 1,name='simplex')
    for ii,lb in A.items():
        if ii != idx:
            expr = LinExpr(-d)
            for i in range(nS):
                coeff = alpha[i]-lb[i]
                if abs(coeff) > 1e-12:
                    expr.add(coeff*b[i])
            mdl.addConstr(expr,GRB.GREATER_EQUAL,0,name='sup_'+str(ii))
    mdl.setObjective(d,GRB.MAXIMIZE)
    mdl.setParam(GRB.Param.OutputFlag,0)
    #mdl.setParam(GRB.Param.LogToConsole,0)
    mdl.update()
    mdl.optimize()
    if mdl.status == GRB.status.OPTIMAL:
        if d.X > 0:
            bval = np.zeros(nS)
            for i in range(nS):
                bval[i] = b[i].X
            return bval
        else:
            return None
    else:
        return None

def delPoints(UB,nS):
    popidx = []
    points,idxs = UB.getPoints()
    for ii in range(nS,len(idxs)):
        b = points[ii,0:nS]
        v = points[ii,-1]
        ub = UB.getV(b)
        if v > ub + 1e-9:
            popidx.append(idxs[ii])
        else:
            pass
            #deal with very close points
            for jj in range(ii+1,len(idxs)):
                diff = np.sum(np.abs(points[ii]-points[jj]))
                if diff < 1e-20:
                    popidx.append(idxs[ii])
                    break
    return popidx

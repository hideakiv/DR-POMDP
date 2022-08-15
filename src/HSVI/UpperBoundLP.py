import numpy as np
from gurobipy import *

"""
Use LP to obtain upperbound
"""

class UpperBoundLP:
    def __init__(self,nS,UB=None,filename=None,count=None):
        if UB is not None:
            self.mdl = Model()
            self.ub1 = self.mdl.addVars(nS,lb=-GRB.INFINITY,name='UB1')
            self.ub2 = self.mdl.addVar(lb=-GRB.INFINITY,name='UB2')
            self.count = 0
            self.UBconstr = {}
            for ii in range(len(UB)):
                self.addPoint(UB[ii][0],UB[ii][1])
            self.mdl.setObjective(0,GRB.MAXIMIZE)
            self.ub2.setAttr(GRB.Attr.Obj,1.0)
            self.mdl.setParam(GRB.Param.OutputFlag,0)
            self.mdl.update()
        else:
            assert count is not None
            self.mdl = read(filename)
            self.mdl.setParam(GRB.Param.OutputFlag,0)
            self.ub1 = {}
            for i in range(nS):
                self.ub1[i] = self.mdl.getVarByName('UB1['+str(i)+']')
            self.ub2 = self.mdl.getVarByName('UB2')
            self.count = count
            self.UBconstr = {}
            constrs = self.mdl.getConstrs()
            for row_idx, constr in enumerate(constrs):
                name = constr.ConstrName
                if name[0:len('point')] == 'point':
                    idx = int(name[len('point'):])
                    self.UBconstr[idx] = constr



    def addPoint(self,b,v):
        nS = len(b)
        self.UBconstr[self.count]=self.mdl.addConstr(quicksum(b[i]*self.ub1[i] for i in range(nS))+self.ub2 <= v,name='point'+str(self.count))
        #expr = LinExpr(self.ub2)
        #expr.addTerms(b,self.ub1)
        #self.mdl.addConstr(expr,GRB.LESS_EQUAL,v,name='point'+str(self.count))
        self.count += 1
        self.mdl.update()

    def getV(self,b):
        nS = len(b)
        for i in range(nS):
            self.ub1[i].setAttr(GRB.Attr.Obj,b[i])
        self.mdl.update()
        #self.mdl.setParam(GRB.Param.DualReductions,0)
        self.mdl.optimize()
        if self.mdl.status == 3:
            print('too many points close together',self.mdl.status)
            #numeric case: return close and smallest point
            Points, idxs = self.getPoints()
            retval = np.inf
            for i in range(len(Points)):
                if sum(abs(Points[i,0:nS]-b))<1e-9 and Points[i,-1]<retval:
                    retval = Points[i,-1]
            return retval

        elif self.mdl.status != GRB.status.OPTIMAL:
            #print(self.mdl.status)
            self.mdl.write('./UB.lp')
        return self.mdl.objVal

    def getPoints(self):
        numPoints = self.mdl.NumConstrs
        dimBelief = self.mdl.NumVars - 1
        Points = np.zeros((numPoints,dimBelief+1))
        idxs = np.zeros(numPoints,dtype=int)

        dvars = self.mdl.getVars()
        constrs = self.mdl.getConstrs()
        var_indices = {v: i for i, v in enumerate(dvars)}

        for row_idx, constr in enumerate(constrs):
            for coeff, col_idx in self.get_expr_coos(self.mdl.getRow(constr), var_indices):
                Points[row_idx, col_idx] = coeff
            Points[row_idx,-1] = constr.RHS
            idxs[row_idx] = int(constr.ConstrName[len('point'):])
        return Points, idxs

    def get_expr_coos(self, expr, var_indices):
        for i in range(expr.size()):
            dvar = expr.getVar(i)
            yield expr.getCoeff(i), var_indices[dvar]

    def getNum(self):
        return self.mdl.NumConstrs

    def popConstr(self,idx):
        self.mdl.remove(self.UBconstr[idx])
        del self.UBconstr[idx]

    def save(self,filename):
        self.mdl.write(filename)

import numpy as np
from gurobipy import *

"""
Provide initial value for UB 
"""

class MomentLPUBinit:
    def __init__(self,prm):
        self.mdl = Model()
        self.v = []
        self.kap_p = []
        self.kap_m = []
        self.yEq = []
        self.defineVars(prm)
        self.defineConstrs(prm)
        self.mdl.setObjective(0,GRB.MINIMIZE)
        for i in range(prm.nS):
            self.v[i].setAttr(GRB.Attr.Obj,1.0)
        self.mdl.setParam(GRB.Param.OutputFlag,0)
        self.mdl.update()
        #self.mdl.setParam(GRB.Param.LogToConsole,0)


    def defineVars(self,prm):
        self.v   = self.mdl.addVars(prm.nS,lb=-GRB.INFINITY,name='v')
        self.kap_p = self.mdl.addVars(prm.nA,prm.nS,prm.nS*prm.nZ,name='kp')
        self.kap_m = self.mdl.addVars(prm.nA,prm.nS,prm.nS*prm.nZ,name='km')
        tup = tuplelist([])
        for a in range(prm.nA):
            for i in range(prm.nS):
                for l in range(len(prm.Aeq[a][i])):
                    tup += [(a,i,l)]
        self.yEq = self.mdl.addVars(tup,lb=-GRB.INFINITY,name='yEq')

    def defineConstrs(self,prm):
        for a in range(prm.nA):
            for i in range(prm.nS):
                for j in range(prm.nS):
                    for z in range(prm.nZ):
                        jz = j*prm.nZ+z
                        expr = LinExpr()
                        expr.add(-self.kap_p[a,i,jz])
                        expr.add( self.kap_m[a,i,jz])
                        expr.addTerms(-prm.Aeq[a][i][:,j*prm.nZ+z],[self.yEq[a,i,l] for l in range(len(prm.Aeq[a][i]))])
                        expr.add(self.v[j],prm.beta)
                        self.mdl.addConstr(expr,GRB.LESS_EQUAL,-prm.R[a,i,j,z],name='P('+str(a)+','+str(i)+','+str(j)+','+str(z)+')')

                expr = LinExpr()
                expr.add(self.v[i])
                expr.addTerms(-prm.cap[a,i,:]-prm.pbar[a,i,:],[self.kap_p[a,i,jz] for jz in range(prm.nS*prm.nZ)])
                expr.addTerms(-prm.cap[a,i,:]+prm.pbar[a,i,:],[self.kap_m[a,i,jz] for jz in range(prm.nS*prm.nZ)])
                expr.addTerms(-prm.beq[a][i],[self.yEq[a,i,l] for l in range(len(prm.Aeq[a][i]))])
                self.mdl.addConstr(expr,GRB.GREATER_EQUAL,0,name='vlb('+str(a)+','+str(i)+')')

    def solve(self,prm):
        self.mdl.setParam(GRB.Param.DualReductions,0)
        self.mdl.optimize()
        if self.mdl.status != GRB.status.OPTIMAL:
            print(self.mdl.status)
            self.mdl.write('./UBinit.lp')
        V = np.zeros(prm.nS)
        for i in range(prm.nS):
            V[i] = self.v[i].getAttr(GRB.Attr.X)
        return V

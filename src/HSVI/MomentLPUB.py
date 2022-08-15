import numpy as np
from gurobipy import *

"""
Upper bound for POMDP with moment-based ambiguity set
"""

class MomentLPUB:
    def __init__(self,a,prm,filename=None,count=None):
        if filename is None:
            self.mdl = Model()
            self.kap_p = []
            self.kap_m = []
            self.yEq = []
            self.ub1 = []
            self.ub2 = []
            self.defineVars(a,prm)
            self.UBconstrs = {}
            self.defineConstrs(a,prm)
            self.defineObj(a,prm)
            self.mdl.setParam(GRB.Param.OutputFlag,0)
            self.mdl.update()
            self.count = 0
            #self.mdl.setParam(GRB.Param.LogToConsole,0)
        else:
            assert count is not None
            self.mdl = read(filename)
            self.mdl.setParam(GRB.Param.OutputFlag,0)
            self.kap_p = {}
            self.kap_m = {}
            self.yEq = {}
            self.ub1 = {}
            self.ub2 = {}
            for z in range(prm.nZ):
                for i in range(prm.nS):
                    self.ub1[z,i] = self.mdl.getVarByName('UB1['+str(z)+','+str(i)+']')
                self.ub2[z] = self.mdl.getVarByName('UB2['+str(z)+']')
            for i in range(prm.nS):
                for jz in range(prm.nS*prm.nZ):
                    self.kap_p[i,jz] = self.mdl.getVarByName('kp['+str(i)+','+str(jz)+']')
                    self.kap_m[i,jz] = self.mdl.getVarByName('km['+str(i)+','+str(jz)+']')
            tup = tuplelist([])
            for i in range(prm.nS):
                for l in range(len(prm.Aeq[a][i])):
                    tup += [(i,l)]
            for tt in tup:
                self.yEq[tt] = self.mdl.getVarByName('yEq['+str(tt[0])+','+str(tt[1])+']')
            self.count = count
            self.UBconstrs = {}
            constrs = self.mdl.getConstrs()
            for row_idx, constr in enumerate(constrs):
                name = constr.ConstrName
                if name[0:len('point')] == 'point':
                    name = name[len('point'):]
                    nums = name.split('_')
                    idx = int(nums[0])
                    z   = int(nums[1])
                    self.UBconstrs[idx,z] = constr


    def defineVars(self,a,prm):
        self.kap_p = self.mdl.addVars(prm.nS,prm.nS*prm.nZ,name='kp')
        self.kap_m = self.mdl.addVars(prm.nS,prm.nS*prm.nZ,name='km')
        tup = tuplelist([])
        for i in range(prm.nS):
            for l in range(len(prm.Aeq[a][i])):
                tup += [(i,l)]
        self.yEq = self.mdl.addVars(tup,lb=-GRB.INFINITY,name='yEq')
        self.ub1 = self.mdl.addVars(prm.nZ,prm.nS,lb=-GRB.INFINITY,name='UB1')
        self.ub2 = self.mdl.addVars(prm.nZ,lb=-GRB.INFINITY,name='UB2')

    def defineConstrs(self,a,prm):
        for i in range(prm.nS):
            for j in range(prm.nS):
                for z in range(prm.nZ):
                    jz = j*prm.nZ+z
                    expr = LinExpr()
                    expr.add(-self.kap_p[i,jz])
                    expr.add( self.kap_m[i,jz])
                    expr.addTerms(-prm.Aeq[a][i][:,j*prm.nZ+z],[self.yEq[i,l] for l in range(len(prm.Aeq[a][i]))])
                    expr.add(self.ub1[z,j])
                    expr.add(self.ub2[z])
                    self.mdl.addConstr(expr,GRB.LESS_EQUAL,0,name='P('+str(i)+','+str(j)+','+str(z)+')')

    def defineObj(self,a,prm):
        expr = LinExpr()
        for i in range(prm.nS):
            expr.addTerms(-prm.cap[a,i,:]-prm.pbar[a,i,:],[self.kap_p[i,jz] for jz in range(prm.nS*prm.nZ)])
            expr.addTerms(-prm.cap[a,i,:]+prm.pbar[a,i,:],[self.kap_m[i,jz] for jz in range(prm.nS*prm.nZ)])
            expr.addTerms(-prm.beq[a][i],[self.yEq[i,l] for l in range(len(prm.Aeq[a][i]))])
        self.mdl.setObjective(expr,GRB.MAXIMIZE)

    def addPoint(self,b,v,prm):
        for z in range(prm.nZ):
            self.UBconstrs[self.count,z]=self.mdl.addConstr(quicksum(b[i]*self.ub1[z,i] for i in range(prm.nS))+self.ub2[z] <= v,name='point'+str(self.count)+'_'+str(z))
        self.count += 1
        self.mdl.update()

    def popConstr(self,idx,nZ):
        for z in range(nZ):
            self.mdl.remove(self.UBconstrs[idx,z])
            del self.UBconstrs[idx,z]

    def solve(self,a,b,prm):
        for i in range(prm.nS):
            for j in range(prm.nS):
                for z in range(prm.nZ):
                    c = self.mdl.getConstrByName('P('+str(i)+','+str(j)+','+str(z)+')')
                    self.mdl.chgCoeff(c,self.ub1[z,j],-prm.beta*b[i])
                    self.mdl.chgCoeff(c,self.ub2[z],-prm.beta*b[i])
                    c.RHS = b[i] * prm.R[a,i,j,z]
        #self.mdl.setParam(GRB.Param.DualReductions,0)
        self.mdl.update()
        self.mdl.optimize()
        if self.mdl.status == 4:
            #sometimes this works
            self.mdl.Params.DualReductions=0
            self.mdl.optimize()
            self.mdl.Params.DualReductions=1
        if self.mdl.status != GRB.status.OPTIMAL:
            print(self.mdl.status)
            self.mdl.write('./momentUB.lp')
        return self.mdl.objVal

    def save(self,filename):
        self.mdl.write(filename)

import numpy as np
from gurobipy import *

"""
Lower bound for POMDP with moment-based ambiguity set
"""

class MomentLPLB:

    def __init__(self,a,prm,filename=None,count=None):
        if filename is None:
            self.mdl = Model()
            self.p = self.mdl.addVars(prm.nS,prm.nS*prm.nZ,name='p')
            self.r = self.mdl.addVars(prm.nS,lb=-GRB.INFINITY,name='r')
            self.s = self.mdl.addVars(prm.nZ,lb=-GRB.INFINITY,name='s')
            self.defineConstrs(a,prm)
            self.mdl.update()

            self.mdl.setObjective(0,GRB.MINIMIZE)
            self.mdl.setParam(GRB.Param.OutputFlag,0)
            #self.mdl.setParam(GRB.Param.LogToConsole,0)
            self.mdl.update()

            self.count = 0
            self.LBconstrs = {}
        else:
            assert count is not None
            self.mdl = read(filename)
            self.mdl.setParam(GRB.Param.OutputFlag,0)
            self.p = {}
            self.r = {}
            self.s = {}
            for i in range(prm.nS):
                for jz in range(prm.nS*prm.nZ):
                    self.p[i,jz] = self.mdl.getVarByName('p['+str(i)+','+str(jz)+']')
                self.r[i] = self.mdl.getVarByName('r['+str(i)+']')
            for z in range(prm.nZ):
                self.s[z] = self.mdl.getVarByName('s['+str(z)+']')
            self.count = count
            self.LBconstrs = {}
            constrs = self.mdl.getConstrs()
            for row_idx, constr in enumerate(constrs):
                name = constr.ConstrName
                if name[0:len('LB')] == 'LB':
                    name = name[len('LB('):-1]
                    nums = name.split(',')
                    idx = int(nums[0])
                    z   = int(nums[1])
                    self.LBconstrs[idx,z] = constr



    def defineConstrs(self,a,prm):
        for i in range(prm.nS):
            for j in range(prm.nS):
                for z in range(prm.nZ):
                    jz = j*prm.nZ+z
                    expr1 = LinExpr()
                    expr2 = LinExpr()
                    expr1.add(self.p[i,jz])
                    expr2.add(self.p[i,jz],-1)
                    self.mdl.addConstr(expr1,GRB.LESS_EQUAL, prm.pbar[a,i,jz]+prm.cap[a,i,jz],name='u('+str(i)+','+str(j)+','+str(z)+')')
                    self.mdl.addConstr(expr2,GRB.LESS_EQUAL,-prm.pbar[a,i,jz]+prm.cap[a,i,jz],name='l('+str(i)+','+str(j)+','+str(z)+')')
            
            self.mdl.addConstr(quicksum(prm.R[a,i,j,z]*self.p[i,j*prm.nZ+z] for j in range(prm.nS) for z in range(prm.nZ))-self.r[i] == 0,name='r('+str(i)+')')
            for l in range(len(prm.beq[a][i])):
                self.mdl.addConstr(quicksum(prm.Aeq[a][i][l,jz]*self.p[i,jz] for jz in range(prm.nS*prm.nZ)) == prm.beq[a][i][l],name='eq('+str(i)+','+str(l)+')')

    def initializeLB(self,prm):
        assert len(self.LBconstrs) == 0
        for i in range(prm.nS):
            self.r[i].setAttr(GRB.Attr.Obj,1.0)
        self.mdl.setParam(GRB.Param.DualReductions,0)
        self.mdl.update()
        self.mdl.optimize()
        if self.mdl.status != GRB.status.OPTIMAL:
            print(self.mdl.status)
            self.mdl.computeIIS()
            self.mdl.write('./LB.ilp')
        ra = np.zeros(prm.nS)
        for i in range(prm.nS):
            ra[i] = self.r[i].getAttr(GRB.Attr.X)

        for z in range(prm.nZ):
            self.s[z].setAttr(GRB.Attr.Obj,1.0)
        return ra

    def addgrad(self,prm):
        for z in range(prm.nZ):
            self.LBconstrs[self.count,z] = self.mdl.addConstr(quicksum(self.p[i,j*prm.nZ+z] for i in range(prm.nS) for j in range(prm.nS)) + self.s[z] >= 0,name='LB('+str(self.count)+','+str(z)+')')
        self.count += 1
        self.mdl.update()

    def popConstr(self,idx,nZ):
        for z in range(nZ):
            self.mdl.remove(self.LBconstrs[idx,z])
            del self.LBconstrs[idx,z]

    def backup(self,b,prm,retry=None):
        for i in range(prm.nS):
            self.r[i].setAttr(GRB.Attr.Obj,b[i])
        for ii,lb in prm.LB.items():
            for z in range(prm.nZ):
                c = self.LBconstrs[ii,z]
                for i in range(prm.nS):
                    for j in range(prm.nS):
                        self.mdl.chgCoeff(c,self.p[i,j*prm.nZ+z],-prm.beta*b[i]*lb[j])
        #self.mdl.setParam(GRB.Param.DualReductions,0)
        self.mdl.update()
        self.mdl.optimize()

        newalph = np.zeros(prm.nS)
        oldalph = np.zeros((prm.nZ,prm.nS))
        for ii,lb in prm.LB.items():
            for z in range(prm.nZ):
                oldalph[z,:] += np.abs(self.LBconstrs[ii,z].getAttr(GRB.Attr.Pi))*lb

        for i in range(prm.nS):
            Pas = np.zeros((prm.nS,prm.nZ))
            ras = self.r[i].getAttr(GRB.Attr.X)
            tot = 0
            for j in range(prm.nS):
                for z in range(prm.nZ):
                    Pas[j,z] = self.p[i,j*prm.nZ+z].getAttr(GRB.Attr.X)
                    tot += Pas[j,z]
            Pas = Pas / tot
            foo = ras
            for z in range(prm.nZ):
                foo += prm.beta * np.dot(oldalph[z,:],Pas[:,z])
            newalph[i] = foo
        return newalph

    def getdist(self,b,prm):
        for i in range(prm.nS):
            self.r[i].setAttr(GRB.Attr.Obj,b[i])
        for ii,lb in prm.LB.items():
            for z in range(prm.nZ):
                c = self.LBconstrs[ii,z]
                for i in range(prm.nS):
                    for j in range(prm.nS):
                        self.mdl.chgCoeff(c,self.p[i,j*prm.nZ+z],-prm.beta*b[i]*lb[j])
        self.mdl.update()
        self.mdl.optimize()

        Pa = np.zeros((prm.nS,prm.nS,prm.nZ))
        for i in range(prm.nS):
            Pas = np.zeros((prm.nS,prm.nZ))
            tot = 0
            for j in range(prm.nS):
                for z in range(prm.nZ):
                    Pas[j,z] = self.p[i,j*prm.nZ+z].getAttr(GRB.Attr.X)
                    tot += Pas[j,z]
            Pas = Pas / tot
            Pa[i,:,:] = Pas
        return Pa

    def getVal(self,b,prm):
        for i in range(prm.nS):
            self.r[i].setAttr(GRB.Attr.Obj,b[i])
        for ii,lb in prm.LB.items():
            for z in range(prm.nZ):
                c = self.LBconstrs[ii,z]
                for i in range(prm.nS):
                    for j in range(prm.nS):
                        self.mdl.chgCoeff(c,self.p[i,j*prm.nZ+z],-prm.beta*b[i]*lb[j])
        #self.mdl.setParam(GRB.Param.DualReductions,0)
        self.mdl.update()
        self.mdl.optimize()
        return self.mdl.objVal

    def save(self,filename):
        self.mdl.write(filename)

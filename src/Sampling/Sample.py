import numpy as np
from . import Support

"""
interfaces for sampling over probability simplex
"""

def sampleUniformOnSimplex(dim,center,radius,nsample=1):
    q = Support.ProbSimplex(dim)

    n1,n2 = q.shape
    points = np.zeros((nsample,n2))
    for i in range(nsample):
        while True:
            #sample from unit sphere (L2)
            while True:
                z = np.random.rand(n1)
                if np.linalg.norm(z) <= 1:
                    break

            y = 2 * z - np.ones(n1)

            point = center + y @ q * radius

            #check if inside bound
            if np.min(point)>=0:
                points[i,:] = point
                break
    return points

def sampleNormalOnSimplex(dim,center,std,nsample=1):
    q = Support.ProbSimplex(dim)

    n1,n2 = q.shape
    points = np.zeros((nsample,n2))
    for i in range(nsample):
        while True:
            #sample from normal distribution
            z = np.random.randn(n1)

            point = center + z @ q * std

            #check if inside bound
            if np.min(point)>=0:
                points[i,:] = point
                break
    return points

def sampleP(center,radius,q):
    if len(q)>0:
        n1,n2 = q.shape
        t = 0
        while True:
            while True:
                z = np.random.rand(n1)
                if np.linalg.norm(z) <= 1:
                    break
            val = center + (2*z-np.ones((1,n1))) @ q * radius
            if np.min(val)>=0:
                break
            t += 1
        return val[0]
    else:
        return center



if __name__ == '__main__':
    dim = 3
    c = np.array([0,0.5,0.5])
    r = 0.1
    n = 5
    points=sampleNormalOnSimplex(dim,c,r,n)
    for p in points:
        print(p,sum(p))

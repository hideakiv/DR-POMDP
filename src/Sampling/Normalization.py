import numpy as np

'''
calculates the orthonormal base vector of a hyperplane Ax=b
'''

def GaussElim(A,b):
    if np.ndim(b)==1:
        b = np.array([b])
    B = np.hstack((A,b))
    B = B.astype(float)
    m,n = B.shape
    h = 0
    k = 0
    while h < m and k < n:
        #Find the k-th pivot
        maxval = 0
        imax = -1
        for i in range(h,m):
            temp = abs(B[i,k])
            if maxval < temp:
                maxval = temp
                imax = i
        if B[imax,k] == 0:
            #No pivot in this column, pass to next column
            k += 1
        else:
            B[[h,imax]] = B[[imax,h]]
            #Do for all rows below pivot:
            for i in range(h+1,m):
                f = B[i,k] / B[h,k]
                #Fill with zeros the lower part of pivot column:
                B[i,k] = 0
                #Do for all remaining elements in current row:
                for j in range(k+1,n):
                    B[i,j] = B[i,j] - B[h,j] * f
                #Increase pivot row and column */
            h += 1
            k += 1
    #row echelon form complete
    for i in range(m-1,-1,-1):
        ech = -1
        if B[i,i] == 0:
            for k in range(i+1,n):
                if abs(B[i,k])>0:
                    ech = k
                    break
            if ech == n-1:
                print('infeasible')
                return None
            #elif ech != -1:
            #    B[:,[i,ech]] = B[:,[ech,i]]
        else:
            ech = i
        if ech == -1:
            continue
        else:
            f = B[i,ech]
            B[i,ech] = 1
            for k in range(ech+1,n):
                B[i,k] = B[i,k]/f
            for h in range(i-1,-1,-1):
                f = B[h,ech]
                B[h,ech] = 0
                for k in range(ech+1,n):
                    B[h,k] = B[h,k] - B[i,k] * f
    term = np.zeros((n-1,1))
    #pdb.set_trace()
    for h in range(m):
        term[h,0] = B[h,-1]
    #vec = [term]
    vec = []
    for i in range(n-1):
        if i >= m or B[i,i] == 0:
            foo = np.zeros((n-1,1))
            for h in range(min(i,m)):
                foo[h,0] = -B[h,i]
            foo[i,0] = 1.0
            vec.append(foo)
    return vec,term

def GramSchmidt(base):
    if len(base)>0:
        C = np.zeros((len(base),len(base[0])))
        #pdb.set_trace()
        for i in range(len(base)):
            C[i] = base[i].T[0]
        q,r = np.linalg.qr(C.T)
        return q.T
    else:
        return []

def getbase(A,b):
    m,n = A.shape
    if m>n:
        return []
    else:
        base,feas = GaussElim(A,b)
        q = GramSchmidt(base)
        return q

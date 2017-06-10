import numpy as np


def reformatdat(datmat):
    #
    # Reformats dataset form a 2-D array of elements, each a 1-D vector, into  
    # two arrays:
    # datx are La \times Lb rows, each row has two columns, which are coordinates in the matrix .
    # daty are the corresponding La \times Lb entries. 
    #
    tmpdim=datmat.shape;
    
    La = tmpdim[0]
    Lb = tmpdim[1]
    if datmat.ndim > 2:
        Ly = tmpdim[2]
    else:
        Ly = 1

    daty = np.zeros([La*Lb,Ly])
    datx = np.zeros([La*Lb,2])
    for j1 in range(La):
        for j2 in range(Lb):
            pass
            datx[j1*Lb+j2,0] = j1 
            datx[j1*Lb+j2,1] = j2
            daty[j1*Lb+j2,:] = datmat[j1,j2,:]
           
    return datx, daty

def reformatdat_rc(datmat):
    #
    # Reformats dataset form a 2-D array of elements, each a 1-D vector, into
    # two arrays:
    # datx are La \times Lb rows, each row has two columns, which are coordinates in the matrix .
    # daty are the corresponding La \times Lb entries.
    #
    tmpdim=datmat.shape;
    
    La = tmpdim[0]
    Lb = tmpdim[1]
    if datmat.ndim > 2:
        Ly = tmpdim[2]
    else:
        Ly = 1
    print Ly

    daty = np.zeros([La*Lb,Ly])
    datr = np.zeros([La*Lb,Lb*Ly])
    datc = np.zeros([La*Lb,La*Ly])
    print datmat.shape

    for j1 in range(La):
        for j2 in range(Lb):
            pass
            daty[j1*Lb+j2,:] = datmat[j1,j2,:]
            #print np.reshape(datmat[j1,:,0],(Lb,1)).shape
            #print np.reshape(datmat[:,j2,0],(La,1)).shape
            datr[j1*Lb+j2,:] = np.reshape(datmat[j1,:,:],(1,Lb*Ly))
            datc[j1*Lb+j2,:] = np.reshape(datmat[:,j2,:],(1,La*Ly))


    return datr, datc, daty





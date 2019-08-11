import numpy as np
from numpy.linalg import inv

a = np.array([ [1,   2,  15,  4],[ 2,  6,  7,  8], [9,  10,  11,  12],[ 13,  14,  15,  16 ]])

ainv = inv(a)



print ("original matrix")
print (a)

print ("inverted matrix")
print (ainv)



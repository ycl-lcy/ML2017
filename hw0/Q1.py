import sys
import numpy as np

mA = np.loadtxt(sys.argv[1], delimiter=',', ndmin=2)
mB = np.loadtxt(sys.argv[2], delimiter=',', ndmin=2)
a = np.matmul(mA, mB)
b = np.sort(a, axis=None)

f = open("ans_one.txt", "w")
for i, val in enumerate(b):
    f.write(str(int(val)) + "\n")

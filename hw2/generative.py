import pandas as pd
import numpy as np
import math
import sys

features = 106

#def gau(xx, m, c):
#    np.exp((-1/2)*(xx - m)numpy.linalg.inv)

x = pd.read_csv(sys.argv[1])
y = pd.read_csv(sys.argv[2], header=None)
test = pd.read_csv(sys.argv[3])

#for i in (0.5,1,2,3):
#    for col in x.ix[:,'age':'hours_per_week']:
#       x[col+"**"+str(i)] = x[col]**i
#       test[col+"**"+str(i)] = test[col]**i
#x.columns = range(len(x.columns))

x = x.as_matrix()
y = y.as_matrix()
test = test.as_matrix()
c1 = []
c2 = []
for i in range(len(x)):
    if y[i] == 1:
        c1.append(x[i])
    else:
        c2.append(x[i])

c1 = np.asarray(c1)
c2 = np.asarray(c2)
c1_mean = c1.mean(axis=0)
c2_mean = c2.mean(axis=0)
c1_std = c1.std(axis=0)
c2_std = c2.std(axis=0)
#c1 = (c1 - c1_mean)/(c1_std+10e-5) #???
#c2 = (c2 - c2_mean)/(c2_std+10e-5) #???
#c1 = np.linalg.norm(c1, axis=0)
#c2 = np.linalg.norm(c2,axis=0)
c1_cov = np.cov(c1.transpose())
c2_cov = np.cov(c2.transpose())
p1 = len(c1)/((len(c1)+len(c2))*1.0)
p2 = len(c2)/((len(c1)+len(c2))*1.0)
cov = (p1*c1_cov)+(p2*c2_cov)

#print (test - c1_mean).shape
#x = x.sub(t_mean).div(t_std)
#test = test.sub(t_mean).div(t_std)

#print cov

print 'id,label'
for i in range(0,len(test)): #16281
    g1 = np.exp((-1/2)*reduce(np.dot,[(test[i] - c1_mean), np.linalg.inv(cov), (test[i] - c1_mean).T]))*(1/((2*math.pi)**(features/2)))*(1/((np.linalg.det(cov))**(1/2)))
    g2 = np.exp((-1/2)*reduce(np.dot,[(test[i] - c2_mean), np.linalg.inv(cov), (test[i] - c2_mean).T]))*(1/((2*math.pi)**(features/2)))*(1/((np.linalg.det(cov))**(1/2)))
    if g1*p1+g2*p2 == 0:
        z = 1
    else:
        z = g1*p1/(g1*p1+g2*p2)
    #print z
    if z >= 0.5:
        #if y[0][i] == 1:
        #    zz += 1
        print '{},{}'.format(i+1, 1)
    else:
        #if y[0][i] == 0:
        #   zz += 1
        print '{},{}'.format(i+1, 0)

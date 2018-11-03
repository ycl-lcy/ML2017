import pandas as pd
import math
import sys

features = 130
b1 = 0.9
b2 = 0.999
eps = 10e-8
mt = 0
vt = 0

#bias = 0
#prev_grad_b = 0
args = pd.Series(0, index=range(features+1))
args = args.astype(float)
#prev_grad_a = pd.Series(0, index=range(features))

x = pd.read_csv(sys.argv[1])
y = pd.read_csv(sys.argv[2], header=None)
test = pd.read_csv(sys.argv[3])
# for i in range(2,7):
    # for j in range(0,6):
        # x2 = x.ix[:,'b':'c']**2
    #x = pd.concat([x,x2], axis=1)

for i in (0.5,1,2,3):
    for col in x.ix[:,'age':'hours_per_week']:
        x[col+"**"+str(i)] = x[col]**i
        test[col+"**"+str(i)] = test[col]**i
t_mean = x.mean(axis=0)
t_std = x.std(axis=0)
x = x.sub(t_mean).div(t_std)
test = test.sub(t_mean).div(t_std)

#x = x.transpose()
#x = (x - x.mean())/((((x**2).sum()/len(x.index)) - x.mean()**2)**0.5)
#x = x.transpose()
x.columns = range(features) #87
x[features] = pd.Series(1, index=range(len(x.index)))
test.columns = range(features) #87
test[features] = pd.Series(1, index=range(len(test.index)))

for t in range(1,2000):
    s = x.dot(args)
    grad = (1/(1+(math.e**(x.dot(args)*(-1))))-y[0]).dot(x)#.sum()
    mt = b1*mt+(1-b1)*grad
    vt = b2*vt+(1-b2)*(grad**2)
    args -= 0.001*(mt/(1-(b1**t)))/(((vt/(1-(b2**t)))**0.5)+eps)

ans = 1/(1+(math.e**(test.dot(args)*(-1))))
print 'id,label'
for i in range(0,len(test.index)): #16281
    if ans[i] >= 0.5:
        #if y[0][i] == 1:
        #    zz += 1
        print '{},{}'.format(i+1, 1)
    else:
        #if y[0][i] == 0:
        #   zz += 1
        print '{},{}'.format(i+1, 0)

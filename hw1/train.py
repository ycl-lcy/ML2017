import sys
import pandas as pd

bias = 0
prev_grad_b = 0
args = pd.Series(0, index=range(9))
args = args.astype(float)
prev_grad_a = pd.Series(0, index=range(9))
data = pd.read_csv(sys.argv[1], encoding="big5")
test_data = pd.read_csv(sys.argv[2], encoding="big5")
train_data = pd.DataFrame(index=range(10), columns=range(5652))

data = data.replace({'NR': 0}, regex=True)
test_data = test_data.replace({'NR': 0}, regex=True)
data = data.loc[data[data.columns[2]] == 'PM2.5']
data = data.drop(data.columns[[0, 1, 2]], axis=1)
data = data.astype(float)
data = data.values.flatten().tolist()
#data2 = [i**2 for i in data]
for month in range(12):
    for hour in range(471):
        train_data[month*471+hour] = data[month*480+hour: month*480+hour+10] #+ data2[month*480+hour: month*480+hour+9] + data[month*480+hour+9: month*480+hour+10]
train_data = train_data.transpose();

x = train_data.drop([9],axis=1)
y = train_data[9]
for i in range(18000):
    s = x.dot(args)+bias-y
    loss = (s**2).sum()/5652
    #print loss
    grad_a = x.mul(s, axis=0).sum()*2/5652
    grad_b = s.sum()*2/5652
    prev_grad_a += grad_a**2
    prev_grad_b += grad_b**2
    args -= 1*grad_a/(prev_grad_a**0.5)
    bias -= 1*grad_b/(prev_grad_b**0.5)

print 'id,value'
test_data = test_data.loc[test_data[test_data.columns[1]] == 'PM2.5']
test_data = test_data.drop(test_data.columns[[0, 1]], axis=1)
test_data.columns = range(test_data.shape[1])
test_data = test_data.transpose()
test_data.columns = range(test_data.shape[1])
test_data = test_data.astype(float)
#test_data2 = pd.DataFrame(index=range(19), columns=range(240))
ans = args.dot(test_data)+bias
for i in range(240):
    print 'id_{},{}'.format(i,ans[i])


# args2 = args[9:18]
# args2.index = range(9)
# for i in range(240):
    # ans = args[0:9].dot(test_data[i]) + args2.dot(test_data[i]**2) + bias
    # print 'id_{},{}'.format(i,ans)

import numpy as np
from PIL import Image
import string
import scipy.misc
#jpgfile = Image.open("faces/A01.bmp")

faces = np.zeros((100, 4096))
x = 0
for i in string.uppercase[:10]:
    for j in range(10):
        print"faces/"+i+"0"+str(j)+".bmp"
        faces[x] = np.reshape(np.array(Image.open("faces/"+i+"0"+str(j)+".bmp")), 64*64)
        x += 1
#faces = np.array(map(lambda x: np.reshape(np.array(Image.open("faces/" + x + "01.bmp")), 64*64), string.uppercase[:10]))
faces_mean = faces.mean(axis=0, keepdims=True)
scipy.misc.imsave("avgfaces/avg.bmp", np.reshape(faces_mean, (64, 64)))
faces_ctr = faces - faces_mean
u, s, v = np.linalg.svd(faces_ctr)
map(lambda i: scipy.misc.imsave("eigenfaces/"+str(i)+".bmp", np.reshape(v[i], (64, 64))), range(10))
for ii in range(1,101):
    print ii
    vv = v[:ii]
    print vv.shape
    c = vv.dot(faces_ctr.T)
    re = vv.T.dot(c).T + faces_mean
    x = 0
    for i in string.uppercase[:10]:
        for j in range(10):
            scipy.misc.imsave("refaces/re_"+i+"0"+str(j)+".bmp", np.reshape(re[x], (64, 64)))
            x += 1
    loss = 0
    for i in range(100):
        for j in range(4096):
            loss += abs(faces[i][j] - re[i][j])**2
    loss = loss/(100*4096)
    loss = loss**(0.5)
    print loss/256
    #loss = abs(faces - re).sum()

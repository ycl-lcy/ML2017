import numpy as np
from PIL import Image
import string
import scipy.misc

# eigenfaces = Image.new('L', (64*3, 64*3))
# for i in range(3):
    # for j in range(3):
        # f = Image.open("eigenfaces/"+str(i*3+j)+".bmp")
        # f = f.convert('RGB')
        # eigenfaces.paste(f, (j*64, i*64))
# eigenfaces.save("eigenfaces/eigenfaces3x3.jpg")

# eigenfaces = Image.new('L', (64*10, 64*10))
# for i in string.uppercase[:10]:
    # for j in range(10):
        # f = Image.open("faces/"+i+"0"+str(j)+".bmp")
        # f = f.convert('RGB')
        # eigenfaces.paste(f, (j*64, (ord(i)-65)*64))
# eigenfaces.save("faces/faces10x10.jpg")

eigenfaces = Image.new('L', (64*10, 64*10))
for i in string.uppercase[:10]:
    for j in range(10):
        f = Image.open("refaces/re_"+i+"0"+str(j)+".bmp")
        f = f.convert('RGB')
        eigenfaces.paste(f, (j*64, (ord(i)-65)*64))
eigenfaces.save("refaces/refaces10x10.jpg")

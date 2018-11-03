import sys
from PIL import Image, ImageChops

a = Image.open(sys.argv[1])
b = Image.open(sys.argv[2])

width, height=a.size
c = Image.new("RGBA", (width, height), (0, 0, 0, 0))

for i in range(width):
    for j in range(height):
        if a.getpixel((i, j)) != b.getpixel((i, j)):
            c.putpixel((i, j), b.getpixel((i, j)))

c.save("ans_two.png")

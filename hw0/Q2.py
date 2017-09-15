from PIL import Image
import sys

file = sys.argv[1]

#read westbrook image----
im = Image.open(file)

#compute /2----
out = im.point(lambda i: i/2)
#out.show()
out.save("Q2.png")

#testing----
"""
pix = im.load()
pixo = out.load()
print(pix[0,0])
print(pix[0,1])
print(pixo[0,0])
print(pixo[0,1])
"""




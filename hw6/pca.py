from skimage import io
import numpy as np
import sys

path1 = sys.argv[1]
input_img = sys.argv[2]
#path1 = './Aberdeen/'

img = []
for i in range(415):
    img.append(io.imread(path1 + '/' + str(i) + '.jpg'))
img = np.array(img)
img = img.reshape(415, -1)

'''
1-1
'''
img_mean = np.mean(img, axis=0)
#io.imsave('mean.jpg', np.reshape(img_mean, (600, 600, 3)).astype(np.uint8))

'''
svd
'''
U, s, V = np.linalg.svd((img-img_mean).T, full_matrices=False)
#np.save('U.npy', U)
#np.save('s.npy', s)
'''
1-2
U = np.load('U.npy')
Ut = U.T
for i in range(4):
    M = Ut[i]
    path = 'egenfaces'+str(i)+'.jpg'
    M -= np.min(M)
    M /= np.max(M)
    M = (M*255).astype(np.uint8)
    io.imsave(path, M.reshape(600,600,3).astype(np.uint8))
'''

'''
1-3
'''
#U = np.load('U.npy')
inputImg = io.imread(path1 + '/' + input_img)
inputImg = np.array(inputImg )
inputImg = inputImg.flatten()

W = []
for i in range(4):
    W.append(np.dot((inputImg-img_mean), U.T[i]))

result = img_mean
for i in range(4):
    result += (W[i] * U.T[i])

result -= np.min(result)
result /= np.max(result)
result = (result*255).astype(np.uint8)
io.imsave('reconstruction.jpg', result.reshape(600,600,3))

'''
1-4
s = np.load('s.npy')
print('0:', s[0]/np.sum(s))
print('1:', s[1]/np.sum(s))
print('2:', s[2]/np.sum(s))
print('3:', s[3]/np.sum(s))
'''


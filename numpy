#numpy
import numpy as np
def npSum():
    a=np.array([0,1,2,3,4])
    b=np.array([9,8,7,6,5])
    c=a**2+b**3
    return c

a=np.array([[0,1,2,3,4],[5,6,7,8,9]])
print(a.ndim)
print(a.shape)
a.size
a.dtype
a.itemsize

a=np.array([[0,1,2,3,4],[5,6,7,8]])
print(a.ndim)
print(a.shape)
a.size
a.dtype
a.itemsize

print(np.arange(10)) #elements range from 0 to 9
print(np.ones((3,6))
print(np.zeros(shape))
print(np.full(shape,val))#according to the shape, all of the val
print(np.eye(5))#5*5 all 0 except the diagonal of 1

a=np.linspace(1,10,4) #等距从1到10
b=np.linspace(1,10,4,endpoint=False)
c=np.concatenate((a,b))

a=np.ones((2,3,4),dtype=np.int32)
print(a.reshape((3,8))) #no modification
print(a)
print(a.resize(3,8))
print(a)
print(a.flatten())

a=np.ones((2,3,4),dtype=np.int32)
a.tolist()

a=np.array([5,6,7,8,9])
print(a[2])
print(a[1:4:2])#from 1st to 3rd, the distance is 2

a=np.arange(24).reshape((2,3,4))
print(a)
print(a[1,2,3])
print(a[-1,-2,-3])
print(a[:,1,-3])
print(a[:,1:3,:])
print(a[:,:,::2])

a=np.arange(24).reshape((2,3,4))
print(a.mean())
a=a.mean()
print(a)

np.abs(x)
np.sqrt(x)
np.square(x)
np.log(x)
np.log10(x)
np.log2(x)
np.ceil(x)
np.floor(x)
np.rint(x)#每个元素四舍五入
np.modf(x)##各元素拆开整数与小数部分
np.cos(x)
np.cosh(x)
np.exp(x)
np.sign(x)#返回正数还是负数

a=np.arange(24).reshape((2,3,4))
b=np.sqrt(a)
print(np.maximum(a,b)) #the result is float if one of them is float
print(a>b)

#read csv
a=np.arange(100).reshape(5,20)
print(np.savetxt('a.csv',a,fmt='%d',delimiter=','))
print(np.savetxt('a.csv',a,fmt='%.1f',delimiter=','))

a=np.arange(100).reshape(5,10,2)
a.tofile('b.dat',sep=',',format='%d')

a=np.arange(100).reshape(5,10,2)
np.save('a.npy',a)
b=np.load('a.npy')

a=np.random.randint(100,200,(3,4))
print(np.random.shuffle(a))#a changed
print(a)
a=np.random.randint(100,200,(3,4))
print(np.random.permutation(a))#a not changed
print(a)

u=np.random.uniform(0,10,(3,4))
u=np.random.normal(10,5,(3,4))#normal distribution, mean=10,var=5

a=np.arange(15).reshape(3,5)
print(a)
np.sum(a)
np.mean(a,axis=1)
np.mean(a,axis=0)
np.std(a)
np.var(a)
np.average(a,axis=0,weights=[10,5,1])

a=np.random.randint(0,20,(5))
print(a)
print(np.gradient(a))

a=np.random.randint(0,50,(3,5))
print(a)
print(np.gradient(a))

#picture
from PIL import Image
import numpy as np
im=np.array(Image.open("D:/pycodes/beijing.jpg"))
print(im.shape,im.dtype)

b=[255,255,255]-im #change pixels
a=Image.fromarray(b.astype('uint8'))
im.save("")

im=np.array(Image.open("Pictures/港珠澳大桥.PNG").convert('L'))
print(im.shape,im.dtype)
b=255-im #change pixels
a=Image.fromarray(b.astype('uint8'))
im.save("")

#素描
from PIL import Image
import numpy as np
a=np.asarray(Image.open('Pictures/港珠澳大桥.PNG').convert('L')).astype('float')
depth=10. #(0-100)
grad=np.gradient(a) #取图像灰度的梯度值
grad_x,grad_y=grad #分别取横纵图像梯度值
grad_x=grad_x*depth/100
grad_y=grad_y*depth/100
A=np.sqrt(grad_x**2+grad_y**2+1.)
uni_x=grad_x/A
uni_y=grad_y/A
uni_z=1./A
vec_el=np.pi/2.2 #光源俯视角度，弧度值
vec_az=np.pi/4 #光源方位角度，弧度值
dx=np.cos(vec_el)*np.cos(vec_az) #光源对x轴影响
dy=np.cos(vec_el)*np.sin(vec_az) #光源对y轴影响
dz=np.sin(vec_el) #光源对z轴影响
b=255*(uni_x+dy*uni_y+dz*uni_z) #光源归一化
b=b.clip(0,255)
im=Image.fromarray(b.astype('uint8')) #重构图像
im.save('Pictures/港珠澳大桥素描.PNG')

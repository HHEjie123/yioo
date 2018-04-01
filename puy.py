import cv2
import numpy as np
import  time

def imgread(imgcutxia,r,g,b):
    x=(imgcutxia[:,:,0]>r)
    y=(imgcutxia[:,:,1]>g)
    z= (imgcutxia[:, :, 2] > b)
    imgcutxia[(x & y & z),:]=0
    NpKernel = np.uint8(np.zeros((3, 3)))
    for i in range(3):
        NpKernel[2, i] = 1  # 感谢chenpingjun1990的提醒，现在是正确的
        NpKernel[i, 2] = 1
    imgcutxia = cv2.erode(imgcutxia, NpKernel)
    x = (imgcutxia[:, :, 0] > 10)
    y = (imgcutxia[:, :, 1] > 10)
    z = (imgcutxia[:, :, 2] > 10)
    p=np.sum((x & y & z))
    return  p
def huojia(picture,p,r,g,b):
    shang=0
    xia=0
    img = cv2.imread(picture)
    imgcutshang = img[0:280, 100:640]
    imgcutxia= img[260:480, 100:640]
    p=imgread(imgcutshang,r,g,b)
    print(picture,'shang has  ',p)
    if(p>500):
        print(picture,'shang is you')
        shang=1
    else:
        print((picture,'shang is wu'))
    p = imgread(imgcutxia,r,g,b)
    print(picture,'xia  has  ', p)
    if (p > 500):
        print(picture,'xia is you')
        xia=1
    else:
        print((picture,'xia is wu'))
    cv2.imwrite('%s_shang.jpg'%(picture),imgcutshang)
    cv2.imwrite('%s_xia.jpg'%(picture),imgcutxia)
    cv2.imwrite('maskshang.jpg',imgcutshang)
    cv2.imwrite('maskxia.jpg', imgcutxia)
    return shang,xia
stri=time.clock()
shang,xia=huojia('3.jpg',p=500,r=70,g=80,b=85)
print(shang,xia)
shang,xia=huojia('1.jpg',p=500,r=70,g=80,b=85)
print(shang,xia)
shang,xia=huojia('2.jpg',p=500,r=70,g=80,b=85)
print(shang,xia)
end = time.clock()
print('cost time is',end-stri)


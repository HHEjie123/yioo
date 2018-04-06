## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################
import sys

sy_root = '/home/nvidia/realsense/librealsense-master/build/wrappers/python'
sys.path.insert(0, sy_root)
# First import the library
#import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
import numpy as np
import time

def hough(picture):
    img=cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY)
    #uu=rgb2gray(picture)
    cv2.imshow("gray", img)
    #gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
    #gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

    #cv2.imshow("gradX", gradX)
    #x=cv2.waitKey(1)& 0xFF
    #cv2.imshow("gradY", gradY)
    #x=cv2.waitKey(1)
    # subtract the y-gradient from the x-gradient
    #gradient = cv2.subtract(gradX, gradY)
    #img = cv2.convertScaleAbs(gradient)
    img = cv2.blur(img, (3, 3))
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    cv2.imshow('cannyuu',edges)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 118)  # 这里对最后一个参数使用了经验型的值
    result = img.copy()
    shuipingx=[]
    if(lines is not None):
        for lop in range(int(lines.size/2)):
            for line in lines[lop]:
                rho = line[0]  # 第一个元素是距离rho
                theta = line[1]  # 第二个元素是角度theta
                print(rho)
                print(theta)
                if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
                    # 该直线与第一行的交点
                    pt1 = (int(rho / np.cos(theta)), 0)
                    # 该直线与最后一行的焦点
                    pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                    # 绘制一条白线
                    cv2.line(result, pt1, pt2,0,1)
                else:  # 水平直线
                    # 该直线与第一列的交点
                    pt1 = (0, int(rho / np.sin(theta)))
                    # 该直线与最后一列的交点
                    pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                    # 绘制一条直线
                    if(pt2[0]-pt1[0]>0):
                       cv2.line(result, pt1, pt2,0, 1)
                       shuipingx.append(pt1[1])
                    print(('pt2=',pt2,'pt1  =',pt1))
            #if shuipingx>0:
             #   break
    return result, shuipingx
def imgread(picture,imgcutxia, r, g, b):
    x = (imgcutxia[:, :, 0] > r)
    y = (imgcutxia[:, :, 1] > g)
    z = (imgcutxia[:, :, 2] > b)
    imgcutxia[(x & y & z), :] = 0
    x = (imgcutxia[:, :, 0] < 8)
    y = (imgcutxia[:, :, 1] < 8)
    z = (imgcutxia[:, :, 2] < 8)
    imgcutxia[(x & y & z), :] = 0
    """
    NpKernel = np.uint8(np.zeros((3, 3)))
    for i in range(3):
        NpKernel[2, i] = 1  # 感谢chenpingjun1990的提醒，现在是正确的
        NpKernel[i, 2] = 1
    imgcutxia = cv2.erode(imgcutxia, NpKernel)
    """
    x = (imgcutxia[:, :, 0] > 10)
    y = (imgcutxia[:, :, 1] > 10)
    z = (imgcutxia[:, :, 2] > 10)

    p = np.sum((x & y & z))
    return p


def huojia(picture, pix, r, g, b):
    shang=0
    xia=0
    img = cv2.imread(picture)
    imgcutshang = img[140:300, 150:550]
    _,shuipingx=hough(imgcutshang)
    max1=max(shuipingx)
    min2=min(shuipingx)
    print('shuipilllllllllllllllllllllllllllllllllllllllllllllllllllllngx',max1,min2)
    #imgcutshang = img[(140+max1-130):(140+max1+4), 150:550]
    imgcutshang = img[(140 + max1 - 130):(140 + max1 -15), 200:480]
    cv2.imwrite('%s_yuanshang.jpg' % (picture), imgcutshang)
    #imgcutxia= img[140+max1+70:480, 150:550]
    imgcutxia = img[140 + max1 + 70:480, 200:480]
    cv2.imwrite('%s_yuanxia.jpg' % (picture), imgcutxia)

    # cv2.imwrite('%s_yuanxia.jpg' % (picture), imgcutxia)
    p1 = imgread(picture + 'shang', imgcutshang, r, g, b)
    print(picture, 'shang has  ', p1)
    if (p1 > pix):
        print(picture, 'shang is you')
        shang = 1
    else:
        print((picture, 'shang is wu'))
    p2 = imgread(picture + 'xia', imgcutxia, r, g, b)
    print(picture, 'xia  has  ', p2)
    if (p2 > pix):
        print(picture, 'xia is you')
        xia = 1
    else:
        print((picture, 'xia is wu'))
    cv2.imwrite('%s_shang.jpg' % (picture), imgcutshang)
    cv2.imwrite('%s_xia.jpg' % (picture), imgcutxia)
    cv2.imwrite('maskshang.jpg', imgcutshang)
    cv2.imwrite('maskxia.jpg', imgcutxia)
    return shang, xia, p1, p2


# Streaming loop
def picture():
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print
    "Depth Scale is: ", depth_scale

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 0.65  # 1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    print('  start  ')

    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image
        # Align the depth frame to color frame
        aligned_frames = align.proccess(frames)
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 0
        depth_image_3d = np.dstack(
            (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        # Render images
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.09), cv2.COLORMAP_JET)
        # images = np.hstack((bg_removed, depth_colormap))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', bg_removed)
        key = cv2.waitKey(110)
        # print key
        picname = 6
        for lm in range(picname + 1):
            if key == ord('%d' % (lm)):
                cv2.imwrite('picture/%d.jpg' % (lm), bg_removed)
                cv2.imwrite('picture/color_%d.jpg' % (lm), color_image)

        if key == ord('s'):
            print('kjkk')
            pipeline.stop()
            break
    """


        # Get frameset of color and depth

    frames = pipeline.wait_for_frames()
    # frames.get_depth_frame() is a 640x360 depth image

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    # Validate that both frames are valid
    #if not aligned_depth_frame or not color_frame:
     #   continue

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Remove background - Set pixels further than clipping_distance to grey
    grey_color = 0
    depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
    bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
    # Render images
    #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.09), cv2.COLORMAP_JET)
    #images = np.hstack((bg_removed, depth_colormap))
    #cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
    #cv2.imshow('Align Example', bg_removed)
    key = cv2.waitKey(12)
    #print key
    cv2.imwrite('picture/%d.jpg' % (1), bg_removed)
    time.sleep(4)
    cv2.imwrite('picture/%d.jpg' % (2), bg_removed)
    time.sleep(4)
    cv2.imwrite('picture/%d.jpg' % (3), bg_removed)
    time.sleep(4)
    cv2.imwrite('picture/%d.jpg' % (4), bg_removed)
    time.sleep(4)
    cv2.imwrite('picture/%d.jpg' % (5), bg_removed)
    time.sleep(4)
    cv2.imwrite('picture/%d.jpg' % (6), bg_removed)
    if key ==ord('1'):
        cv2.imwrite('picture/%d.jpg'%(1),bg_removed)
        cv2.imwrite('picture/color_%d.jpg' % (1),color_image)
    if key ==ord('2'):
        cv2.imwrite('picture/_%d.jpg'%(2),bg_removed)
        cv2.imwrite('picture/color_%d.jpg' % (2), color_image)
    if key ==ord('3'):
        cv2.imwrite('picture/_%d.jpg'%(3),bg_removed)
        cv2.imwrite('picture/color_%d.jpg' % (3), color_image)
    if key == ord('4'):
        cv2.imwrite('picture/%d.jpg' % (4), bg_removed)
        cv2.imwrite('picture/color_%d.jpg' % (4), color_image)
    if key == ord('5'):
        cv2.imwrite('picture/%d.jpg' % (5), bg_removed)
        cv2.imwrite('picture/color_%d.jpg' % (5), color_image)
    if key == ord('6'):
        cv2.imwrite('picture/%d.jpg' % (6), bg_removed)
        cv2.imwrite('picture/color_%d.jpg' % (6), color_image)
    if key == ord('s'):
        print('kjkk')
    print('true  is  pp')
    time.sleep(5)
    cv2.destroyAllWindows()
    pipeline.stop()
    """


def huop():
    # alist1=[]
    # alist2=[]
    pix1 = []
    pix2 = []
    pix3 = []
    #picture()
    start = time.clock()
    for kl in range(1, 7):
        shang, xia, p1, p2 = huojia('picture/%d.jpg' % (kl), pix=800, r=70, g=70, b=75)
        # alist1.append([shang])
        #  alist2.append([xia])
        pix3.append(p1)
        pix3.append(p2)
        pix1.append(p1)
        pix2.append(p2)
    # alist=alist2+alist1
    pixt = pix1 + pix2
    pix3.sort()
    # print(alist)
    print(pixt)
    print((pix3))
    for xpo in pixt:
        if xpo == pix3[0] or xpo == pix3[1] or xpo == pix3[2]:
            pixt[pixt.index(xpo)] = 0
        else:
            pixt[pixt.index(xpo)] = 1

    # print(alist)
    print(pixt)
    last_tiem = [[pixt[0], pixt[1], pixt[2], pixt[3], pixt[4], pixt[5]],
                 [pixt[6], pixt[7], pixt[8], pixt[9], pixt[10], pixt[11]
                  ]]
    # print(' last item',last_tiem)
    end = time.clock()
    print('cost time is', end - start)
    return last_tiem


if __name__ == "__main__":
    """
    cap = cv2.VideoCapture(3)
    cap.set(3, 1920)
    cap.set(4, 1080)
    time.sleep(2)
    ret, frame = cap.read()
    print(frame.shape)
    cv2.imshow('hhhh', frame)
    cv2.waitKey(110)
    time.sleep(0.74)
    cv2.imwrite('ggggg.jpg', frame)
    print('has done  ')
    cap.release()
    cv2.destroyAllWindows()
    """
    last_tiem = huop()
    print(' last item', last_tiem)

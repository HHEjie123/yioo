#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import time

# from forcaption import RobotCarHandle

# CLASSES = ('__background__',
#            'greenzhuan', 'blue zhuan', 'xuehua', 'green zhuan',
#                                 'xuebi', 'red zhuan', 'red zhuan', 'shuangwaiwai', 'apple',
#                                 'apple','gangsiqiu','wangqiu')

CLASSES = ('__background__', # always index 0
                        'red zhuan', 'green zhuan', 'blue zhuan','yellow zhuan','yangleduo','xuebi','xuehua','shuangwaiwai','gangsiqiu','wangqiu','yumaoqiu','apple')
NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
               '3318000ZF_faster_rcnn_final.caffemodel')}


def zuobiao(x1, x2, x3, x4):
    return x1, x2, x3, x4


def vis_detections(im, class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    x1 = 0
    x2 = 0
    x3 = 0
    x4 = 0
    if len(inds) == 0:
        return x1, x2, x3, x4, class_name

    # im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        x1, x2, x3, x4 = zuobiao(bbox[0], bbox[1], bbox[2], bbox[3])
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.draw()
    print 'kkkk', class_name, ' ', x1, ' ', x2, ' ', x3, ' ', x4
    # if class_name != None :
    return x1, x2, x3, x4, class_name


def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.6
    NMS_THRESH = 0.3
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        x1, x2, x3, x4, class_name = vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)
        if (x1 != 0 and x1 < 1600):
            if x1 < 600:
                print>> f, 'left', class_name
                dictf[image_name+'left']=class_name
            elif x1 < 1000:
                print>> f, 'mid', class_name
                dictf[image_name+'mid'] =  class_name
            else:
                print>> f, 'right', class_name
                dictf[image_name+'right'] =  class_name
    plt.axis('off')
    plt.tight_layout()
    plt.draw()




def load_model():
    t = time.time()
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    prototxt = os.path.join(cfg.MODELS_DIR, NETS["zf"][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS["zf"][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    caffe.set_mode_gpu()
    caffe.set_device(0)
    cfg.GPU_ID = 0
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    return net
    print '\n\nLoaded network {:s}'.format(caffemodel)
def recon():
    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(net, im)

    im_names = ['1.jpg', '2.jpg', '3.jpg', '4.jpg']
    f = open('out.txt', 'w')
    # filepath = time.strftime('%Y-%m-%d-%H-%M-%S')
    filepath = "ssss"
    # vision = RobotCarHandle(filepath)
    # vision.open_camera()
    # vision.camera_takephoto(29);
    # vision.close_camera();
    dictf={}
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        print>> f, im_name
        print(im_name)
        demo(net, im_name)

    itemlistp={}
    itemlist=('1.jpgleft','1.jpgmid','1.jpgright','2.jpgleft','2.jpgmid','2.jpgright',
              '3.jpgleft','3.jpgmid','3.jpgright','4.jpgleft','4.jpgmid','4.jpgright')
    for klp in range(len(itemlist)):
        if itemlist[klp] in dictf:
            itemlistp[itemlist[klp]]=dictf[itemlist[klp]]
        else:
            itemlistp[itemlist[klp]]='None'
    item_list={"A": {"left": itemlistp['1.jpgleft'], "mid": itemlistp['1.jpgmid'], "right": itemlistp['1.jpgright']},
               "B": {"left": itemlistp['2.jpgleft'], "mid": itemlistp['2.jpgmid'], "right": itemlistp['2.jpgright']},
               "C": {"left": itemlistp['3.jpgleft'], "mid": itemlistp['3.jpgmid'], "right": itemlistp['3.jpgright']},
               "D": {"left": itemlistp['4.jpgleft'], "mid": itemlistp['4.jpgmid'], "right": itemlistp['4.jpgright']}}

    return item_list


    f.close()
    print "Time:", time.time() - t
    #plt.show()
load_model()
item_list=recon()


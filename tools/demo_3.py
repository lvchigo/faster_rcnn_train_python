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
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

#CLASSES = ('__background__',
#           'aeroplane', 'bicycle', 'bird', 'boat',
#           'bottle', 'bus', 'car', 'cat', 'chair',
#           'cow', 'diningtable', 'dog', 'horse',
#           'motorbike', 'person', 'pottedplant',
#           'sheep', 'sofa', 'train', 'tvmonitor')

#faceness
#CLASSES = ('__background__', 'face', 'eye', 'mouse', 'nose', 'hair', 'bread')
CLASSES = ('__background__', 'face', 'eye', 'mouse', 'nose', 'hair', 'bread', 'other', 'pet', 'puppet', 'sticker')

#90+1class
#CLASSES = ('__background__', # always index 0
#'food.bread', 'food.candy', 'food.coffee', 'food.cookie', 'food.crab', 'food.diningtable.diningtable', 'food.dumpling', 'food.food.barbecue', 'food.food.cake', 'food.food.cook',
#'food.food.food', 'food.food.fruit', 'food.food.hotpot', 'food.food.icecream', 'food.hamburger', 'food.pasta', 'food.pizza', 'food.rice', 'food.steak', 'food.sushi',
#'goods.airplane.airplane', 'goods.bag.bag', 'goods.bangle', 'goods.bottle.bottle', 'goods.bracelet', 'goods.camera.camera', 'goods.car.bicycle', 'goods.car.bus', 'goods.car.car', 'goods.car.motorbike',
#'goods.car.train', 'goods.chair.chair', 'goods.clothes.clothes', 'goods.drawbar', 'goods.goods.cosmetics', 'goods.goods.flower', 'goods.goods.glass', 'goods.goods.goods', 'goods.goods.jewelry', 'goods.goods.manicure',
#'goods.goods.watch', 'goods.guitar', 'goods.hat', 'goods.laptop', 'goods.lipstick.lipstick', 'goods.pendant', 'goods.phone', 'goods.pottedplant.pottedplant', 'goods.puppet', 'goods.ring',
#'goods.ship.ship', 'goods.shoe.shoe', 'goods.sofa.sofa', 'goods.tvmonitor.tvmonitor', 'other', 'other.2dcode.2dcode', 'other.sticker.sticker', 'other.text.text', 'people.eye.eye', 'people.friend.friend',
#'people.hair.hair', 'people.kid.kid', 'people.lip.lip', 'people.people.people', 'people.self.female', 'people.self.male', 'people.street.street', 'pet.alpaca', 'pet.bird.bird', 'pet.cat.cat',
#'pet.cow.cow', 'pet.dog.dog', 'pet.horse.horse', 'pet.pet.pet', 'pet.rabbit', 'pet.sheep.sheep', 'scene.clothingshop', 'scene.courtyard', 'scene.desert', 'scene.forest',
#'scene.highway', 'scene.mountain', 'scene.scene.grasslands', 'scene.scene.house', 'scene.scene.scene', 'scene.scene.sky', 'scene.scene.supermarket', 'scene.sea', 'scene.street', 'scene.tallbuilding' )

NETS = {'vgg16': ('VGG16', 'VGG16_faster_rcnn_final_face1000sample_1000iter.caffemodel'),
        'zf': ('ZF', 'ZF_faster_rcnn_final.caffemodel')}

#'zf': ('ZF','ZF_faster_rcnn_final.caffemodel')
#'zf': ('ZF','ZF_faster_rcnn_final_in.caffemodel')

COLORS = ( (0,0,255), (0,255,255), (0,255,0),   (255,255,0),
           (255,0,0), (255,0,255), (0,128,255), (255,128,0))

def save_image_oneimage_onelabel(input_file, output_path, im_name, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
        
    img = cv2.imread(input_file)
    #print 'input_file:{}'.format(im_file)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        output_file = os.path.join( output_path, class_name, '{:s}_{:.3f}_{:s}'.format(class_name,score,im_name) )      
        #print 'output_file:{}'.format(output_file)

        cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),COLORS[i%8],2)

        text='{:s}_{:.3f}'.format(class_name, score)
        cv2.putText(img,text,(int(bbox[0]+5),int(bbox[1]+15)), font, 0.5,COLORS[i%8],2,8) 
        
    cv2.imwrite(output_file, img)

def demo_oneimage_onelabel(net, im_file, output_path, im_name, isPrint):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    if isPrint == 1:
        print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #vis_detections(im, cls, dets, thresh=CONF_THRESH)
        save_image_oneimage_onelabel(im_file, output_path, im_name, cls, dets, thresh=CONF_THRESH)

def demo_oneimage_mutilabel(net, im_file, output_path, im_name, isPrint):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(im_file)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    if isPrint == 1:
        print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3

    #save img
    Sv_Img = 0

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            continue

        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),COLORS[i%8],2)

            text='{:s}_{:.3f}'.format(cls, score)
            cv2.putText(im,text,(int(bbox[0]+5),int(bbox[1]+15)), font, 0.5,COLORS[i%8],2,8) 

            #save image
            if cls_ind<7:
                Sv_Img = 1

    if Sv_Img==1:
        # save image
        output_file = os.path.join( output_path, '{:s}'.format(im_name) )      
        #print 'output_file:{}'.format(output_file)
        cv2.imwrite(output_file, im)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [zf]',
                        choices=NETS.keys(), default='vgg16')
    #parser.add_argument('--input', dest='input_file', default='/home/xiaogao/job/research/faster-rcnn/py-faster-rcnn-master_add/data/imglist/list_fddb_img.txt')
    parser.add_argument('--input', dest='input_file', default='/home/xiaogao/img/facedetect/img_add_20160315/imglist.txt')
    parser.add_argument('--output', dest='output_path', default='/home/xiaogao/job/research/faster-rcnn/py-faster-rcnn-master_add/data/output_img/')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    output_dir = args.output_path[:args.output_path.rfind('/')]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
#    for cls_ind, cls in enumerate(CLASSES[1:]):
#        cls_dir = '{:s}/{:s}'.format(output_dir,cls)
#        if not os.path.exists(cls_dir):
#            os.makedirs(cls_dir)

    file = open(args.input_file, "r")
    alllines=file.readlines();
    file.close();

    nCount = 0;
    for line in alllines:

        #print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        #print 'line:{}'.format(line)
        im_name = line[(line.rfind('/')+1):];
        im_name = im_name[:im_name.rfind('.')];
        im_name = '{:s}.jpg'.format(im_name)
        input_path = line[:line.rfind('/')];
        im_file = '{:s}/{:s}'.format(input_path,im_name)
        #print 'input_file:{}'.format(im_file)

        isPrint = 0;
        nCount = nCount+1;    
        if nCount%50 == 0:
            print 'load img:{}...'.format(nCount)
            isPrint = 1;
        else:
            isPrint = 0;

        #add by chigo                    
        #demo_oneimage_onelabel(net, im_file, args.output_path, im_name, isPrint)
        demo_oneimage_mutilabel(net, im_file, args.output_path, im_name, isPrint)

    print 'All load img:{}!!'.format(nCount)
    

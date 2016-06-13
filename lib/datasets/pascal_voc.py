# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.pascal_voc
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess

class pascal_voc(datasets.imdb):
    def __init__(self, image_set, year, devkit_path=None):
        datasets.imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        
        #voc2007
#        self._classes = ('__background__', # always index 0
#                         'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
#                         'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

        # mutilabel
        self._classes = ('__background__', # always index 0
'animal.bird', 'animal.cat', 'animal.cow', 'animal.dog', 'animal.horse',
'animal.other', 'animal.panda', 'animal.rabbit', 'animal.sheep', 'electronics.camera',
'electronics.cellphone', 'electronics.keyboard', 'electronics.monitor', 'electronics.notebook', 'electronics.other',
'food.barbecue', 'food.cake', 'food.coffee', 'food.cook', 'food.fruit',
'food.hotpot', 'food.icecream', 'food.other', 'food.pizza', 'food.sushi',
'furniture.chair', 'furniture.diningtable', 'furniture.other', 'furniture.pottedplant', 'furniture.sofa',
'goods.bag', 'goods.ball', 'goods.book', 'goods.bottle', 'goods.clock',
'goods.clothes', 'goods.cosmetics', 'goods.cup', 'goods.drawbar', 'goods.flower',
'goods.glass', 'goods.guitar', 'goods.hat', 'goods.jewelry', 'goods.other',
'goods.puppet', 'goods.shoe', 'other.2dcode', 'other.logo', 'other.other',
'other.sticker', 'other.text', 'person.body', 'person.face', 'person.other',
'vehicle.airplane', 'vehicle.bicycle', 'vehicle.boat', 'vehicle.bus', 'vehicle.car',
'vehicle.motorbike', 'vehicle.other', 'vehicle.train')

        #coco
#        self._classes = ('__background__', # always index 0
#            'accessory.backpack', 'accessory.handbag', 'accessory.suitcase', 'accessory.tie', 'accessory.umbrella', 
#            'animal.bear', 'animal.bird', 'animal.cat', 'animal.cow', 'animal.dog', 
#            'animal.elephant', 'animal.giraffe', 'animal.horse', 'animal.sheep', 'animal.zebra', 
#            'appliance.microwave', 'appliance.oven', 'appliance.refrigerator', 'appliance.sink', 'appliance.toaster', 
#            'electronic.cell phone', 'electronic.keyboard', 'electronic.laptop', 'electronic.mouse', 'electronic.remote', 
#            'electronic.tv', 'food.apple', 'food.banana', 'food.broccoli', 'food.cake', 
#            'food.carrot', 'food.donut', 'food.hot dog', 'food.orange', 'food.pizza', 
#            'food.sandwich', 'furniture.bed', 'furniture.chair', 'furniture.couch', 'furniture.dining table', 
#            'furniture.potted plant', 'furniture.toilet', 'indoor.book', 'indoor.clock', 'indoor.hair drier', 
#            'indoor.scissors', 'indoor.teddy bear', 'indoor.toothbrush', 'indoor.vase', 'kitchen.bottle', 
#            'kitchen.bowl', 'kitchen.cup', 'kitchen.fork', 'kitchen.knife', 'kitchen.spoon', 
#            'kitchen.wine glass', 'outdoor.bench', 'outdoor.fire hydrant', 'outdoor.parking meter', 'outdoor.stop sign', 
#            'outdoor.traffic light', 'person.person', 'sports.baseball bat', 'sports.baseball glove', 'sports.frisbee', 
#            'sports.kite', 'sports.skateboard', 'sports.skis', 'sports.snowboard', 'sports.sports ball', 
#            'sports.surfboard', 'sports.tennis racket', 'vehicle.airplane', 'vehicle.bicycle', 'vehicle.boat', 
#            'vehicle.bus', 'vehicle.car', 'vehicle.motorcycle', 'vehicle.train', 'vehicle.truck')

		#faceness
#        self._classes = ('__background__', 'face', 'eye', 'mouse', 'nose', 'hair', 'bread')
#        self._classes = ('__background__', 'face', 'halfface', 'eye', 'mouse', 'nose', 'hair', 'bread', 'other', 'pet', 'puppet', 'sticker')

        #90+1class
#        self._classes = ('__background__', # always index 0
#'food.bread', 'food.candy', 'food.coffee', 'food.cookie', 'food.crab', 'food.diningtable.diningtable', 'food.dumpling', 'food.food.barbecue', 'food.food.cake', 'food.food.cook',
#'food.food.food', 'food.food.fruit', 'food.food.hotpot', 'food.food.icecream', 'food.hamburger', 'food.pasta', 'food.pizza', 'food.rice', 'food.steak', 'food.sushi',
#'goods.airplane.airplane', 'goods.bag.bag', 'goods.bangle', 'goods.bottle.bottle', 'goods.bracelet', 'goods.camera.camera', 'goods.car.bicycle', 'goods.car.bus', 'goods.car.car', 'goods.car.motorbike',
#'goods.car.train', 'goods.chair.chair', 'goods.clothes.clothes', 'goods.drawbar', 'goods.goods.cosmetics', 'goods.goods.flower', 'goods.goods.glass', 'goods.goods.goods', 'goods.goods.jewelry', 'goods.goods.manicure',
#'goods.goods.watch', 'goods.guitar', 'goods.hat', 'goods.laptop', 'goods.lipstick.lipstick', 'goods.pendant', 'goods.phone', 'goods.pottedplant.pottedplant', 'goods.puppet', 'goods.ring',
#'goods.ship.ship', 'goods.shoe.shoe', 'goods.sofa.sofa', 'goods.tvmonitor.tvmonitor', 'other', 'other.2dcode.2dcode', 'other.sticker.sticker', 'other.text.text', 'people.eye.eye', 'people.friend.friend',
#'people.hair.hair', 'people.kid.kid', 'people.lip.lip', 'people.people.people', 'people.self.female', 'people.self.male', 'people.street.street', 'pet.alpaca', 'pet.bird.bird', 'pet.cat.cat',
#'pet.cow.cow', 'pet.dog.dog', 'pet.horse.horse', 'pet.pet.pet', 'pet.rabbit', 'pet.sheep.sheep', 'scene.clothingshop', 'scene.courtyard', 'scene.desert', 'scene.forest',
#'scene.highway', 'scene.mountain', 'scene.scene.grasslands', 'scene.scene.house', 'scene.scene.scene', 'scene.scene.sky', 'scene.scene.supermarket', 'scene.sea',
#'scene.street', 'scene.tallbuilding' )
                         
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000,
                       'use_diff' : False,
                       'rpn_file' : None}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'VOCdevkit' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        # print 'Loading: {}'.format(filename)
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        #if not self.config['use_diff']:
        #    # Exclude the samples labeled as difficult
        #    non_diff_objs = [obj for obj in objs
        #                     if int(get_data_from_tag(obj, 'difficult')) == 0]
        #    if len(non_diff_objs) != len(objs):
        #        print 'Removed {} difficult objects' \
        #            .format(len(objs) - len(non_diff_objs))
        #    objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            x1 = float(get_data_from_tag(obj, 'xmin')) - 1
            y1 = float(get_data_from_tag(obj, 'ymin')) - 1
            x2 = float(get_data_from_tag(obj, 'xmax')) - 1
            y2 = float(get_data_from_tag(obj, 'ymax')) - 1

            #fixed by chigo--20151224
            cls_str = str(get_data_from_tag(obj, "name")).lower().strip()
            if cls_str not in self._classes:
                cls = 0
            else:
                cls = self._class_to_ind[cls_str]

            #cls = self._class_to_ind[
            #        str(get_data_from_tag(obj, "name")).lower().strip()]
                
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def _write_voc_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # VOCdevkit/results/VOC2007/Main/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(self._devkit_path, 'results', 'VOC' + self._year,
                            'Main', comp_id + '_')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return comp_id

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_voc_results_file(all_boxes)
        self._do_matlab_eval(comp_id, output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed; embed()

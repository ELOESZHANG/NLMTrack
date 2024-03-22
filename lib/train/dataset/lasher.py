import os
import os.path
import numpy as np
import torch
import csv
import json
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class LaSHeR(BaseVideoDataset):
    """ GOT-10k dataset.

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """

    def __init__(self, root=env_settings().lasher_dir, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):
        """
        args:
            root - path to the got-10k training data. Note: This should point to the 'train' folder inside GOT-10k
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - 'train' or 'val'. Note: The validation split here is a subset of the official got-10k train split,
                    not NOT the official got-10k validation split. To use the official validation split, provide that as
                    the root folder instead.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        self.root=root
        self.root_i = root
        super().__init__('LaSHeR', root, image_loader)

        # all folders inside the root
        # self.sequence_list = self._get_sequence_list()

        # seq_id is the index of the folder inside the got10k root path
        if split is not None:
            self.split=split
            if seq_ids is not None:
                raise ValueError('Cannot set both split_name and seq_ids.')
            # ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                self.anno="/media/SSDPA/yanmiao/rgb/SeqTrack/lib/train/data_specs/lasher_train.json"
            elif split == 'val':
                self.anno="/media/SSDPA/yanmiao/rgb/SeqTrack/lib/train/data_specs/lasher_val.json"
            else:
                raise ValueError('Unknown split name.')
        else:
            raise ValueError("split can't be None" )
        with open(self.anno, 'r') as f:
            meta_data = json.load(f)
        # all folders inside the root
        self.sequence_list = list(meta_data.keys())
        for video in self.sequence_list:
            for modal in meta_data[video]:
                frames = meta_data[video][modal]
                frames = list(frames.keys())
                # frames.sort()
                meta_data[video][modal]['frames'] = frames

        if seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))
        # self.seq_ids = seq_ids


        self.labels = meta_data

        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        # if data_fraction is not None:
        #     self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.sequence_meta_info = self._load_meta_info()
        self.seq_per_class = self._build_seq_per_class()

        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()

    def get_name(self):
        return 'lasher'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def _load_meta_info(self):
        if self.split=='train':
            sequence_meta_info = {s: self._read_meta(os.path.join(self.root, "train/", s)) for s in self.sequence_list}
        elif self.split=='val':
            sequence_meta_info = {s: self._read_meta(os.path.join(self.root, "test/", s)) for s in self.sequence_list}
        else: raise ValueError("split can't be None")
        return sequence_meta_info

    def _read_meta(self, seq_path):

        object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        return object_meta

    def _build_seq_per_class(self):
        seq_per_class = {}

        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class_name']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_sequence_list(self):
        if self.split=="train":
            with open("/media/SSDPA/yanmiao/rgb/SeqTrack/lib/train/data_specs/lasher_train.txt") as f:
                # dir_names = f.readlines()
                dir_list = list(csv.reader(f))
        elif self.split=="val":
            with open("/media/SSDPA/yanmiao/rgb/SeqTrack/lib/train/data_specs/lasher_val.txt") as f:
                # dir_names = f.readlines()
                dir_list = list(csv.reader(f))

        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "visible.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "absence.label")
        cover_file = os.path.join(seq_path, "cover.label")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
        with open(cover_file, 'r', newline='') as f:
            cover = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])

        target_visible = ~occlusion & (cover>0).byte()

        visible_ratio = cover.float() / 8
        return target_visible, visible_ratio

    def _get_sequence_path(self, seq_id):
        if self.split=='train':
            return os.path.join(self.root, "train/", self.sequence_list[seq_id])
        elif self.split=='val':
            return os.path.join(self.root, "test/", self.sequence_list[seq_id])
    def get_sequence_name(self,seq_id):
        return self.sequence_list[seq_id]

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        # valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        valid = torch.ceil(torch.sqrt(bbox[:, 2] * bbox[:, 3])) * 5.0 >= 1.0
        # visible = self._read_target_visible(seq_path) & valid

        return {'bbox': bbox, 'valid': valid, 'visible': valid}

    def _get_frame_path(self, seq_path, frames, frame_id):

        return os.path.join(seq_path, frames[frame_id])   # frames start from 1

    def _get_frame(self, seq_path, frames,frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frames,frame_id))

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        return obj_meta['object_class_name']

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_name = self.sequence_list[seq_id]
        # frames_visible = self.labels[seq_name]["visible"]["frames"]
        frames_infrared = self.labels[seq_name]["infrared"]["frames"]
        # seq_path = os.path.join(self._get_sequence_path(seq_id), 'visible')
        seq_path_i = os.path.join(self._get_sequence_path(seq_id), 'infrared')
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        # frame_list_v = [self._get_frame(seq_path, frames_visible, f_id) for f_id in frame_ids]
        frame_list_i = [self._get_frame(seq_path_i, frames_infrared, f_id) for f_id in frame_ids]
        frame_list = frame_list_i

        anno_frames = {}
        anno_frames['bbox'] = []
        [anno_frames['bbox'].append(torch.tensor(self.labels[seq_name]['infrared'][frames_infrared[f_id]])) for f_id in frame_ids]
        # anno_frames['valid']=(anno_frames['bbox'][:, 2] > 0) & (anno_frames['bbox'][:, 3] > 0)

        # for key, value in anno.items():
        #     anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
        #     #anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids] + [value[f_id, ...].clone() for f_id in frame_ids]

        return frame_list, anno_frames, obj_meta

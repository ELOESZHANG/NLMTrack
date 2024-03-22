import os

import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
# a=[]
# for filename in os.listdir('/home/ym/work/track/RGB/Stark-main/data/Lsot_TIR'):
#     a.append(filename)
#
# print(a)
# sequence_list = ['ssdw-scc-1','2']
# cls = sequence_list[0].split('-')
# print(cls)

class Lsotb_TIRDataset(BaseDataset):

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.lsotb_tir_path
        self.sequence_list = self._get_sequence_list()
        self.clean_list = self.clean_seq_list()

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)):
            cls= self.sequence_list[i].split('_')
            clean_lst.append(cls)
        return  clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        class_name = sequence_name.split('_')[0]
        anno_path = '{}/{}/groundtruth_rect.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        #target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

        frames_path = '{}/{}/img'.format(self.base_path,  sequence_name)

        frames_list = ['{}/{:04d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]
        target_class = class_name
        return Sequence(sequence_name, frames_list, 'Lsotb_TIR', ground_truth_rect.reshape(-1, 4), object_class=target_class)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):

        sequence_list = []
        for filename in os.listdir(f'{self.base_path}'):
            sequence_list.append(filename)
        return sequence_list

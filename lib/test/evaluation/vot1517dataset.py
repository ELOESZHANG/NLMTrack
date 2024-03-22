import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os
import cv2 as cv
class VOTDataset(BaseDataset):
    """
    VOT2018 dataset

    Publication:
        The sixth Visual Object Tracking VOT2018 challenge results.
        Matej Kristan, Ales Leonardis, Jiri Matas, Michael Felsberg, Roman Pfugfelder, Luka Cehovin Zajc, Tomas Vojir,
        Goutam Bhat, Alan Lukezic et al.
        ECCV, 2018
        https://prints.vicos.si/publications/365

    Download the dataset from http://www.votchallenge.net/vot2018/dataset.html
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.vot1517_path
        self.sequence_list = self._get_sequence_list()
        self.clean_list = self.clean_seq_list()

    def clean_seq_list(self):
        clean_lst = self.sequence_list
        # for i in range(len(self.sequence_list)):
        #     #cls= self.sequence_list[i].split('_')
        #     clean_lst.append(cls)
        return clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])



    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)
        # Convert gt
        if ground_truth_rect.shape[1] > 4:
            gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
            gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

            x1 = np.amin(gt_x_all, 1).reshape(-1,1)
            y1 = np.amin(gt_y_all, 1).reshape(-1,1)
            x2 = np.amax(gt_x_all, 1).reshape(-1,1)
            y2 = np.amax(gt_y_all, 1).reshape(-1,1)

            ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)
            # image = cv.imread('{}/{}/00000001.png'.format(self.base_path, sequence_name))
            # image_BGR = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            # cv.rectangle(image_BGR, (int(x1[0]), int(y1[0])), (int(x2[0]), int(y2[0])), color=(0, 0, 255), thickness=2)
            # cv.imshow("tracking result", image_BGR)
            # cv.waitKey(100)
            # print(1)
        #target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

        frames_path = '{}/{}'.format(self.base_path,  sequence_name)

        frames_list = ['{}/{:08d}.png'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]
        return Sequence(sequence_name, frames_list, 'vot', ground_truth_rect.reshape(-1, 4), object_class=sequence_name)
        # sequence_path = sequence_name
        # nz = 8
        # ext = 'jpg'
        # start_frame = 1
        #
        # anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        # try:
        #     ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        # except:
        #     ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)
        #
        # end_frame = ground_truth_rect.shape[0]
        #
        # frames = ['{base_path}/{sequence_path}/color/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
        #           sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext)
        #           for frame_num in range(start_frame, end_frame+1)]
        #
        # # Convert gt
        # if ground_truth_rect.shape[1] > 4:
        #     gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
        #     gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]
        #
        #     x1 = np.amin(gt_x_all, 1).reshape(-1,1)
        #     y1 = np.amin(gt_y_all, 1).reshape(-1,1)
        #     x2 = np.amax(gt_x_all, 1).reshape(-1,1)
        #     y2 = np.amax(gt_y_all, 1).reshape(-1,1)
        #
        #     ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)
        # return Sequence(sequence_name, frames, 'vot', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):

        sequence_list = []
        for filename in os.listdir(f'{self.base_path}'):
            if filename != 'list.txt':
                sequence_list.append(filename)
        #sequence_list = sequence_list.remove('list.txt')
        # print(os.path.join(self.base_path, "list.txt"))
        # file = open(os.path.join(self.base_path, "list.txt"), "r")
        # while True:
        #     line = file.readline()
        #     sequence_list.append(line.strip())
        # file.close()
        return sequence_list



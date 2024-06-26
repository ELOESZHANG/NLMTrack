from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.nlmtrack_utils import sample_target, transform_image_to_crop
import cv2
from lib.utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh
from lib.models.nlmtrack import build_nlmtrack
from lib.test.tracker.nlmtrack_utils import Preprocessor
from lib.utils.box_ops import clip_box
import numpy as np


class NLMTRACK(BaseTracker):
    def __init__(self, params, dataset_name):
        super(NLMTRACK, self).__init__(params)
        network = build_nlmtrack(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.seq_format = self.cfg.DATA.SEQ_FORMAT
        self.num_template = self.cfg.TEST.NUM_TEMPLATES
        self.bins = self.cfg.MODEL.BINS
        if self.cfg.TEST.WINDOW == True: # for window penalty
            self.hanning = torch.tensor(np.hanning(self.bins)).unsqueeze(0).cuda()
            self.hanning = self.hanning
        else:
            self.hanning = None
        self.start = self.bins + 1 # start token
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.debug = params.debug
        self.frame_id = 0

        # online update settings
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS.DEFAULT
        print("Update interval is: ", self.update_intervals)
        if hasattr(self.cfg.TEST.UPDATE_THRESHOLD, DATASET_NAME):
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD[DATASET_NAME]
        else:
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD.DEFAULT
        print("Update threshold is: ", self.update_threshold)



    def initialize(self, image, info: dict):

        # get the initial templates
        z_patch_arr, _ = sample_target(image, info['init_bbox'], self.params.template_factor,
                                       output_sz=self.params.template_size)

        # cv2.imshow("tracking result",z_patch_arr)
        # cv2.waitKey(0)
        template = self.preprocessor.process(z_patch_arr)
        self.template_list = [template] * self.num_template

        # get the initial sequence i.e., [start]
        batch = template.shape[0]
        self.init_seq = (torch.ones([batch, 1]).to(template) * self.start).type(dtype=torch.int64)

        self.state = info['init_bbox']
        self.frame_id = 0

    def track(self, image, info: dict = None, gtbbox=None, seq_name=None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor = sample_target(image, self.state, self.params.search_factor,
                                                   output_sz=self.params.search_size)  # (x1, y1, w, h)
        # cv2.imshow("tracking result",x_patch_arr)
        # cv2.waitKey(0)
        search = self.preprocessor.process(x_patch_arr)
        images_list = self.template_list + [search]

        with torch.no_grad():
            xz = self.network.forward_encoder(images_list)
            neck_feat = self.network.forward_neck(xz)

        # run the decoder
        with torch.no_grad():
            out_dict = self.network.inference_decoder(xz=neck_feat,
                                                      sequence=self.init_seq,
                                                      window=self.hanning,
                                                      seq_format=self.seq_format)

        pred_boxes = out_dict['pred_boxes'].view(-1, 4)

        # if use other formats of sequence
        if self.seq_format == 'corner':
            pred_boxes = box_xyxy_to_cxcywh(pred_boxes)
        if self.seq_format == 'whxy':
            pred_boxes = pred_boxes[:, [2, 3, 0, 1]]

        pred_boxes = pred_boxes / (self.bins-1)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]

        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # update the template
        if self.num_template > 1:
            conf_score = out_dict['confidence'].sum().item() * 10 # the confidence score
            if (self.frame_id % self.update_intervals == 0) and (conf_score > self.update_threshold):
                z_patch_arr, _ = sample_target(image, self.state, self.params.template_factor,
                                               output_sz=self.params.template_size)
                template = self.preprocessor.process(z_patch_arr)
                self.template_list.append(template)
                if len(self.template_list) > self.num_template:
                    self.template_list.pop(1)

        # for debug
        if self.debug == 1:
            x1, y1, w, h = self.state
            # print(x1, y1, w, h)
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(gtbbox[0]), int(gtbbox[3])), (int(gtbbox[4]), int(gtbbox[1])), color=(0, 0, 255), thickness=2)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1+w), int(y1+h)), color=(0,255,0), thickness=2)
            cv2.putText(image_BGR, "conf_score:" + str(conf_score)[:5]+",id:"+str(self.frame_id), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            cv2.imshow(f'{seq_name}', image_BGR)
            # 键盘控制视频播放  waitKey(x)控制视频显示速度
            key = cv2.waitKey(100) & 0xFF
            if key == ord(' '):
                cv2.waitKey(0)
            
            # if key == ord('q'):
            #     break

            # cv2.waitKey(1)

        return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return NLMTRACK

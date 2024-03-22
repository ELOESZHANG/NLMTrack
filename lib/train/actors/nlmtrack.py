from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou
import torch
import numpy as np
import math
def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


def generate_sa_simdr(joints):
    '''
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    '''
    num_joints = 48
    image_size = [256, 256]
    simdr_split_ratio = 1.5625
    sigma = 6

    target_x1 = np.zeros((num_joints,
                          int(image_size[0] * simdr_split_ratio)),
                         dtype=np.float32)
    target_y1 = np.zeros((num_joints,
                          int(image_size[1] * simdr_split_ratio)),
                         dtype=np.float32)
    target_x2 = np.zeros((num_joints,
                          int(image_size[0] * simdr_split_ratio)),
                         dtype=np.float32)
    target_y2 = np.zeros((num_joints,
                          int(image_size[1] * simdr_split_ratio)),
                         dtype=np.float32)
    zero_4_begin = np.zeros((num_joints, 1), dtype=np.float32)

    tmp_size = sigma * 3

    for joint_id in range(num_joints):
        mu_x1 = joints[joint_id][0]
        mu_y1 = joints[joint_id][1]
        mu_x2 = joints[joint_id][2]
        mu_y2 = joints[joint_id][3]

        x1 = np.arange(0, int(image_size[0] * simdr_split_ratio), 1, np.float32)
        y1 = np.arange(0, int(image_size[1] * simdr_split_ratio), 1, np.float32)
        x2 = np.arange(0, int(image_size[0] * simdr_split_ratio), 1, np.float32)
        y2 = np.arange(0, int(image_size[1] * simdr_split_ratio), 1, np.float32)

        target_x1[joint_id] = (np.exp(- ((x1 - mu_x1) ** 2) / (2 * sigma ** 2))) / (
                sigma * np.sqrt(np.pi * 2))
        target_y1[joint_id] = (np.exp(- ((y1 - mu_y1) ** 2) / (2 * sigma ** 2))) / (
                sigma * np.sqrt(np.pi * 2))
        target_x2[joint_id] = (np.exp(- ((x2 - mu_x2) ** 2) / (2 * sigma ** 2))) / (
                sigma * np.sqrt(np.pi * 2))
        target_y2[joint_id] = (np.exp(- ((y2 - mu_y2) ** 2) / (2 * sigma ** 2))) / (
                sigma * np.sqrt(np.pi * 2))
    return target_x1, target_y1, target_x2, target_y2

def SIoU_loss(test1, test2, theta=4):
    eps = 1e-7
    cx_pred = (test1[:, 0] + test1[:, 2]) / 2
    cy_pred = (test1[:, 1] + test1[:, 3]) / 2
    cx_gt = (test2[:, 0] + test2[:, 2]) / 2
    cy_gt = (test2[:, 1] + test2[:, 3]) / 2

    dist = ((cx_pred - cx_gt) ** 2 + (cy_pred - cy_gt) ** 2) ** 0.5
    ch = torch.max(cy_gt, cy_pred) - torch.min(cy_gt, cy_pred)
    x = ch / (dist + eps)

    angle = 1 - 2 * torch.sin(torch.arcsin(x) - torch.pi / 4) ** 2
    # distance cost
    xmin = torch.min(test1[:, 0], test2[:, 0])
    xmax = torch.max(test1[:, 2], test2[:, 2])
    ymin = torch.min(test1[:, 1], test2[:, 1])
    ymax = torch.max(test1[:, 3], test2[:, 3])
    cw = xmax - xmin
    ch = ymax - ymin
    px = ((cx_gt - cx_pred) / (cw + eps)) ** 2
    py = ((cy_gt - cy_pred) / (ch + eps)) ** 2
    gama = 2 - angle
    dis = (1 - torch.exp(-1 * gama * px)) + (1 - torch.exp(-1 * gama * py))

    # shape cost
    w_pred = test1[:, 2] - test1[:, 0]
    h_pred = test1[:, 3] - test1[:, 1]
    w_gt = test2[:, 2] - test2[:, 0]
    h_gt = test2[:, 3] - test2[:, 1]
    ww = torch.abs(w_pred - w_gt) / (torch.max(w_pred, w_gt) + eps)
    wh = torch.abs(h_gt - h_pred) / (torch.max(h_gt, h_pred) + eps)
    omega = (1 - torch.exp(-1 * wh)) ** theta + (1 - torch.exp(-1 * ww)) ** theta

    # IoU loss
    lt = torch.max(test1[..., :2], test2[..., :2])  # [B, rows, 2]
    rb = torch.min(test1[..., 2:], test2[..., 2:])  # [B, rows, 2]

    wh = fp16_clamp(rb - lt, min=0)
    overlap = wh[..., 0] * wh[..., 1]
    area1 = (test1[..., 2] - test1[..., 0]) * (
            test1[..., 3] - test1[..., 1])
    area2 = (test2[..., 2] - test2[..., 0]) * (
            test2[..., 3] - test2[..., 1])
    iou = overlap / (area1 + area2 - overlap)

    SIoU = 1 - iou + (omega + dis) / 2
    return SIoU, iou


def ciou(pred, target, eps=1e-7):
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw ** 2 + ch ** 2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
    rho2 = left + right

    factor = 4 / math.pi ** 2
    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

    # CIoU
    cious = ious - (rho2 / c2 + v ** 2 / (1 - ious + v))
    return cious, ious


class NLMTrackActor(BaseActor):
    """ Actor for training the SeqTrack"""
    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.BINS = cfg.MODEL.BINS
        self.seq_format = cfg.DATA.SEQ_FORMAT
        self.range = 2

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'search_anno'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        outputs, target_seqs = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(outputs, target_seqs)

        return loss, status

    def forward_pass(self, data):
        n, b, _, _, _ = data['search_images'].shape   # n,b,c,h,w
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (n*b, c, h, w)
        search_list = search_img.split(b,dim=0)
        template_img = data['template_images'].view(-1, *data['template_images'].shape[2:])
        template_list = template_img.split(b,dim=0)
        feature_zx = self.net(images_list=template_list+search_list, mode='encoder') # forward the encoder


        # neck
        neck_x_feature = self.net(zx=feature_zx, mode="neck")

        bins = self.BINS # coorinate token
        start = bins + 1 # START token
        end = bins # End token
        len_embedding = bins + 2 # number of embeddings, including the coordinate tokens and the special tokens

        # box of search region
        targets = data['search_anno'].permute(1,0,2).reshape(-1, data['search_anno'].shape[2])   # x0y0wh
        targets = box_xywh_to_xyxy(targets)   # x0y0wh --> x0y0x1y1
        targets = torch.max(targets, torch.tensor([0.]).to(targets)) # Truncate out-of-range values
        targets = torch.min(targets, torch.tensor([1.]).to(targets))

        # different formats of sequence, for ablation study
        if self.seq_format != 'corner':
            targets = box_xyxy_to_cxcywh(targets)

        box = (targets * (bins - 1)).int() # discretize the coordinates

        if self.seq_format == 'whxy':
            box = box[:, [2, 3, 0, 1]]

        batch = box.shape[0]
        # inpute sequence
        input_start = torch.ones([batch, 1]).to(box) * start
        input_seqs = torch.cat([input_start, box], dim=1)
        input_seqs = input_seqs.reshape(b,n,input_seqs.shape[-1])
        input_seqs = input_seqs.flatten(1)

        # target sequence
        target_end = torch.ones([batch, 1]).to(box) * end
        target_seqs = torch.cat([box, target_end], dim=1)
        target_seqs = target_seqs.reshape(b, n, target_seqs.shape[-1])
        target_seqs = target_seqs.flatten()
        target_seqs = target_seqs.type(dtype=torch.int64)

        outputs = self.net(zx=neck_x_feature, seq=input_seqs, mode="decoder")
        # out =outputs[-1]
        outputs = outputs[-1].reshape(-1, len_embedding)

        return outputs, target_seqs

    def compute_losses(self, outputs, targets_seq, return_status=True):
        # Get loss
        ce_loss = self.objective['ce'](outputs, targets_seq)
        # weighted sum
        # loss = self.loss_weight['ce'] * ce_loss

        outputs = outputs.softmax(-1)
        outputs = outputs[:, :self.BINS]
        value, extra_seq = outputs.topk(dim=-1, k=1)
        boxes_pred = extra_seq.squeeze(-1).reshape(-1,5)[:, 0:-1]
        boxes_target = targets_seq.reshape(-1,5)[:,0:-1]
        boxes_pred = box_cxcywh_to_xyxy(boxes_pred)
        boxes_target = box_cxcywh_to_xyxy(boxes_target)
        iou2 = box_iou(boxes_pred, boxes_target)[0].mean()

        sious, iou = SIoU_loss(boxes_pred, boxes_target, 4)
        sious = sious.mean()
        siou_loss = sious
        l1_loss = self.objective['l1'](boxes_pred, boxes_target)

        loss = self.loss_weight['giou'] * siou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
            'ce'] * ce_loss


        if return_status:
            # status for log
            # status = {"Loss/total": loss.item(),
            #           "IoU": iou2.item()}
            status = {"Loss/total": loss.item(),
                      "Loss/giou": siou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": ce_loss.item(),
                      "IoU": iou2.item()}
            return loss, status
        else:
            return loss

    def forward_pass_artrack(self, data):
        n, b, _, _, _ = data['search_images'].shape   # n,b,c,h,w
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (n*b, c, h, w)
        search_list = search_img.split(b,dim=0)
        template_img = data['template_images'].view(-1, *data['template_images'].shape[2:])
        template_list = template_img.split(b,dim=0)
        feature_xz = self.net(images_list=template_list+search_list, mode='encoder') # forward the encoder

        bins = self.BINS # coorinate token
        start = bins + 1 # START token
        end = bins # End token
        len_embedding = bins + 2 # number of embeddings, including the coordinate tokens and the special tokens

        # box of search region
        targets = data['search_anno'].permute(1,0,2).reshape(-1, data['search_anno'].shape[2])   # x0y0wh
        targets = box_xywh_to_xyxy(targets)   # x0y0wh --> x0y0x1y1
        targets = torch.max(targets, torch.tensor([0.]).to(targets)) # Truncate out-of-range values
        targets = torch.min(targets, torch.tensor([1.]).to(targets))

        # different formats of sequence, for ablation study
        if self.seq_format != 'corner':
            targets = box_xyxy_to_cxcywh(targets)

        box = (targets * (bins - 1)).int() # discretize the coordinates

        if self.seq_format == 'whxy':
            box = box[:, [2, 3, 0, 1]]

        batch = box.shape[0]
        # inpute sequence
        input_start = torch.ones([batch, 1]).to(box) * start
        input_seqs = torch.cat([input_start, box], dim=1)
        input_seqs = input_seqs.reshape(b,n,input_seqs.shape[-1])
        input_seqs = input_seqs.flatten(1)

        # target sequence
        target_end = torch.ones([batch, 1]).to(box) * end
        target_seqs = torch.cat([box, target_end], dim=1)
        # target_seqs = target_seqs.reshape(b, n, target_seqs.shape[-1])
        # target_seqs = target_seqs.flatten()
        target_seqs = target_seqs.type(dtype=torch.int64)

        outputs = self.net(xz=feature_xz, seq=input_seqs, mode="decoder")

        # outputs = outputs[-1].reshape(-1, len_embedding)
        outputs = outputs[-1]

        return outputs, target_seqs   # target_seqs.shape [4,5],outputs.shape [4,5,4002]

    def compute_loss_artrack(self, outputs, targets_seq, return_status=True):
        bins = self.BINS  # 4000 ,bins = self.bins
        magic_num = (self.range - 1) * 0.5
        # seq_output = gt_dict['seq_output']
        # pred_feat = pred_dict["feat"]
        seq_output = outputs
        pred_feat = targets_seq
        if self.focal == None:
            # weight = torch.ones(bins * self.range + 2) * 1
            # weight[bins * self.range + 1] = 0.1
            # weight[bins * self.range] = 0.1
            weight = torch.ones(bins + 2) * 1
            weight[bins + 1] = 0.1
            weight[bins] = 0.1
            weight.to(pred_feat)
            self.klloss = torch.nn.KLDivLoss(reduction='none').to(pred_feat)

            self.focal = torch.nn.CrossEntropyLoss(weight=weight, size_average=True).to(pred_feat)
        # compute varfifocal loss
        # pred = pred_feat.permute(1, 0, 2).reshape(-1, bins * 2 + 2)
        # target = seq_output.reshape(-1).to(torch.int64)
        pred = pred_feat.reshape(-1, bins+2)
        target = seq_output.reshape(-1)
        varifocal_loss = self.focal(pred, target)
        # compute giou and L1 loss
        beta = 1
        pred = pred_feat[0:4, :, 0:bins] * beta
        target = seq_output[:, 0:4].to(pred_feat)

        out = pred.softmax(-1).to(pred)
        mul = torch.range((-1 * magic_num + 1 / (bins)),
                          (1 + magic_num - 1 / (bins)), 2 / (bins)).to(pred)
        ans = out * mul
        ans = ans.sum(dim=-1)
        ans = ans.permute(1, 0).to(pred)
        target = target / (bins - 1) - magic_num
        extra_seq = ans
        extra_seq = extra_seq.to(pred)
        sious, iou = SIoU_loss(extra_seq, target, 4)
        sious = sious.mean()
        siou_loss = sious
        l1_loss = self.objective['l1'](extra_seq, target)

        loss = self.loss_weight['giou'] * siou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
            'focal'] * varifocal_loss

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": siou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": varifocal_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss


    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.net.to(device)
        self.objective['ce'].to(device)
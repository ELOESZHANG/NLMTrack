class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/media/SSDPA/yanmiao/rgb/NLMTTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/media/SSDPA/yanmiao/rgb/NLMTTrack/got_tensorboard/tensorboard_NLMTRACK'  # Directory for tensorboard files.
        self.lsotb_tir_dir = '/media/SSDPA/yanmiao/LSOTB-TIR_TrainingData'  # LSOTB-TIR training dataset path
        self.lsot_tir_dir = '/media/SSDPA/yanmiao/sequences'  # LSOTB-TIR evaluation dataset path
        self.got10k_dir = '/media/HDDP/VOT/datasets/GOT-10k/train'  # GOT-10K training dataset path
        self.lasher_dir = ''



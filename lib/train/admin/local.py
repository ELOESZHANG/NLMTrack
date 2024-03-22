class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = ''    # Base directory for saving network checkpoints.
        self.tensorboard_dir = ''  # Directory for saving tensorboard files.
        self.lsotb_tir_dir = ''  # LSOTB-TIR training dataset path
        self.lsot_tir_dir = ''  # LSOTB-TIR evaluation dataset path
        self.got10k_dir = ''  # GOT-10K training dataset path
        self.lasher_dir = ''



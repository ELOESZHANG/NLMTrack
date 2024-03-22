from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.network_path = '/home/ym/work/track/RGB/VideoX-master/NLMTrack/test/networks'    # Where tracking networks are stored.
    settings.prj_dir = '/media/SSDPA/yanmiao/rgb/NLMTrack'
    settings.result_plot_path = '/media/SSDPA/yanmiao/rgb/NLMTrack/test/result_plots'
    settings.results_path = '/media/SSDPA/yanmiao/rgb/NLMTrack/results/ptb_tir_047'  # Where to store tracking results
    settings.save_dir = '/media/SSDPA/yanmiao/rgb/NLMTrack'
    settings.lsotb_tir_path = '/media/SSDPA/yanmiao/sequences'
    settings.ptb_tir_path = '/media/SSDPA/yanmiao/ptbtir'
    settings.vot1517_path = ''

    return settings


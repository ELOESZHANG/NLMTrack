from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.network_path = ''    # Where tracking networks are stored.
    settings.prj_dir = ''    # project path
    settings.result_plot_path = ''
    settings.results_path = ''  # Where to store tracking results
    settings.save_dir = ''
    settings.lsotb_tir_path = ''
    settings.ptb_tir_path = ''
    settings.vot1517_path = ''

    return settings


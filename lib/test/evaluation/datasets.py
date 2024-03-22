from collections import namedtuple
import importlib
from lib.test.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = "lib.test.evaluation.%sdataset"  # Useful abbreviations to reduce the clutter

dataset_dict = dict(
    vot1517=DatasetInfo(module=pt % "vot1517", class_name="VOTDataset", kwargs=dict()),
    lsotb_tir=DatasetInfo(module=pt % "lsotb_tir", class_name="Lsotb_TIRDataset", kwargs=dict()),
    ptb_tir=DatasetInfo(module=pt % "ptb_tir", class_name="Ptb_TIRDataset", kwargs=dict()),

)


def load_dataset(name: str):
    """ Import and load a single dataset."""
    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs)  # Call the constructor
    return dataset.get_sequence_list()


def get_dataset(*args):
    """ Get a single or set of datasets."""
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name))
    return dset
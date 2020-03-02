from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class CustomVOCDataset(XMLDataset):

    def __init__(self, classes, *args, **kwargs):
        self.CLASSES = classes
        super().__init__(*args, **kwargs)

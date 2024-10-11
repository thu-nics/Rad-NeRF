from .nerf import NeRFDataset
from .nsvf import NSVFDataset
from .colmap import ColmapDataset
from .nerfpp import NeRFPPDataset
from .rtmv import RTMVDataset
from .scannet import ScanNetDataset
from .replica import ReplicaDataset
from .nerf360v2 import NeRF360v2Dataset
from .mill19 import Mill19Dataset
from .eyeful import EyefulDataset


dataset_dict = {'nerf': NeRFDataset,
                'nsvf': NSVFDataset,
                'colmap': ColmapDataset,
                'nerfpp': NeRFPPDataset,
                'rtmv': RTMVDataset,
                'scannet': ScanNetDataset,
                'replica': ReplicaDataset,
                '360v2': NeRF360v2Dataset,
                'mill19': Mill19Dataset,
                'eyeful': EyefulDataset
                }
# Import packages
from .histogram import RGBHistogram
from .searcher import Searcher, SearchFeatures
from .remove_background import RemoveBackground
from .histogram import HistogramDescriptor
from .remove_text import RemoveText
from .iou import bb_intersection_over_union
from .texture import TextureDescriptors
from .ocr import read_text, get_text_distance, get_k_images
from .noise_red import RemoveNoise
from .text_descriptors import TextDescriptors
from .text_removal import detect_text_box, extract_text
from .feature_descriptor import DetectAndDescribe
from .extract_angles import extract_angle

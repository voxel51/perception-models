import sys

# Relative import works because package_dir maps "." to perception_models
from . import core

sys.modules['core'] = core

from .core import vision_encoder

sys.modules['core.vision_encoder'] = vision_encoder

from .core.vision_encoder import rope, config

sys.modules['core.vision_encoder.rope'] = rope
sys.modules['core.vision_encoder.config'] = config

from .core.vision_encoder import pe, transforms

sys.modules['core.vision_encoder.pe'] = pe
sys.modules['core.vision_encoder.transforms'] = transforms
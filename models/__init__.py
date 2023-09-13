from .basemodel import BaseModel
from .gcn import GCN
from .savn import SAVN
from .mjolnir_r import MJOLNIR_R
from .mjolnir_o import MJOLNIR_O
from .selfatt_test import SelfAttention_test
from .zero_nav import ZeroNavigation
from .base_clipmodel import BaselineClipModel
from .zero_object import ZeroZero
from .zero_object import ZeroGCN
from .zero_object import ZeroNoGCN

__all__ = ["BaseModel", "GCN", "SAVN", "MJOLNIR_O","MJOLNIR_R","SelfAttention_test",
           "ZeroNavigation", 'BaselineClipModel', 'ZeroZero', 'ZeroGCN', 'ZeroNoGCN']

variables = locals()

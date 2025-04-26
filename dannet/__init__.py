from . import core
from .core import Constant, Variable

from .ops import*
from . import nnet
from .gradient import gradients

from . import dtype
from . import typing
from . import utils
from .utils import convert_to_tensor

from .compiler import function, is_eager, eval

from .device import Device, default_device
current_device = Device.current_device

from . import timestat
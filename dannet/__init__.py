from dannet import device
from dannet.device import Device, current_device, default_device
from dannet.compiler import jit
import dannet.lib as lib
import dannet.core as core


from dannet.core import (
    array as array,
    broadcast_shapes as broadcast_shapes,

    broadcast_to as broadcast_to,
    flip as flip,
    transpose as transpose,
    expand_dims as expand_dims,
    squeeze as squeeze,
    slice as slice,
    real as real,
    imag as imag,
    ravel as ravel,
    reshape as reshape,

    negative as negative,
    positive as positive,
    astype as astype,
    copy as copy,
    abs as abs,
    absolute as absolute,
    square as square,
    sqrt as sqrt,
    sign as sign,
    conjugate as conjugate,
    conj as conj,

    sin as sin,
    cos as cos,
    tan as tan,
    sinh as sinh,
    cosh as cosh,
    tanh as tanh,

    arcsin as arcsin,
    arccos as arccos,
    arctan as arctan,
    arcsinh as arcsinh,
    arccosh as arccosh,
    arctanh as arctanh,

    exp as exp,
    exp2 as exp2,
    exp10 as exp10,
    expm1 as expm1,
    log as log,
    log2 as log2,
    log10 as log10,
    log1p as log1p,

    deg2rad as deg2rad,
    rad2deg as rad2deg,
    radians as radians,
    degrees as degrees,
    angle as angle,


    add as add,
    subtract as subtract,
    multiply as multiply,
    divide as divide,
    arctan2 as arctan2,

    equal as equal,
    not_equal as not_equal,
    less as less,
    less_equal as less_equal,
    greater as greater,
    greater_equal as greater_equal,

    where as where,

    sum as sum,
    mean as mean,
    prod as prod,
    min as min,
    max as max
)


from dannet.array_creation import (
    empty as empty,
    zeros as zeros,
    ones as ones,
    full as full,
    empty_like as empty_like,
    zeros_like as zeros_like,
    ones_like as ones_like,
    full_like as full_like,
    eye as eye,
)

from dannet import linalg as linalg

from dannet.tensor_contractions import (
    outer as outer,
    matmul as matmul
)

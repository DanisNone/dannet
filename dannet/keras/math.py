import dannet as dt
from dannet.keras.core import convert_to_tensor

def rsqrt(x):
    x = convert_to_tensor(x)
    return dt.rsqrt(x)
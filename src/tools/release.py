import numpy as np

def xor_buffers(aa,bb,dtype=np.uint8):
    b = np.fromstring(bb, dtype=dtype)
    np.bitwise_xor(np.frombuffer(aa,dtype=dtype), b, b)
    return b.tostring()

def apply_key(string):
    key = np.array([237,  38, 109,  60, 156,  36,  42,   2,  91,  43, 216,  78, 193,
                179, 152, 131, 177,  99,  78, 155, 252, 215, 209,  57, 138, 128,
                102,  82, 242,  20,  62, 176,  76,   9,  16, 208, 188, 221,   9,
                136, 155,  56, 124,  72, 213,  87,  69,  16,   9, 253, 112,  92,
                6, 160,  46, 128,  12,  95, 176, 172, 115, 255, 116,  30, 147,
                159,  58, 252,  12, 151,  44,  78, 136,  48, 208, 157, 248, 157,
                68,  50,   8,  98,  68, 174, 222, 242,  73, 150, 246, 117, 130,
                101,  65, 198, 190, 180,  98,  48, 104, 100, 133,  44, 167,  17,
                96,  72,  34, 165,  20,   8, 114,  41,  76,  17, 158, 201, 162,
                53, 181, 204,  29, 255,  19, 164, 229, 171, 157, 226], dtype=np.uint8)
    key = np.getbuffer(key)
    key = ''.join(key)
    nkey = len(key)
    nstring = len(string)
    nchunk = nstring / nkey
    newkey = key*nchunk
    newstr = xor_buffers(string[:nchunk*nkey],newkey,dtype=np.uint64)
    if nstring%nkey:
        newstr += xor_buffers(string[nchunk*nkey:],key[:nstring%nkey],dtype=np.uint8)
    return newstr
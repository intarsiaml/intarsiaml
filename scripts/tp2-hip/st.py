import json, struct
import numpy as np
DT={'F32':np.float32,'I32':np.int32,'I64':np.int64,'U32':np.uint32,'F16':np.float16,'U8':np.uint8,'I8':np.int8,'BF16':None}
def load(p):
    b=open(p,'rb').read(); n=struct.unpack('<Q',b[:8])[0]; h=json.loads(b[8:8+n]); data=b[8+n:]
    out={}
    for k,v in h.items():
        if k=='__metadata__': continue
        s,e=v['data_offsets']; dt=DT[v['dtype']]
        out[k]=np.frombuffer(data[s:e],dtype=dt).reshape(v['shape'])
    return out, h.get('__metadata__',{})

import os,sys,glob
for pat in sys.argv[1:]:
  for p in sorted(glob.glob(pat)):
    d=open(p,'rb').read(); h=0xcbf29ce484222325
    for b in d: h=((h^b)*0x100000001b3)&0xFFFFFFFFFFFFFFFF
    print(format(h,'016x'),len(d),p)

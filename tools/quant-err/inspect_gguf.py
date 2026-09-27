#!/usr/bin/env python3
# Minimal pure-python GGUF header inspector (no numpy).
import struct, sys

GGUF_TYPES = {0:"F32",1:"F16",2:"I8",3:"I16",4:"I32",5:"I64",6:"F64",
              7:"Q4_0",8:"Q4_1",9:"Q5_0",10:"Q5_1",11:"Q8_0",12:"Q8_1",
              13:"Q2_K",14:"Q3_K",15:"Q4_K",16:"Q5_K",17:"Q6_K",18:"Q2_O",
              19:"Q3_O",20:"IQ2_XXS",21:"IQ2_XS",22:"IQ2_S",23:"IQ3_XXS",
              24:"IQ1_S",25:"IQ4_NL",26:"IQ3_S",27:"IQ2_M",28:"IQ4_XS",
              29:"IQ1_M",30:"BF16",43:"Q4_0_64"}

def rd_u8(f):  return struct.unpack("<B", f.read(1))[0]
def rd_u16(f): return struct.unpack("<H", f.read(2))[0]
def rd_u32(f): return struct.unpack("<I", f.read(4))[0]
def rd_u64(f): return struct.unpack("<Q", f.read(8))[0]
def rd_str(f):
    n = rd_u64(f)
    return f.read(n).decode("utf-8", "replace")

def main(path):
    f = open(path, "rb")
    magic = f.read(4)
    assert magic == b"GGUF", f"bad magic {magic!r}"
    version = rd_u32(f)
    n_tensors = rd_u64(f)
    n_kv = rd_u64(f)
    print(f"version={version} n_tensors={n_tensors} n_kv={n_kv}")

    # KV pairs (skip values, but capture a few interesting keys)
    for _ in range(n_kv):
        key = rd_str(f)
        vtype = rd_u32(f)
        # skip value based on type
        if vtype in (0,1,2,3,4,5,6):  # scalars
            f.read({0:4,1:2,2:1,3:2,4:4,5:8,6:8}[vtype])
        elif vtype == 7:  # string
            rd_str(f)
        elif vtype == 8:  # array
            at = rd_u32(f); an = rd_u64(f)
            for _ in range(an):
                if at in (0,1,2,3,4,5,6):
                    f.read({0:4,1:2,2:1,3:2,4:4,5:8,6:8}[at])
                elif at == 7:
                    rd_str(f)
        if key in ("general.architecture","general.name","general.file_type"):
            pass
    # tensor info
    print("\nname                                        type        ndims  dims")
    for i in range(n_tensors):
        name = rd_str(f)
        ndims = rd_u32(f)
        dims = [rd_u64(f) for _ in range(ndims)]
        ttype = rd_u32(f)
        offset = rd_u64(f)
        tname = GGUF_TYPES.get(ttype, f"UNK({ttype})")
        if i < 14 or ttype == 43:
            print(f"{name:45s} {tname:10s} {ndims}      {dims}")
    alignment = rd_u64(f)
    print(f"\nalignment={alignment}")

if __name__ == "__main__":
    main(sys.argv[1])

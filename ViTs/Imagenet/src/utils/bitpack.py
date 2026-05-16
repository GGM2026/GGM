# src/utils/bitpack.py

import torch

# def pack_kbit(tensor, k):
#     assert k in (1, 2, 4, 8), "pack_kbit only supports k in {1,2,4,8}"

#     flat = tensor.detach().cpu().flatten().to(torch.int64)

#     vals_per_byte = 8 // k
#     pad = (-flat.numel()) % vals_per_byte
#     if pad:
#         flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])

#     flat = flat.view(-1, vals_per_byte)
#     packed = torch.zeros(flat.size(0), dtype=torch.uint8)

#     for i in range(vals_per_byte):
#         packed |= (flat[:, i] << (i * k))

#     return packed


# def unpack_kbit(packed, k, shape):
#     assert k in (1, 2, 4, 8), "unpack_kbit only supports k in {1,2,4,8}"
#     vals_per_byte = 8 // k
#     total = shape[0] * shape[1]
#     out = []

#     for byte in packed:
#         for i in range(vals_per_byte):
#             out.append((byte >> (i * k)) & ((1 << k) - 1))
#             if len(out) == total:
#                 return torch.tensor(out).view(shape)
def pack_kbit(tensor, k):
    """
    Pack unsigned integer values (0 .. 2^k - 1) into a byte stream.
    Works for ANY integer k >= 1.
    """
    flat = tensor.detach().cpu().flatten().to(torch.int64)

    bit_buffer = 0
    bit_count = 0
    out = []

    for v in flat:
        bit_buffer |= int(v) << bit_count
        bit_count += k

        while bit_count >= 8:
            out.append(bit_buffer & 0xFF)
            bit_buffer >>= 8
            bit_count -= 8

    # flush remaining bits (pad with zeros)
    if bit_count > 0:
        out.append(bit_buffer & 0xFF)

    return torch.tensor(out, dtype=torch.uint8)

def unpack_kbit(packed, k, shape):
    """
    Unpack a byte stream produced by pack_kbit back into
    unsigned integer values (0 .. 2^k - 1).
    """
    total = shape[0] * shape[1]

    bit_buffer = 0
    bit_count = 0
    out = []

    for byte in packed:
        bit_buffer |= int(byte) << bit_count
        bit_count += 8

        while bit_count >= k and len(out) < total:
            out.append(bit_buffer & ((1 << k) - 1))
            bit_buffer >>= k
            bit_count -= k

        if len(out) == total:
            break

    return torch.tensor(out, dtype=torch.int64).view(shape)


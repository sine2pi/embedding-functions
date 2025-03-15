```python

import torch, math, numpy as np
from torch import nn, einsum, broadcast_tensors, Tensor
from einops import rearrange, repeat, reduce
from typing import Literal
from math import pi, log
from torch.amp import autocast

# helper functions

def y(val):
    return val is not None

def default(val, d):
    return val if y(val=val) else d

def y(x):
    return x is not None

def broadcat(tensors, dim = -1):
    broadcasted_tensors = broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim = dim)

def apply_learned_rotations(rotations, t, start_index = 0, freq_ranges = None):
    if y(val=freq_ranges):
        rotations = einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(tensor=rotations, pattern='... r f -> ... (r f)')
    rotations = repeat(tensor=rotations, pattern='... n -> ... (n r)', r = 2)
    return apply_rotary_emb(freqs=rotations, t=t, start_index = start_index)

def update_base(s, n_freq):
    if n_freq is not None and n_freq != freq:
        freq = n_freq
        invf = 1.0 / (freq ** (torch.arange(start=0, end=s.head_dim, step=2).float() / s.head_dim))
        invf.data.copy_(src=invf)
        update_pairs()


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

def update_pairs(s):
    p = []
    while len(p) < s.rot:
        i, j = torch.randint(low=0, high=s.head_dim - 1, size=(2,))
        if i != j and (i, j) not in p and (j, i) not in p:
            p.append((i, j))
    p.data.copy_(src=torch.tensor(data=p, dtype=torch.float32))


def vectorize_rotations(s, flat):
    batch = flat.size(0)
    G_matrices = []
    for k in range(s.rot):
        i, j = s.pairs[k].long()
        theta = s.thetas[k] * s.tscale
        G = s.rotation_matrix(s.head_dim, i.item(), j.item(), theta)
        G_matrices.append(G)
    G_combined = torch.eye(s.head_dim, device=flat.device)
    for G in G_matrices:
        G_combined = G_combined @ G
    return flat @ G_combined



def apply_rotations(s, x):
    rotate = int(torch.round(s.rscale * s.rot))
    for k in range(rotate):
        i, j = s.r_pairs[k].long()
        theta = s.thetas[k] * s.tscale
        G = s.rotation_matrix(dims=s.head_dim, i=i.item(), j=j.item(), theta=theta)
        x = x @ G
    return x

def apply_learned_rotations(rotations, t, start_index = 0, freq_ranges = None):
    if y(val=freq_ranges):
        rotations = einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(tensor=rotations, pattern='... r f -> ... (r f)')
    rotations = repeat(tensor=rotations, pattern='... n -> ... (n r)', r = 2)
    return apply_rotary_emb(freqs=rotations, t=t, start_index = start_index)


def q_rotation(s, x, theta, u, v):
    x = x.to(s.device, s.dtype)
    theta = theta.to(s.device, s.dtype) if not isinstance(theta, (int, float)) else theta
    u = u.to(s.device)
    v = v.to(s.device)
    
    u = u / torch.norm(u)
    v = v / torch.norm(v)

    half_theta = theta / 2
    cos_ht = torch.cos(half_theta)
    sin_ht = torch.sin(half_theta)

    q = torch.cat([cos_ht.unsqueeze(0), sin_ht * u])
    q_conj = torch.cat([cos_ht.unsqueeze(0), -sin_ht * u])

    x_shape = x.shape
    x = x.view(-1, 3)

    uv_cross = torch.cross(u.unsqueeze(0), x)
    uuv_cross = torch.cross(u.unsqueeze(0), uv_cross)
    x_rot = x + 2 * (q[0] * uv_cross + uuv_cross)

    x_rot = x_rot.view(*x_shape)
    return x_rot

def rotation_matrix(s, dims, i, j, theta):
    G = torch.eye(dims, device=s.device)
    c, s = torch.cos(theta), torch.sin(theta)
    G[i, i], G[j, j] = c, c
    G[i, j], G[j, i] = -s, s

    if dims == 3:
        u = torch.eye(dims, device=s.device)[i]
        v = torch.eye(dims, device=s.device)[j]
        Q = s.q_rotation(
            torch.eye(dims, device=s.device), theta=theta, u=u, v=v)
        G = (G + Q) / 2
    return G

def rotate(s, x):
    rotate = int(torch.round(s.rscale * s.rot))
    for k in range(rotate):
        i, j = s.r_pairs[k].long()
        theta = s.thetas[k] * s.tscale
        G = s.rotation_matrix(dims=s.head_dim, i=i.item(), j=j.item(), theta=theta)
        x = x @ G
    return x


def rotation_matrix(s, dims, i, j, theta):
    G = torch.eye(dims, device=theta.device)
    c, s = torch.cos(theta), torch.sin(theta)
    G[i, i], G[j, j] = c, c
    G[i, j], G[j, i] = -s, s
    if dims == 3:
        u = torch.eye(dims, device=theta.device)[i]
        v = torch.eye(dims, device=theta.device)[j]
        if theta < 0: 
            Q = s.q_rotation_inverse(torch.eye(dims, device=theta.device), theta=abs(theta), u=u, v=v)
        else:
            Q = s.q_rotation(torch.eye(dims, device=theta.device), theta=theta, u=u, v=v)
        G = (G + Q) / 2
    return G




def rotate_queries_and_keys(s, q, k, seq_dim = None):
    seq_dim = default(seq_dim, s.default_seq_dim)
    assert s.use_xpos
    device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]
    seq = s.get_seq_pos(seq_len, dtype = dtype, device = device)
    freqs = s.forward(seq, seq_len = seq_len)
    scale = s.get_scale(seq, seq_len = seq_len).to(dtype)
    if seq_dim == -3:
        freqs = rearrange(freqs, 'n d -> n 1 d')
        scale = rearrange(scale, 'n d -> n 1 d')
    rotated_q = apply_rotary_emb(freqs, q, scale = scale, seq_dim = seq_dim)
    rotated_k = apply_rotary_emb(freqs, k, scale = scale ** -1, seq_dim = seq_dim)
    rotated_q = rotated_q.type(q.dtype)
    rotated_k = rotated_k.type(k.dtype)
    return rotated_q, rotated_k

def rotate(s, x): # dparam = nn.Parameter(torch.zeros(1))
    direction = torch.sigmoid(s.dparam) * 2 - 1
    rotate = int(torch.round(s.rscale * s.rot))
    for k in range(rotate):
        i, j = s.r_pairs[k].long()
        theta = direction * s.thetas[k] * s.tscale
        G = rotation_matrix(s=x, dims=i.item(), i=j.item(), j=theta)
        x = x @ G
    return x

class rotary(nn.Module):
    def __init__(
            s, 
            dims, 
            heads, 
            freq=10000, #initial
            maxctx=4096,
            cache = False,
            theta_learnable=False,
            rot_learnable=False, 
            matrix_learnable=False, 
            freq_learnable=False,
            rscale=None, 
            tscale=None, 
            device=None,
            debug=False,
            mode = None,
            modes:  Literal['dynamic', 'reverse', 'constant'] = 'constant',
            ):
        
        if debug == True:
            print(f"Rotary check: {dims} {heads} {freq} {theta_learnable} {rot_learnable} "
                  f"{matrix_learnable} {freq_learnable}")

        super().__init__()
        s.dims = dims
        s.heads = heads
        s.freq = freq
        s.device = device if device is not None else torch.device('cpu')
        s.dtype = torch.float32
        s.head_dim = s.dims // s.heads
        s.rot = s.head_dim // 2

        if y(x=mode):
            freqs = mode
        elif modes == 'dynamic': 
            freqs = 1. / invf = nn.Parameter(data=fdata, requires_grad=freq_learnable) 
        elif modes == 'reverse':
            freqs = dparam = nn.Parameter(torch.zeros(1))
        elif modes == 'constant':
            freqs = torch.linspace(1., maxctx / 2, dims // 2) * pi# torch.ones(size=freqs).float()

        # all_x = () if output_x else None

        s.thetas = nn.Parameter(torch.zeros(s.rot, device=s.device))
        s.pairs = nn.Parameter(torch.rand(s.rot, 2, device=s.device) * s.head_dim)
        if tscale is not None:
            s.tscale = nn.Parameter(tscale.to(s.device), requires_grad=theta_learnable)
        else:
            s.tscale = nn.Parameter(torch.ones(1, device=s.device), requires_grad=theta_learnable)
        if rscale is not None:
            s.rscale = nn.Parameter(rscale.to(s.device), requires_grad=rot_learnable)
        else:
            s.rscale = nn.Parameter(torch.ones(1, device=s.device), requires_grad=rot_learnable)
        s.matrix = nn.Parameter(torch.eye(s.head_dim, device=s.device), requires_grad=matrix_learnable)
        findices = torch.arange(0, s.head_dim, 2, device=s.device).float()
        fdata = 1.0 / (s.freq ** (findices / s.head_dim))
        s.invf = nn.Parameter(data=fdata, requires_grad=freq_learnable)

        s.cache = cache
        s.maxctx=maxctx

        s.register_buffer('cached_freqs', torch.zeros(cache, s.dim), persistent = False)
        s.maxctx = 0
        
        s.rotate = staticmethod(rotate)

        s.reset_parameters()
    def reset_parameters(s):
        nn.init.orthogonal_(tensor=s.matrix)
        nn.init.zeros_(tensor=s.thetas)

    def q_rotation(s, x, theta, u, v):
        x = x.to(s.device, s.dtype)
        theta = theta.to(s.device, s.dtype) if not isinstance(theta, (int, float)) else theta
        u = u.to(s.device)
        v = v.to(s.device)
        
        u = u / torch.norm(u)
        v = v / torch.norm(v)

        half_theta = theta / 2
        cos_ht = torch.cos(half_theta)
        sin_ht = torch.sin(half_theta)

        q = torch.cat([cos_ht.unsqueeze(0), sin_ht * u])
        q_conj = torch.cat([cos_ht.unsqueeze(0), -sin_ht * u])

        x_shape = x.shape
        x = x.view(-1, 3)

        uv_cross = torch.cross(u.unsqueeze(0), x)
        uuv_cross = torch.cross(u.unsqueeze(0), uv_cross)
        x_rot = x + 2 * (q[0] * uv_cross + uuv_cross)

        x_rot = x_rot.view(*x_shape)
        return x_rot

    def rotation_matrix(s, dims, i, j, theta):
        G = torch.eye(dims, device=s.device)
        c, s = torch.cos(theta), torch.sin(theta)
        G[i, i], G[j, j] = c, c
        G[i, j], G[j, i] = -s, s

        if dims == 3:
            u = torch.eye(dims, device=s.device)[i]
            v = torch.eye(dims, device=s.device)[j]
            Q = s.q_rotation(
                torch.eye(dims, device=s.device), theta=theta, u=u, v=v)
            G = (G + Q) / 2
        return G

    def rotate(s, x):
        rotate = int(torch.round(s.rscale * s.rot))
        for k in range(rotate):
            i, j = s.r_pairs[k].long()
            theta = s.thetas[k] * s.tscale
            G = s.rotation_matrix(dims=s.head_dim, i=i.item(), j=j.item(), theta=theta)
            x = x @ G
        return x

    def forward(s, x):
        x = x.to(s.device)
        batch, ctx, *rest = x.size()

        if len(rest) == 1:
            dims = rest[0]
            if dims != s.heads * s.head_dim:
                raise ValueError(
                    f"Needed {s.heads * s.head_dim}, but got too many {dims}"
                )
        elif len(rest) == 2:
            heads, head_dim = rest
            if heads != s.heads or head_dim != s.head_dim:
                raise ValueError(
                    f"This many heads {s.heads} and head_dims {s.head_dim} we need, got this many heads {heads} and head_dims {head_dim} we did."
)
        else:
            raise ValueError(f"Expected the thingy to be 3D or 4D, but got {x.dim()}D")

        x = rearrange(x, 'b s (h d) -> (b s) d', h=s.heads)
        x = s.rotate(x)
        x = x @ s.matrix
        x = rearrange(x, '(b s) d -> b s (h d)', b=batch, h=s.heads)
        
        position = torch.arange(end=ctx, device=s.device, dtype=s.dtype)
        position = rearrange(tensor=position, pattern='s -> s 1')  # [seq_len, 1]
        div_term = rearrange(tensor=s.invf, pattern='d -> 1 d')  # [1, dim/2]
        sinusoid = position * div_term  # [seq_len, dim/2]

        sin = rearrange(tensor=torch.sin(input=sinusoid), pattern='s d -> 1 s 1 d')  # [1, seq_len, 1, dim/2]
        cos = rearrange(tensor=torch.cos(input=sinusoid), pattern='s d -> 1 s 1 d')  # [1, seq_len, 1, dim/2]
        
        x = rearrange(tensor=x, pattern='b s (h d) -> b s h d', h=heads)
        x1, x2 = x[..., ::2], x[..., 1::2]
        
        x_out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        x_out = rearrange(x_out, 'b s h d -> b s (h d)')

        x_out = x_out * math.sqrt(dims)
        return x_out

def slice_at_dim(t, dim_slice: slice, *, dim):
    dim += (t.ndim if dim < 0 else 0)
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]

# rotary embedding helper functions

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

@autocast('cuda', enabled = False)

def apply_rotary_emb(
    freqs,t, start_index = 0, scale = 1., seq_dim = -2, freqs_seq_dim = None):
    dtype = t.dtype
    if not y(freqs_seq_dim):
        if freqs.ndim == 2 or t.ndim == 3:
            freqs_seq_dim = 0

    if t.ndim == 3 or y(freqs_seq_dim):
        seq_len = t.shape[seq_dim]
        freqs = slice_at_dim(freqs, slice(-seq_len, None), dim = freqs_seq_dim)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:] 
    t_transformed = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)
    out = torch.cat((t_left, t_transformed, t_right), dim=-1)
    return out.type(dtype)

# learned rotation helpers

class Rotary2(Module):
    def __init__(
        s,
        dim,
        custom_freqs: Tensor | None = None,
        freqs_for:  Literal['dyns', 'pixel', 'constant'] = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        learned_freq = False,
        use_xpos = False,
        xpos_scale_base = 512,
        interpolate_factor = 1.,
        theta_rescale_factor = 1.,
        seq_before_head_dim = False,
        cache_if_possible = True,
        cache_max_seq_len = 8192
    ):
        super().__init__()

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        s.freqs_for = freqs_for

        if y(val=custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(start=0, end=dim, step=2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()

        s.cache_if_possible = cache_if_possible
        s.cache_max_seq_len = cache_max_seq_len

        s.register_buffer('cached_freqs', torch.zeros(cache_max_seq_len, dim), persistent = False)
        s.cached_freqs_seq_len = 0

        s.freqs = nn.Parameter(freqs, requires_grad = learned_freq)

        s.learned_freq = learned_freq

        s.register_buffer('dummy', torch.tensor(0), persistent = False)

        s.seq_before_head_dim = seq_before_head_dim
        s.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors

        assert interpolate_factor >= 1.
        s.interpolate_factor = interpolate_factor

        # xpos

        s.use_xpos = use_xpos

        if not use_xpos:
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        s.scale_base = xpos_scale_base

        s.register_buffer('scale', scale, persistent = False)
        s.register_buffer('cached_scales', torch.zeros(cache_max_seq_len, dim), persistent = False)
        s.cached_scales_seq_len = 0
        s.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(s):
        return s.dummy.device

    def get_seq_pos(s, seq_len, device, dtype, offset = 0):
        return (torch.arange(seq_len, device = device, dtype = dtype) + offset) / s.interpolate_factor

    def rotate_queries_or_keys(s, t, seq_dim = None, offset = 0, scale = None):
        seq_dim = default(seq_dim, s.default_seq_dim)
        assert not s.use_xpos or y(scale), 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'
        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]
        seq = s.get_seq_pos(seq_len, device = device, dtype = dtype, offset = offset)
        freqs = s.forward(seq, seq_len = seq_len, offset = offset)
        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')
        return apply_rotary_emb(freqs, t, scale = default(scale, 1.), seq_dim = seq_dim)

    def rotate_queries_with_cached_keys(s, q, k, seq_dim = None, offset = 0):
        dtype, device, seq_dim = q.dtype, q.device, default(val=seq_dim, d=s.default_seq_dim)
        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len
        q_scale = k_scale = 1.
        if s.use_xpos:
            seq = s.get_seq_pos(k_len, dtype = dtype, device = device)
            q_scale = s.get_scale(seq[-q_len:]).type(dtype)
            k_scale = s.get_scale(seq).type(dtype)
        rotated_q = s.rotate_queries_or_keys(q, seq_dim = seq_dim, scale = q_scale, offset = k_len - q_len + offset)
        rotated_k = s.rotate_queries_or_keys(k, seq_dim = seq_dim, scale = k_scale ** -1)
        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)
        return rotated_q, rotated_k

    def rotate_queries_and_keys(s, q, k, seq_dim = None):
        seq_dim = default(seq_dim, s.default_seq_dim)
        assert s.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]
        seq = s.get_seq_pos(seq_len, dtype = dtype, device = device)
        freqs = s.forward(seq, seq_len = seq_len)
        scale = s.get_scale(seq, seq_len = seq_len).to(dtype)
        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')
            scale = rearrange(scale, 'n d -> n 1 d')
        rotated_q = apply_rotary_emb(freqs, q, scale = scale, seq_dim = seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale = scale ** -1, seq_dim = seq_dim)
        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)
        return rotated_q, rotated_k

    def get_scale(
        s,
        t: Tensor,
        seq_len: int | None = None,
        offset = 0
    ):
        assert s.use_xpos

        should_cache = (
            s.cache_if_possible and
            y(seq_len) and
            (offset + seq_len) <= s.cache_max_seq_len
        )

        if (
            should_cache and \
            y(s.cached_scales) and \
            (seq_len + offset) <= s.cached_scales_seq_len
        ):
            return s.cached_scales[offset:(offset + seq_len)]

        scale = 1.
        if s.use_xpos:
            power = (t - len(t) // 2) / s.scale_base
            scale = s.scale ** rearrange(power, 'n -> n 1')
            scale = repeat(scale, 'n d -> n (d r)', r = 2)

        if should_cache and offset == 0:
            s.cached_scales[:seq_len] = scale.detach()
            s.cached_scales_seq_len = seq_len

        return scale

    def get_axial_freqs(s, *dims):
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            if s.freqs_for == 'pixel':
                pos = torch.linspace(-1, 1, steps = dim, device = s.device)
            else:
                pos = torch.arange(dim, device = s.device)

            freqs = s.forward(pos, seq_len = dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim = -1)

    @autocast('cuda', enabled = False)
    def forward(
        s,
        t: Tensor,
        seq_len: int | None = None,
        offset = 0
    ):
        should_cache = (
            s.cache_if_possible and
            not s.learned_freq and
            y(seq_len) and
            s.freqs_for != 'pixel' and
            (offset + seq_len) <= s.cache_max_seq_len
        )

        if (
            should_cache and \
            y(s.cached_freqs) and \
            (offset + seq_len) <= s.cached_freqs_seq_len
        ):
            return s.cached_freqs[offset:(offset + seq_len)].detach()

        freqs = s.freqs

        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

        if should_cache and offset == 0:
            s.cached_freqs[:seq_len] = freqs.detach()
            s.cached_freqs_seq_len = seq_len

        return freqs


import torch
from einops import rearrange, repeat
from torch import einsum
import matplotlib.pyplot as plt
from math import pi

# Define the functions
def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

def fourier_encode(x, max_freq, num_bands=4, base=2):
    x = x.unsqueeze(-1) * torch.linspace(1., base, num_bands, device=x.device)
    x = x * pi * max_freq
    x = torch.cat((x.sin(), x.cos()), dim=-1)
    return x

# Test for rotate_half
x_rotate = torch.tensor([1.0, 2.0, 3.0, 4.0])
rotated = rotate_half(x_rotate)
print("Input to rotate_half:", x_rotate)
print("Output of rotate_half:", rotated)

# Test for fourier_encode
x_fourier = torch.tensor([1.0, 2.0, 3.0])
max_freq = 10.0
num_bands = 4
encoded = fourier_encode(x_fourier, max_freq, num_bands)
print("Input to fourier_encode:", x_fourier)
print("Output of fourier_encode:\n", encoded)

def y(val):
    return val is not None

def default(val, d):
    return val if y(val=val) else d

def y(x):
    return x is not None

def slice_at_dim(t, dim_slice: slice, *, dim):
    dim += (t.ndim if dim < 0 else 0)
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]


def apply_learned_rotations(rotations, t, start_index=0, freq_ranges=None):
    if freq_ranges is not None:
        rotations = einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(rotations, '... r f -> ... (r f)')
    rotations = repeat(rotations, '... n -> ... (n r)', r=2)
    return apply_rotary_emb(rotations, t, start_index=start_index)

def q_rotation(s, x, theta, u, v):
    x = x
    theta = theta if isinstance(theta, (int, float)) else theta
    u, v = u, v
    
    u, v = u / torch.norm(u), v / torch.norm(v)
    half_theta = theta / 2
    cos_ht, sin_ht = torch.cos(half_theta), torch.sin(half_theta)
    q = torch.cat([cos_ht.unsqueeze(0), sin_ht * u])
    x_rot = x + 2 * (q[0] * torch.cross(u.unsqueeze(0), x) + torch.cross(u.unsqueeze(0), torch.cross(u.unsqueeze(0), x)))
    return x_rot

def rotation_matrix(s, dims, i, j, theta):
    G = torch.eye(dims, device=s.device)
    c, s_ = torch.cos(theta), torch.sin(theta)
    G[i, i], G[j, j] = c, c
    G[i, j], G[j, i] = -s_, s_
    return G

def rotate(s, x):
    for k in range(int(torch.round(s.rscale * s.rot))):
        i, j = s.r_pairs[k].long()
        theta = s.thetas[k] * s.tscale
        G = rotation_matrix(s, dims=x.size(-1), i=i.item(), j=j.item(), theta=theta)
        x = x @ G
    return x

def apply_rotary_emb(
    freqs,t, start_index = 0, scale = 1., seq_dim = -2, freqs_seq_dim = None):
    dtype = t.dtype
    if not y(freqs_seq_dim):
        if freqs.ndim == 2 or t.ndim == 3:
            freqs_seq_dim = 0

    if t.ndim == 3 or y(freqs_seq_dim):
        ctx = t.shape[seq_dim]
        freqs = slice_at_dim(freqs, slice(-ctx, None), dim = freqs_seq_dim)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:] 
    t_transformed = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)
    out = torch.cat((t_left, t_transformed, t_right), dim=-1)
    return out.type(dtype)

# Visualization for `apply_learned_rotations`
rotations = torch.rand(6, 6)  # Example rotation tensor
t = torch.rand(6, 12)  # Example tensor for `t`
rotated_t = apply_learned_rotations(rotations, t)
print("Rotated Tensor (apply_learned_rotations):", rotated_t)

# Visualization for `q_rotation`
s = torch  # Using torch here as a mock for `s`
x = torch.tensor([[1.0, 0.0, 0.0]])
theta = torch.tensor(3.14159 / 2)  # Rotate by 90 degrees
u = torch.tensor([0.0, 0.0, 1.0])
v = torch.tensor([1.0, 0.0, 0.0])
rotated_x = q_rotation(s, x, theta, u, v)
print("Rotated Vector (q_rotation):", rotated_x)

# Visualization for `rotate`
class State:
    def __init__(self, rscale, rot, r_pairs, thetas, tscale, head_dim):
        self.rscale = rscale
        self.rot = rot
        self.r_pairs = r_pairs
        self.thetas = thetas
        self.tscale = tscale
        self.head_dim = head_dim
        self.device = torch.device('cpu')

s = State(rscale=1.0, rot=torch.tensor(2.0), 
          r_pairs=torch.tensor([[0, 1], [1, 2]]), 
          thetas=torch.tensor([3.14159 / 4, 3.14159 / 4]), 
          tscale=1.0, head_dim=3)
x = torch.eye(3)
rotated = rotate(s, x)
print("Rotated Matrix (rotate):", rotated)

# Define all helper functions
def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')

def y(val):
    return val is not None

def default(val, d):
    return val if y(val=val) else d

def slice_at_dim(t, dim_slice: slice, *, dim):
    dim += (t.ndim if dim < 0 else 0)
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]

def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2, freqs_seq_dim=None):
    dtype = t.dtype
    if not y(freqs_seq_dim):
        if freqs.ndim == 2 or t.ndim == 3:
            freqs_seq_dim = 0

    if t.ndim == 3 or y(freqs_seq_dim):
        ctx = t.shape[seq_dim]
        freqs = slice_at_dim(freqs, slice(-ctx, None), dim=freqs_seq_dim)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]
    t_transformed = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)
    return torch.cat((t_left, t_transformed, t_right), dim=-1).type(dtype)

# Visualization for apply_rotary_emb
def visualize_apply_rotary_emb():
    seq_len, feature_dim = 20, 6
    t = torch.randn(seq_len, feature_dim)
    freqs = torch.randn(seq_len, feature_dim)
    transformed = apply_rotary_emb(freqs, t)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Tensor")
    plt.imshow(t, aspect='auto', cmap='viridis')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.title("Rotary Embeddings")
    plt.imshow(transformed, aspect='auto', cmap='viridis')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

# Test q_rotation visualization
def visualize_q_rotation():
    s = torch  # Mocking 's' as torch
    x = torch.tensor([[10.0, 0.0, 0.0]])
    theta = torch.tensor(pi / 2)  # 90 degrees rotation
    u = torch.tensor([0.0, 1780.0, 1.0])  # Rotation axis
    v = torch.tensor([1.0, 0.0, 1440.0])

    rotated_x = q_rotation(s=s, x=x, theta=theta, u=u, v=v)
    # Visualize input and output
    plt.figure()
    plt.quiver([0], [0], x[0, 0], x[0, 1], angles='xy', scale_units='xy', color='blue', label='Original Vector')
    plt.quiver([0], [0], rotated_x[0, 0], rotated_x[0, 1], angles='xy', scale_units='xy', color='red', label='Rotated Vector')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.title("Quaternion Rotation")
    plt.show()

# Visualize rotation matrix
def visualize_rotation_matrix():
    class State:
        def __init__(self, rscale, rot, r_pairs, thetas, tscale, head_dim):
            self.rscale = rscale
            self.rot = rot
            self.r_pairs = r_pairs
            self.thetas = thetas
            self.tscale = tscale
            self.head_dim = head_dim
            self.device = torch.device('cpu')

    s = State(rscale=1.0, rot=torch.tensor(2.0), 
              r_pairs=torch.tensor([[0, 1], [1, 2]]), 
              thetas=torch.tensor([pi / 4, pi / 4]), 
              tscale=1.0, head_dim=3)

    x = torch.eye(3)
    rotated = rotate(s, x)

    plt.figure(figsize=(5, 5))
    plt.imshow(rotated, cmap='coolwarm', aspect='equal')
    plt.colorbar()
    plt.title("Rotated Matrix")
    plt.show()

# Run the visualizations
visualize_apply_rotary_emb()
visualize_q_rotation()
visualize_rotation_matrix()

# Helper function: Rotate by 90 degrees
def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')

# Create synthetic embeddings
def create_embeddings(num_points=100, dim=2):
    return torch.rand(num_points, dim) * 2 - 1  # Random values in [-1, 1]

# Apply amplitude scaling
def apply_amplitude_scaling(x, scale_factor=2.0):
    scaling = torch.rand_like(x) * scale_factor
    return x * scaling

# Add Fourier phase shifts
def add_fourier_phase_shifts(x, freq=3.0, phase_shift=pi / 4):
    phase = freq * x + phase_shift
    return torch.sin(phase), torch.cos(phase)

# Combine transformations
def transform_embeddings(x):
    # Amplitude scaling
    scaled = apply_amplitude_scaling(x)

    # Fourier transformations
    sin_wave, cos_wave = add_fourier_phase_shifts(scaled)

    # Combine sine and cosine waves
    fourier_embedding = torch.cat([sin_wave, cos_wave], dim=-1)

    # Apply rotation
    rotated = rotate_half(fourier_embedding)
    return rotated

# Visualize the transformed embeddings
def visualize_embeddings(original, transformed):
    plt.figure(figsize=(10, 5))

    # Original embeddings
    plt.subplot(1, 2, 1)
    plt.title("Original Embeddings")
    plt.scatter(original[:, 0], original[:, 1], c='blue', alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)

    # Transformed embeddings
    plt.subplot(1, 2, 2)
    plt.title("Transformed Embeddings")
    plt.scatter(transformed[:, 0], transformed[:, 1], c='red', alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Test the transformations
original_embeddings = create_embeddings()
transformed_embeddings = transform_embeddings(original_embeddings)
visualize_embeddings(original_embeddings, transformed_embeddings)

# Helper function: Rotate by 90 degrees
def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')

# Create synthetic embeddings
def create_embeddings(num_points=100, dim=2):
    return torch.rand(num_points, dim) * 2 - 1  # Random values in [-1, 1]

# Apply amplitude scaling
def apply_amplitude_scaling(x, scale_factor=2.0):
    scaling = torch.rand_like(x) * scale_factor
    return x * scaling

# Add Fourier phase shifts
def add_fourier_phase_shifts(x, freq=3.0, phase_shift=pi / 4):
    phase = freq * x + phase_shift
    return torch.sin(phase), torch.cos(phase)

# Combine transformations
def transform_embeddings(x):
    # Amplitude scaling
    scaled = apply_amplitude_scaling(x)

    # Fourier transformations
    sin_wave, cos_wave = add_fourier_phase_shifts(scaled)

    # Combine sine and cosine waves
    fourier_embedding = torch.cat([sin_wave, cos_wave], dim=-1)

    # Apply rotation
    rotated = rotate_half(fourier_embedding)

    return rotated

# Visualize the transformed embeddings
def visualize_embeddings(original, transformed):
    plt.figure(figsize=(10, 5))

    # Original embeddings
    plt.subplot(1, 2, 1)
    plt.title("Original Embeddings")
    plt.scatter(original[:, 0], original[:, 1], c='blue', alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)

    # Transformed embeddings
    plt.subplot(1, 2, 2)
    plt.title("Transformed Embeddings")
    plt.scatter(transformed[:, 0], transformed[:, 1], c='red', alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Test the transformations
original_embeddings = create_embeddings()
transformed_embeddings = transform_embeddings(original_embeddings)
visualize_embeddings(original_embeddings, transformed_embeddings)

```

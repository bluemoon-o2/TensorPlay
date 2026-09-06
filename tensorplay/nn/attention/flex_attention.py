"""Block-sparse attention metadata and eager flex attention."""

from __future__ import annotations

import math
from typing import Any, Callable, NamedTuple

import tensorplay
from tensorplay import functional as F

Tensor = tensorplay.Tensor
MaskMod = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
ScoreMod = Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]

_DEFAULT_SPARSE_BLOCK_SIZE = 128
_LARGE_SPARSE_BLOCK_SIZE = 1 << 30

__all__ = [
    "AuxOutput",
    "AuxRequest",
    "BlockMask",
    "and_masks",
    "create_block_mask",
    "create_mask",
    "flex_attention",
    "noop_mask",
    "or_masks",
]


class AuxRequest:
    def __init__(self, *, lse: bool = False, max_scores: bool = False) -> None:
        self.lse = bool(lse)
        self.max_scores = bool(max_scores)


class AuxOutput(NamedTuple):
    lse: Tensor | None
    max_scores: Tensor | None


def noop_mask(
    batch: Tensor, head: Tensor, token_q: Tensor, token_kv: Tensor
) -> Tensor:
    del head, token_q, token_kv
    return batch.new_ones((), dtype=tensorplay.bool)


def _sliced_mask_mod_error(
    batch: Tensor, head: Tensor, token_q: Tensor, token_kv: Tensor
) -> Tensor:
    del batch, head, token_q, token_kv
    raise RuntimeError("Cannot use the mask function from a sliced BlockMask")


def _cdiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _ordered_to_dense(
    num_blocks_in_row: Tensor, col_indices: Tensor, num_cols: int
) -> Tensor:
    num_rows = int(col_indices.shape[-2])
    max_entries = int(col_indices.shape[-1])
    batch_dims = tuple(int(size) for size in num_blocks_in_row.shape[:-1])
    output_shape = (*batch_dims, num_rows, max_entries)
    dense = tensorplay.zeros(
        (*batch_dims, num_rows, num_cols + 1),
        dtype=tensorplay.int32,
        device=col_indices.device,
    )
    grids = []
    for axis, size in enumerate(batch_dims):
        shape = [1] * (len(batch_dims) + 2)
        shape[axis] = size
        grids.append(
            tensorplay.arange(size, dtype=tensorplay.int64, device=col_indices.device)
            .reshape(shape)
            .expand(output_shape)
        )
    row_shape = [1] * (len(batch_dims) + 2)
    row_shape[-2] = num_rows
    rows = (
        tensorplay.arange(num_rows, dtype=tensorplay.int64, device=col_indices.device)
        .reshape(row_shape)
        .expand(output_shape)
    )
    col_shape = [1] * (len(batch_dims) + 2)
    col_shape[-1] = max_entries
    columns = (
        tensorplay.arange(max_entries, dtype=tensorplay.int64, device=col_indices.device)
        .reshape(col_shape)
        .expand(output_shape)
    )
    valid = columns < num_blocks_in_row.unsqueeze(-1)
    indices = tensorplay.where(valid, col_indices, num_cols)
    dense[tuple((*grids, rows, indices))] = 1
    return dense[..., :num_cols].contiguous()


def _dense_to_ordered(dense_mask: Tensor) -> tuple[Tensor, Tensor]:
    dense_mask = dense_mask.to(dtype=tensorplay.int32)
    num_blocks = dense_mask.sum(dim=-1).to(dtype=tensorplay.int32)
    indices = tensorplay.argsort(dense_mask, dim=-1, descending=True).to(
        dtype=tensorplay.int32
    )
    return num_blocks.contiguous(), indices.contiguous()


def _transpose_ordered(
    num_blocks_in_row: Tensor, col_indices: Tensor, num_cols: int | None = None
) -> tuple[Tensor, Tensor]:
    dense = _ordered_to_dense(
        num_blocks_in_row,
        col_indices,
        int(col_indices.shape[-1]) if num_cols is None else num_cols,
    )
    return _dense_to_ordered(dense.transpose(-2, -1))


def _adjust_num_blocks_and_indices(
    num_blocks: Tensor,
    indices: Tensor,
    new_num_rows: int,
    new_num_cols: int,
) -> tuple[Tensor, Tensor]:
    indices = indices[..., :new_num_rows, :new_num_cols]
    num_blocks = num_blocks[..., :new_num_rows]
    valid = tensorplay.arange(
        int(indices.shape[-1]), dtype=num_blocks.dtype, device=indices.device
    ) < num_blocks.unsqueeze(-1)
    num_blocks = (valid & (indices < new_num_cols)).sum(dim=-1).to(
        dtype=tensorplay.int32
    )
    return num_blocks, indices


class BlockMask:
    _TENSOR_ATTRS = (
        "kv_num_blocks",
        "kv_indices",
        "full_kv_num_blocks",
        "full_kv_indices",
        "q_num_blocks",
        "q_indices",
        "full_q_num_blocks",
        "full_q_indices",
        "dq_write_order",
        "dq_write_order_full",
        "dq_kv_order",
    )

    def __init__(
        self,
        seq_lengths: tuple[int, int],
        kv_num_blocks: Tensor,
        kv_indices: Tensor,
        full_kv_num_blocks: Tensor | None,
        full_kv_indices: Tensor | None,
        q_num_blocks: Tensor | None,
        q_indices: Tensor | None,
        full_q_num_blocks: Tensor | None,
        full_q_indices: Tensor | None,
        BLOCK_SIZE: tuple[int, int] = (
            _DEFAULT_SPARSE_BLOCK_SIZE,
            _DEFAULT_SPARSE_BLOCK_SIZE,
        ),
        mask_mod: MaskMod = noop_mask,
        *,
        dq_write_order: Tensor | None = None,
        dq_write_order_full: Tensor | None = None,
        dq_kv_order: Tensor | None = None,
        dq_kv_order_spt: bool | None = None,
    ) -> None:
        if kv_indices.dim() < 2:
            raise RuntimeError("BlockMask must have at least 2 dimensions")
        if (full_kv_num_blocks is None) != (full_kv_indices is None):
            raise AssertionError("full key/value metadata must be paired")
        if (full_q_num_blocks is None) != (full_q_indices is None):
            raise AssertionError("full query metadata must be paired")
        if dq_write_order_full is not None and dq_write_order is None:
            raise ValueError("dq_write_order_full requires dq_write_order")
        self.seq_lengths = tuple(int(size) for size in seq_lengths)
        self.kv_num_blocks = kv_num_blocks
        self.kv_indices = kv_indices
        self.full_kv_num_blocks = full_kv_num_blocks
        self.full_kv_indices = full_kv_indices
        self.q_num_blocks = q_num_blocks
        self.q_indices = q_indices
        self.full_q_num_blocks = full_q_num_blocks
        self.full_q_indices = full_q_indices
        self.dq_write_order = dq_write_order
        self.dq_write_order_full = dq_write_order_full
        self.dq_kv_order = dq_kv_order
        self.dq_kv_order_spt = dq_kv_order_spt
        self.BLOCK_SIZE = tuple(int(size) for size in BLOCK_SIZE)
        self.mask_mod = mask_mod

    @classmethod
    def from_kv_blocks(
        cls,
        kv_num_blocks: Tensor,
        kv_indices: Tensor,
        full_kv_num_blocks: Tensor | None = None,
        full_kv_indices: Tensor | None = None,
        BLOCK_SIZE: int | tuple[int, int] = _DEFAULT_SPARSE_BLOCK_SIZE,
        mask_mod: MaskMod | None = None,
        seq_lengths: tuple[int, int] | None = None,
        compute_q_blocks: bool = True,
        *,
        dq_write_order: Tensor | None = None,
        dq_write_order_full: Tensor | None = None,
        dq_kv_order: Tensor | bool | None = None,
    ) -> "BlockMask":
        if isinstance(BLOCK_SIZE, int):
            BLOCK_SIZE = (BLOCK_SIZE, BLOCK_SIZE)
        if (full_kv_num_blocks is None) != (full_kv_indices is None):
            raise AssertionError("full key/value metadata must be paired")
        if seq_lengths is None:
            seq_lengths = (
                int(kv_indices.shape[-2]) * BLOCK_SIZE[0],
                int(kv_indices.shape[-1]) * BLOCK_SIZE[1],
            )
        if compute_q_blocks:
            kv_num_cols = _cdiv(seq_lengths[1], BLOCK_SIZE[1])
            q_num_blocks, q_indices = _transpose_ordered(
                kv_num_blocks, kv_indices, kv_num_cols
            )
            if full_kv_num_blocks is not None and full_kv_indices is not None:
                full_q_num_blocks, full_q_indices = _transpose_ordered(
                    full_kv_num_blocks, full_kv_indices, kv_num_cols
                )
            else:
                full_q_num_blocks, full_q_indices = None, None
        else:
            q_num_blocks = q_indices = None
            full_q_num_blocks = full_q_indices = None
        if dq_kv_order is not None and not isinstance(dq_kv_order, (bool, Tensor)):
            raise ValueError("dq_kv_order must be a bool, Tensor, or None")
        return cls(
            seq_lengths=seq_lengths,
            kv_num_blocks=kv_num_blocks,
            kv_indices=kv_indices,
            full_kv_num_blocks=full_kv_num_blocks,
            full_kv_indices=full_kv_indices,
            q_num_blocks=q_num_blocks,
            q_indices=q_indices,
            full_q_num_blocks=full_q_num_blocks,
            full_q_indices=full_q_indices,
            BLOCK_SIZE=BLOCK_SIZE,
            mask_mod=noop_mask if mask_mod is None else mask_mod,
            dq_write_order=dq_write_order,
            dq_write_order_full=dq_write_order_full,
            dq_kv_order=dq_kv_order if isinstance(dq_kv_order, Tensor) else None,
            dq_kv_order_spt=dq_kv_order if isinstance(dq_kv_order, bool) else None,
        )

    def as_tuple(self, flatten: bool = True) -> tuple[Any, ...]:
        sequence = self.seq_lengths if flatten else (self.seq_lengths,)
        block_size = self.BLOCK_SIZE if flatten else (self.BLOCK_SIZE,)
        return (
            *sequence,
            self.kv_num_blocks,
            self.kv_indices,
            self.full_kv_num_blocks,
            self.full_kv_indices,
            self.q_num_blocks,
            self.q_indices,
            self.full_q_num_blocks,
            self.full_q_indices,
            self.dq_write_order,
            self.dq_write_order_full,
            self.dq_kv_order,
            self.dq_kv_order_spt,
            *block_size,
            self.mask_mod,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.kv_indices.shape[:-2]) + self.seq_lengths

    def __getitem__(self, index: Any) -> "BlockMask":
        if self.dq_kv_order is not None:
            raise NotImplementedError("slicing a BlockMask with tensor order metadata")
        index = (index,) if not isinstance(index, tuple) else index
        if len(index) > 3:
            raise IndexError("BlockMask indexing supports batch, head, and query blocks")
        padded = (*index, slice(None), slice(None), slice(None))[:3]
        sizes = self.kv_num_blocks.shape[:3]
        normalized = tuple(
            slice(item + size, item + size + 1)
            if isinstance(item, int) and -size <= item < 0
            else slice(item, item + 1)
            if isinstance(item, int)
            else item
            for item, size in zip(padded, sizes)
        )
        full_num = None
        full_idx = None
        if self.full_kv_num_blocks is not None and self.full_kv_indices is not None:
            full_num = self.full_kv_num_blocks[normalized]
            full_idx = self.full_kv_indices[normalized]
        q_blocks = self.q_indices is not None
        result = type(self).from_kv_blocks(
            self.kv_num_blocks[normalized],
            self.kv_indices[normalized],
            full_num,
            full_idx,
            BLOCK_SIZE=self.BLOCK_SIZE,
            mask_mod=_sliced_mask_mod_error,
            seq_lengths=(
                int(self.kv_num_blocks[normalized].shape[-1]) * self.BLOCK_SIZE[0],
                self.seq_lengths[1],
            ),
            compute_q_blocks=q_blocks,
        )
        return result

    def to(self, device: Any) -> "BlockMask":
        values = {
            attr: (getattr(self, attr).to(device) if getattr(self, attr) is not None else None)
            for attr in self._TENSOR_ATTRS
        }
        return type(self)(
            seq_lengths=self.seq_lengths,
            BLOCK_SIZE=self.BLOCK_SIZE,
            mask_mod=self.mask_mod,
            dq_kv_order_spt=self.dq_kv_order_spt,
            **values,
        )

    def to_dense(self) -> Tensor:
        dense = _ordered_to_dense(
            self.kv_num_blocks, self.kv_indices, int(self.kv_indices.shape[-1])
        ) > 0
        if self.full_kv_num_blocks is not None and self.full_kv_indices is not None:
            dense = dense | (
                _ordered_to_dense(
                    self.full_kv_num_blocks,
                    self.full_kv_indices,
                    int(self.full_kv_indices.shape[-1]),
                )
                > 0
            )
        return dense

    def numel(self) -> int:
        result = 1
        for size in self.shape:
            result *= int(size)
        return result

    def sparsity(self) -> float:
        total = math.prod(int(size) for size in self.kv_num_blocks.shape[:-1])
        total *= _cdiv(self.seq_lengths[0], self.BLOCK_SIZE[0])
        total *= _cdiv(self.seq_lengths[1], self.BLOCK_SIZE[1])
        computed = self.kv_num_blocks.sum()
        if self.full_kv_num_blocks is not None:
            computed = computed + self.full_kv_num_blocks.sum()
        return 100.0 * (1.0 - computed.item() / total)


def _broadcast_to_dim(value: Tensor, dimension: int) -> Tensor:
    while value.dim() < dimension:
        value = value.unsqueeze(0)
    return value


def _convert_mask_to_block_mask(
    mask: Tensor,
    Q_BLOCK_SIZE: int,
    KV_BLOCK_SIZE: int,
    separate_full_blocks: bool,
) -> tuple[Tensor, Tensor | None]:
    if mask.dtype != tensorplay.bool:
        raise AssertionError(f"mask.dtype must be bool, got {mask.dtype}")
    mask = _broadcast_to_dim(mask, 4)
    B, H, original_q, original_kv = (int(size) for size in mask.shape)
    padded_q = _cdiv(original_q, Q_BLOCK_SIZE) * Q_BLOCK_SIZE
    padded_kv = _cdiv(original_kv, KV_BLOCK_SIZE) * KV_BLOCK_SIZE
    if (padded_q, padded_kv) != (original_q, original_kv):
        padded = tensorplay.zeros(
            (B, H, padded_q, padded_kv), dtype=tensorplay.bool, device=mask.device
        )
        padded[..., :original_q, :original_kv] = mask
        mask = padded
    Q, KV = (int(size) for size in mask.shape[-2:])
    mask = mask.view(
        B,
        H,
        Q // Q_BLOCK_SIZE,
        Q_BLOCK_SIZE,
        KV // KV_BLOCK_SIZE,
        KV_BLOCK_SIZE,
    ).permute(0, 1, 2, 4, 3, 5)
    block_sum = mask.to(tensorplay.int32).sum(dim=(-2, -1))
    if separate_full_blocks:
        full_size = Q_BLOCK_SIZE * KV_BLOCK_SIZE
        return (
            ((block_sum > 0) & (block_sum < full_size)).to(tensorplay.int8),
            (block_sum == full_size).to(tensorplay.int8),
        )
    return (block_sum > 0).to(tensorplay.int8), None


def _create_sparse_block_from_block_mask(
    block_mask: tuple[Tensor, Tensor | None],
    mask_mod: MaskMod | None,
    seq_lengths: tuple[int, int],
    Q_BLOCK_SIZE: int,
    KV_BLOCK_SIZE: int,
) -> BlockMask:
    partial, full = block_mask
    partial_num, partial_idx = _dense_to_ordered(partial)
    if full is None:
        full_num = full_idx = None
    else:
        full_num, full_idx = _dense_to_ordered(full)
    return BlockMask.from_kv_blocks(
        partial_num,
        partial_idx,
        full_num,
        full_idx,
        BLOCK_SIZE=(Q_BLOCK_SIZE, KV_BLOCK_SIZE),
        mask_mod=mask_mod,
        seq_lengths=seq_lengths,
    )


def create_mask(
    mod_fn: ScoreMod | MaskMod,
    B: int | None,
    H: int | None,
    Q_LEN: int,
    KV_LEN: int,
    device: Any = None,
) -> Tensor:
    B = 1 if B is None else int(B)
    H = 1 if H is None else int(H)
    device = "cpu" if device is None else device
    batch = tensorplay.arange(B, device=device).reshape(B, 1, 1, 1)
    head = tensorplay.arange(H, device=device).reshape(1, H, 1, 1)
    query = tensorplay.arange(Q_LEN, device=device).reshape(1, 1, Q_LEN, 1)
    key = tensorplay.arange(KV_LEN, device=device).reshape(1, 1, 1, KV_LEN)
    if mod_fn.__code__.co_argcount >= 5:
        scores = tensorplay.zeros((B, H, Q_LEN, KV_LEN), device=device)
        return ~F.isneginf(mod_fn(scores, batch, head, query, key))
    return mod_fn(batch, head, query, key).expand((B, H, Q_LEN, KV_LEN)).contiguous()


def create_block_mask(
    mask_mod: MaskMod,
    B: int | None,
    H: int | None,
    Q_LEN: int,
    KV_LEN: int,
    device: Any = None,
    BLOCK_SIZE: int | tuple[int, int] = _DEFAULT_SPARSE_BLOCK_SIZE,
    _compile: bool = False,
    separate_full_blocks: bool = True,
    compute_dq_write_order: bool = False,
    dq_kv_order: bool = True,
) -> BlockMask:
    del _compile, compute_dq_write_order, dq_kv_order
    if isinstance(BLOCK_SIZE, int):
        q_block = kv_block = BLOCK_SIZE
    else:
        q_block, kv_block = BLOCK_SIZE
    if q_block <= 0 or kv_block <= 0:
        raise ValueError("BLOCK_SIZE values must be positive")
    mask = create_mask(mask_mod, B, H, Q_LEN, KV_LEN, device)
    blocks = _convert_mask_to_block_mask(
        mask, q_block, kv_block, separate_full_blocks
    )
    return _create_sparse_block_from_block_mask(
        blocks, mask_mod, (Q_LEN, KV_LEN), q_block, kv_block
    )


def or_masks(*mask_mods: MaskMod) -> MaskMod:
    if not all(callable(mask) for mask in mask_mods):
        raise RuntimeError("all mask functions must be callable")

    def or_mask(batch: Tensor, head: Tensor, query: Tensor, key: Tensor) -> Tensor:
        result = batch.new_zeros((), dtype=tensorplay.bool)
        for mask in mask_mods:
            result = result | mask(batch, head, query, key)
        return result

    return or_mask


def and_masks(*mask_mods: MaskMod) -> MaskMod:
    if not all(callable(mask) for mask in mask_mods):
        raise RuntimeError("all mask functions must be callable")

    def and_mask(batch: Tensor, head: Tensor, query: Tensor, key: Tensor) -> Tensor:
        result = batch.new_ones((), dtype=tensorplay.bool)
        for mask in mask_mods:
            result = result & mask(batch, head, query, key)
        return result

    return and_mask


def _attention_mask(
    block_mask: BlockMask | None,
    batch: int,
    heads: int,
    query_length: int,
    key_length: int,
    device: Any,
) -> Tensor:
    if block_mask is None:
        return tensorplay.ones(
            (batch, heads, query_length, key_length),
            dtype=tensorplay.bool,
            device=device,
        )
    dense = block_mask.to_dense()
    if (
        block_mask.BLOCK_SIZE[0] == _LARGE_SPARSE_BLOCK_SIZE
        and block_mask.BLOCK_SIZE[1] == _LARGE_SPARSE_BLOCK_SIZE
    ):
        dense = tensorplay.ones(
            (batch, heads, query_length, key_length),
            dtype=tensorplay.bool,
            device=device,
        )
    else:
        dense = dense.repeat_interleave(block_mask.BLOCK_SIZE[0], dim=-2)
        dense = dense.repeat_interleave(block_mask.BLOCK_SIZE[1], dim=-1)
        dense = dense[..., :query_length, :key_length]
        dense = dense.expand((batch, heads, query_length, key_length))
    batch_idx = tensorplay.arange(batch, device=device).reshape(batch, 1, 1, 1)
    head_idx = tensorplay.arange(heads, device=device).reshape(1, heads, 1, 1)
    query_idx = tensorplay.arange(query_length, device=device).reshape(1, 1, query_length, 1)
    key_idx = tensorplay.arange(key_length, device=device).reshape(1, 1, 1, key_length)
    if block_mask.mask_mod is not noop_mask:
        dense = dense & block_mask.mask_mod(batch_idx, head_idx, query_idx, key_idx)
    return dense


def flex_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    score_mod: ScoreMod | None = None,
    block_mask: BlockMask | None = None,
    scale: float | None = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    kernel_options: dict[str, Any] | None = None,
    *,
    return_aux: AuxRequest | None = None,
) -> Any:
    del kernel_options
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query, key, and value must be four-dimensional")
    if enable_gqa and query.shape[1] != key.shape[1]:
        if int(query.shape[1]) % int(key.shape[1]) != 0:
            raise ValueError("query heads must be divisible by key/value heads")
        repeats = int(query.shape[1]) // int(key.shape[1])
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key/value leading dimensions are incompatible")
    scale = 1.0 / math.sqrt(int(query.shape[-1])) if scale is None else float(scale)
    scores = F.matmul(query, key.transpose(-2, -1)) * scale
    batch, heads, query_length, key_length = (int(size) for size in scores.shape)
    batch_idx = tensorplay.arange(batch, device=query.device).reshape(batch, 1, 1, 1)
    head_idx = tensorplay.arange(heads, device=query.device).reshape(1, heads, 1, 1)
    query_idx = tensorplay.arange(query_length, device=query.device).reshape(1, 1, query_length, 1)
    key_idx = tensorplay.arange(key_length, device=query.device).reshape(1, 1, 1, key_length)
    if score_mod is not None:
        scores = score_mod(scores, batch_idx, head_idx, query_idx, key_idx)
    allowed = _attention_mask(
        block_mask, batch, heads, query_length, key_length, query.device
    )
    scores = F.masked_fill(scores, ~allowed, float("-inf"))
    lse = tensorplay.logsumexp(scores, dim=-1)
    output = F.matmul(F.softmax(scores, dim=-1), value)
    if return_aux is not None:
        return output, AuxOutput(
            lse=lse if return_aux.lse else None,
            max_scores=tensorplay.amax(scores, dim=-1)
            if return_aux.max_scores
            else None,
        )
    if return_lse:
        return output, lse
    return output

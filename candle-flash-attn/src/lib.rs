mod ffi;

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle::{CpuStorage, DType, Layout, Result, Shape, Tensor};
use half::{bf16, f16};

/// K-dimension tile size of the splitkv kernels.
///
/// This MUST match the kBlockN baked into run_mha_fwd_splitkv_paged_dispatch
/// in kernels/flash_fwd_launch_template.h. If that block size changes, this
/// constant needs the same change or num_splits will be miscomputed.
///
/// Exposed for integration tests only; not part of the supported API.
#[doc(hidden)]
pub const SPLITKV_BLOCK_N: usize = 32;

/// M-dimension tile size of the splitkv kernels; see SPLITKV_BLOCK_N.
const SPLITKV_BLOCK_M: usize = 64;

/// Port of num_splits_heuristic from Tri Dao's flash_api.cpp. Picks the
/// smallest split count whose SM-occupancy efficiency reaches at least 85
/// percent of the best achievable, capped at max_splits. Returns 1 (no
/// split) when the grid already nearly fills the SMs.
///
/// Exposed so integration tests can verify the dispatcher actually enters
/// the splitkv path for a given shape (catching silent fallback to the
/// dense path if this heuristic regresses); not part of the supported API.
#[doc(hidden)]
pub fn num_splits_heuristic(
    batch_nheads_mblocks: usize,
    num_sms: usize,
    num_n_blocks: usize,
    max_splits: usize,
) -> usize {
    // If we already nearly fill the SMs, splitting adds overhead without
    // helping. Integer form of upstream's ">= 0.8 * num_SMs" check.
    if batch_nheads_mblocks.saturating_mul(5) >= num_sms.saturating_mul(4) {
        return 1;
    }
    let max_splits = max_splits.min(num_sms).min(num_n_blocks);
    if max_splits == 0 {
        return 1;
    }
    let ceildiv = |a: usize, b: usize| a.div_ceil(b);
    // Skip split counts that produce the same per-split block layout as a
    // smaller count (e.g. 12 splits across 64 blocks collapses to 11).
    let is_eligible =
        |n: usize| -> bool { n == 1 || ceildiv(num_n_blocks, n) != ceildiv(num_n_blocks, n - 1) };
    let mut max_eff = 0.0f32;
    let mut effs = Vec::with_capacity(max_splits);
    for n in 1..=max_splits {
        if !is_eligible(n) {
            effs.push(0.0);
            continue;
        }
        let n_waves = (batch_nheads_mblocks * n) as f32 / num_sms as f32;
        let eff = n_waves / n_waves.ceil();
        if eff > max_eff {
            max_eff = eff;
        }
        effs.push(eff);
    }
    for n in 1..=max_splits {
        if is_eligible(n) && effs[n - 1] >= 0.85 * max_eff {
            return n;
        }
    }
    1
}

/// (softmax_lse_accum, out_accum) buffers for the splitkv combine kernel.
type SplitKvAccum = (
    candle::cuda_backend::cudarc::driver::CudaSlice<f32>,
    candle::cuda_backend::cudarc::driver::CudaSlice<f32>,
);

/// Head-dim bucket the splitkv kernels are dispatched at, mirroring the gate
/// in kernels/flash_api.cu (run_mha_fwd). The paged/dense splitkv kernels are
/// instantiated only at 64/128/256/512.
///
/// Exposed for integration tests; not part of the supported API.
#[doc(hidden)]
pub fn splitkv_dispatch_bucket(head_size: usize) -> usize {
    if head_size <= 64 {
        64
    } else if head_size <= 128 {
        128
    } else if head_size <= 256 {
        256
    } else {
        512
    }
}

/// Whether the dense forward may split the K dimension for this head size.
///
/// Two constraints, both of which produce silent corruption if ignored:
/// - The split write path strides output rows by the compile-time dispatch
///   bucket, while the accumulator layout strides by head_size_rounded. When
///   they differ (head dims like 96/160/192/224) rows past the first in a tile
///   are written out of bounds. See toddwbucy/candle#28.
/// - Head dims above 256 can select the 2-warp splitkv config on low-smem
///   devices, but the combine kernel statically requires 128 threads
///   (see run_mha_fwd_splitkv_paged_hdim512), so those never split.
///
/// Exposed for integration tests; not part of the supported API.
#[doc(hidden)]
pub fn splitkv_allowed(head_size: usize, head_size_rounded: usize) -> bool {
    head_size <= 256 && splitkv_dispatch_bucket(head_size) == head_size_rounded
}

/// Rust port of set_params_splitkv from Tri Dao's flash_api.cpp: decides
/// whether the non-paged forward should split the K dimension across SMs
/// and, if so, allocates the two fp32 accumulator buffers the splitkv
/// combine kernel reads.
///
/// Returns (num_splits, accumulators). When num_splits == 1 the dense path
/// is taken and accumulators is None. The returned CudaSlices must outlive
/// the ffi::run_mha call (their device_ptr guards are taken at the call
/// site).
///
/// Accumulator layouts match flash_fwd_splitkv_combine_kernel:
///   softmax_lse_accum: (num_splits, batch_size, num_heads, seqlen_q)
///   out_accum: (num_splits, batch_size, num_heads, seqlen_q, head_size_rounded)
fn set_params_splitkv(
    dev: &candle::CudaDevice,
    batch_size: usize,
    num_heads: usize,
    head_size: usize,
    head_size_rounded: usize,
    seqlen_q: usize,
    seqlen_k: usize,
) -> Result<(i32, Option<SplitKvAccum>)> {
    if !splitkv_allowed(head_size, head_size_rounded) {
        return Ok((1, None));
    }
    let num_n_blocks = seqlen_k.div_ceil(SPLITKV_BLOCK_N);
    let num_m_blocks = seqlen_q.div_ceil(SPLITKV_BLOCK_M);

    // Upstream doubles num_SMs because the 128-thread splitkv kernel lets
    // each SM host two CTAs.
    let num_sm = dev
        .cuda_stream()
        .context()
        .attribute(
            candle::cuda_backend::cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
        .map_err(|e| {
            candle::Error::Msg(format!("cuDeviceGetAttribute(MULTIPROCESSOR_COUNT): {e}"))
        })? as usize;

    let num_splits = num_splits_heuristic(
        batch_size * num_heads * num_m_blocks,
        num_sm * 2,
        num_n_blocks,
        128,
    );
    if num_splits <= 1 {
        return Ok((1, None));
    }

    let lse_n = num_splits * batch_size * num_heads * seqlen_q;
    let out_n = lse_n * head_size_rounded;
    let lse_accum = unsafe { dev.alloc::<f32>(lse_n)? };
    let out_accum = unsafe { dev.alloc::<f32>(out_n)? };
    Ok((num_splits as i32, Some((lse_accum, out_accum))))
}

pub struct FlashAttn {
    pub softmax_scale: f32,
    pub alibi_slopes: Option<Tensor>,
    pub window_size_left: Option<usize>,
    pub window_size_right: Option<usize>,
    pub softcap: Option<f32>,
}

fn round_multiple(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
}

impl FlashAttn {
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        k: &candle::CudaStorage,
        k_l: &Layout,
        v: &candle::CudaStorage,
        v_l: &Layout,
        is_bf16: bool,
    ) -> Result<(candle::CudaStorage, Shape)> {
        // https://github.com/Dao-AILab/flash-attention/blob/b252072409e69c25f2b9d473cc534e49b24decd2/csrc/flash_attn/flash_api.cpp#L187
        let dev = q.device();
        let out_shape = q_l.shape().clone();
        let out_l = Layout::contiguous(&out_shape);

        let q = q.as_cuda_slice::<T>()?;
        let k = k.as_cuda_slice::<T>()?;
        let v = v.as_cuda_slice::<T>()?;
        let q = q.slice(q_l.start_offset()..);
        let k = k.slice(k_l.start_offset()..);
        let v = v.slice(v_l.start_offset()..);

        let q_stride = q_l.stride();
        let k_stride = k_l.stride();
        let v_stride = v_l.stride();
        let o_stride = out_l.stride();

        let q_rank = q_stride.len();
        let k_rank = k_stride.len();
        let v_rank = v_stride.len();
        let o_rank = o_stride.len();

        if q_rank != 4 || k_rank != 4 || v_rank != 4 {
            candle::bail!(
                "flash-attn expects input tensors of rank 4 (q: {q_rank}, k: {k_rank}, v: {v_rank}"
            )
        }
        if q_stride[q_rank - 1] != 1 {
            candle::bail!("the last dim of q must be contiguous {q_stride:?}")
        }
        if k_stride[k_rank - 1] != 1 {
            candle::bail!("the last dim of k must be contiguous {k_stride:?}")
        }
        if v_stride[v_rank - 1] != 1 {
            candle::bail!("the last dim of v must be contiguous {v_stride:?}")
        }

        let (b_sz, seqlen_q, num_heads, head_size_og) = q_l.shape().dims4()?;
        let (_b_sz, seqlen_k, num_heads_k, _head_size_og) = k_l.shape().dims4()?;
        let expected_kv = (b_sz, seqlen_k, num_heads_k, head_size_og);
        if expected_kv != k_l.shape().dims4()? {
            candle::bail!("shape mismatch q {:?} and k {:?}", q_l.shape(), k_l.shape())
        }
        if expected_kv != v_l.shape().dims4()? {
            candle::bail!("shape mismatch q {:?} and v {:?}", q_l.shape(), v_l.shape())
        }
        if head_size_og > 512 {
            candle::bail!("only supports head dimension at most 512 (got {head_size_og})")
        }
        if head_size_og % 8 != 0 {
            // TODO: Handle head sizes that are not a multiple of 8 via some padding.
            candle::bail!("only supports head sizes that are a multiple of 8 (got {head_size_og})")
        }
        if num_heads % num_heads_k != 0 {
            candle::bail!("number of k/v heads {num_heads_k} must divide number of heads in query {num_heads}")
        }

        let stream = dev.cuda_stream();
        let alibi_slopes_ptr = if let Some(alibi_slopes) = &self.alibi_slopes {
            if alibi_slopes.dtype() != DType::F32 {
                candle::bail!(
                    "DType mismatch alibi_slopes {:?}, expected {:?}",
                    alibi_slopes.dtype(),
                    DType::F32
                );
            }

            let (alibi_slopes, alibi_slopes_layout) = alibi_slopes.storage_and_layout();

            if num_heads != alibi_slopes_layout.shape().dims1()? {
                candle::bail!(
                    "shape mismatch alibi_slopes {:?}, expected {:?}",
                    alibi_slopes_layout.shape(),
                    (num_heads)
                );
            }

            let alibi_slopes = match &*alibi_slopes {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("alibi_slopes must be a cuda tensor"),
            };

            let alibi_slopes = alibi_slopes.slice(alibi_slopes_layout.start_offset()..);

            // Dropping the guard here doesn't seem very safe.
            let (ptr, _guard) = alibi_slopes.device_ptr(&stream);
            ptr as *const core::ffi::c_void
        } else {
            std::ptr::null()
        };

        // if window_size_left > self.max_seqlen_k or None => -1
        let mut window_size_left = self
            .window_size_left
            .filter(|v| v <= &seqlen_k)
            .map(|v| v as i32)
            .unwrap_or(-1);

        // if window_size_right > self.max_seqlen_k or None => -1
        let mut window_size_right = self
            .window_size_right
            .filter(|v| v <= &seqlen_k)
            .map(|v| v as i32)
            .unwrap_or(-1);

        let head_size = round_multiple(head_size_og, 8);
        let head_size_rounded = round_multiple(head_size, 32);
        let seqlen_q_rounded = round_multiple(seqlen_q, 128);
        let seqlen_k_rounded = round_multiple(seqlen_k, 128);

        let elem_count = out_shape.elem_count();
        let mut dst = unsafe { dev.alloc::<T>(elem_count)? };
        let mut softmax_lse = dev.alloc_zeros::<f32>(b_sz * num_heads * seqlen_q)?;

        // Decide between the dense kernel (num_splits == 1) and splitting the
        // K dimension across SMs (num_splits > 1), allocating the fp32
        // accumulators the combine kernel needs when splitting. Long-context
        // decode shapes (small batch * heads * m_blocks vs the SM count) are
        // where this engages; large grids keep the dense path.
        let (num_splits, mut splitkv_buffers) = set_params_splitkv(
            dev,
            b_sz,
            num_heads,
            head_size,
            head_size_rounded,
            seqlen_q,
            seqlen_k,
        )?;

        let is_bf16 = if is_bf16 { 1 } else { 0 };

        // Causal is the special case where window_size_right == 0 and window_size_left < 0.
        // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
        let is_causal = if window_size_left < 0 && window_size_right == 0 {
            1
        } else {
            0
        };
        if window_size_left < 0 && window_size_right >= 0 {
            window_size_left = seqlen_k as i32;
        }
        if window_size_left >= 0 && window_size_right < 0 {
            window_size_right = seqlen_k as i32;
        }

        unsafe {
            let (q_ptr, _guard) = q.device_ptr(&stream);
            let (k_ptr, _guard) = k.device_ptr(&stream);
            let (v_ptr, _guard) = v.device_ptr(&stream);
            let (dst_ptr, _guard) = dst.device_ptr_mut(&stream);
            let (softmax_lse_ptr, _guard) = softmax_lse.device_ptr_mut(&stream);
            // Splitkv accumulator pointers and their guards; bound here so
            // the guards live for the FFI call. Null when num_splits == 1.
            let lseaccum_ptr;
            let oaccum_ptr;
            let _splitkv_lse_guard;
            let _splitkv_out_guard;
            match &mut splitkv_buffers {
                Some((lse_acc, out_acc)) => {
                    let (l, lg) = lse_acc.device_ptr_mut(&stream);
                    let (o, og) = out_acc.device_ptr_mut(&stream);
                    lseaccum_ptr = l as *const core::ffi::c_void;
                    oaccum_ptr = o as *const core::ffi::c_void;
                    _splitkv_lse_guard = Some(lg);
                    _splitkv_out_guard = Some(og);
                }
                None => {
                    lseaccum_ptr = std::ptr::null();
                    oaccum_ptr = std::ptr::null();
                    _splitkv_lse_guard = None;
                    _splitkv_out_guard = None;
                }
            }
            ffi::run_mha(
                q_ptr as *const core::ffi::c_void,
                k_ptr as *const core::ffi::c_void,
                v_ptr as *const core::ffi::c_void,
                dst_ptr as *const core::ffi::c_void,
                softmax_lse_ptr as *const core::ffi::c_void,
                /* alibi_slopes_ptr */ alibi_slopes_ptr,
                /* cu_seqlens_q_ptr */ std::ptr::null(),
                /* cu_seqlens_k_ptr */ std::ptr::null(),
                /* q_batch_stride */ q_stride[0] as u32,
                /* k_batch_stride */ k_stride[0] as u32,
                /* v_batch_stride */ v_stride[0] as u32,
                /* o_batch_stride */ o_stride[0] as u32,
                /* alibi_slopes_batch_stride */ 0,
                /* q_row_stride   */ q_stride[q_rank - 3] as u32,
                /* k_row_stride   */ k_stride[k_rank - 3] as u32,
                /* v_row_stride   */ v_stride[v_rank - 3] as u32,
                /* o_row_stride   */ o_stride[o_rank - 3] as u32,
                /* q_head_stride  */ q_stride[q_rank - 2] as u32,
                /* k_head_stride  */ k_stride[k_rank - 2] as u32,
                /* v_head_stride  */ v_stride[v_rank - 2] as u32,
                /* o_head_stride  */ o_stride[o_rank - 2] as u32,
                /* b */ b_sz as u32,
                /* h */ num_heads as u32,
                /* h_k */ num_heads_k as u32,
                /* d */ head_size as u32,
                /* d_rounded */ head_size_rounded as u32,
                /* softmax_scale*/ self.softmax_scale,
                /* seqlen_q */ seqlen_q as u32,
                /* seqlen_k */ seqlen_k as u32,
                /* seqlen_q_rounded */ seqlen_q_rounded as u32,
                /* seqlen_k_rounded */ seqlen_k_rounded as u32,
                /* total_q */ (b_sz * seqlen_q) as u32,
                /* is_bf16 */ is_bf16,
                /* is_causal */ is_causal,
                /* upadded_lse */ 0,
                /* window_size_left */ window_size_left,
                /* window_size_right */ window_size_right,
                /* softcap */ self.softcap.unwrap_or(0f32),
                /* block_table_ptr */ std::ptr::null(),
                /* block_table_batch_stride */ 0,
                /* page_block_size */ -1,
                /* mm_prefix_ranges_ptr */ std::ptr::null(),
                /* mm_prefix_range_batch_stride */ 0,
                /* max_mm_prefix_ranges */ 0,
                /* stream_ptr */ stream.cu_stream() as *mut core::ffi::c_void,
                /* num_splits */ num_splits,
                /* softmax_lseaccum_ptr */ lseaccum_ptr,
                /* oaccum_ptr */ oaccum_ptr,
                /* force_split_kernel */ 0,
            )
        }

        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, out_shape))
    }
}

impl candle::CustomOp3 for FlashAttn {
    fn name(&self) -> &'static str {
        "flash-attn"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for flash-attn")
    }

    fn cuda_fwd(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        k: &candle::CudaStorage,
        k_l: &Layout,
        v: &candle::CudaStorage,
        v_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        match q.dtype() {
            candle::DType::F16 => self.cuda_fwd_t::<f16>(q, q_l, k, k_l, v, v_l, false),
            candle::DType::BF16 => self.cuda_fwd_t::<bf16>(q, q_l, k, k_l, v, v_l, true),
            dt => candle::bail!("flash-attn is only supported for f16/bf16 ({dt:?})"),
        }
    }
}

/// Flash-attention v2 layer.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(batch, seq_len_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
///
/// The resulting tensor has dimensions `(batch, seq_len_q, num_heads_q, head_size)`.
pub fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let window_size_left = None;
    let window_size_right = if causal { Some(0) } else { None };

    let op = FlashAttn {
        softmax_scale,
        alibi_slopes: None,
        window_size_left,
        window_size_right,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

/// Flash-attention v2 layer.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(batch, seq_len_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `window_size_left` - Limit left attention to value tokens.
/// * `window_size_right` - Limit right attention to value tokens.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
///
/// The resulting tensor has dimensions `(batch, seq_len_q, num_heads_q, head_size)`.
pub fn flash_attn_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    let op = FlashAttn {
        softmax_scale,
        alibi_slopes: None,
        window_size_left,
        window_size_right,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

/// Flash-attention v2 layer.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(batch, seq_len_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `alibi_slopes` - Alibi slopes tensor with shape `(num_heads_q)`.
///
/// The resulting tensor has dimensions `(batch, seq_len_q, num_heads_q, head_size)`.
pub fn flash_attn_alibi(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let window_size_left = None;
    let window_size_right = if causal { Some(0) } else { None };

    let op = FlashAttn {
        softmax_scale,
        alibi_slopes: Some(alibi_slopes.clone()),
        window_size_left,
        window_size_right,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

/// Flash-attention v2 layer.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(batch, seq_len_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `alibi_slopes` - Alibi slopes tensor with shape `(num_heads_q)`.
/// * `window_size_left` - Limit left attention to value tokens.
/// * `window_size_right` - Limit right attention to value tokens.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
///
/// The resulting tensor has dimensions `(batch, seq_len_q, num_heads_q, head_size)`.
pub fn flash_attn_alibi_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: &Tensor,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    let op = FlashAttn {
        softmax_scale,
        alibi_slopes: Some(alibi_slopes.clone()),
        window_size_left,
        window_size_right,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

/// Flash-attention v2 layer.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors `k` and `v` with fewer heads
/// than `q`. The number of heads in `k` and `v` must be divisible by the number of heads in `q`.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(batch, seq_len_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `alibi_slopes` - Optional alibi slopes tensor with shape `(num_heads_q)`.
/// * `softmax_scale` - Scaling factor for the softmax operation.
/// * `window_size_left` - Optional limit on left attention to value tokens.
/// * `window_size_right` - Optional limit on right attention to value tokens.
/// * `softcap` - Gemma style softcap the attention logits before the softmax.
///
/// # Causal Mask
///
/// Setting `window_size_left=None` and `window_size_right=Some(0)` applies a causal mask to the result
/// of `Q @ K^T`.
///
/// # Returns
///
/// The resulting tensor has dimensions `(batch, seq_len_q, num_heads_q, head_size)`.
pub fn flash_attn_alibi_windowed_softcap(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: Option<&Tensor>,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
    softcap: f32,
) -> Result<Tensor> {
    let op = FlashAttn {
        softmax_scale,
        alibi_slopes: alibi_slopes.cloned(),
        window_size_left,
        window_size_right,
        softcap: Some(softcap),
    };
    q.apply_op3(k, v, op)
}

struct FlashAttnVarLen {
    pub softmax_scale: f32,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub seqlens_q: Tensor,
    pub seqlens_k: Tensor,
    pub block_table: Option<Tensor>,
    pub mm_prefix_ranges: Option<Tensor>,
    pub page_block_size: Option<usize>,
    pub alibi_slopes: Option<Tensor>,
    pub window_size_left: Option<usize>,
    pub window_size_right: Option<usize>,
    pub softcap: Option<f32>,
}

impl FlashAttnVarLen {
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        k: &candle::CudaStorage,
        k_l: &Layout,
        v: &candle::CudaStorage,
        v_l: &Layout,
        is_bf16: bool,
    ) -> Result<(candle::CudaStorage, Shape)> {
        // https://github.com/Dao-AILab/flash-attention/blob/184b992dcb2a0890adaa19eb9b541c3e4f9d2a08/csrc/flash_attn/flash_api.cpp#L327
        let dev = q.device();
        let out_shape = q_l.shape().clone();
        let out_l = Layout::contiguous(&out_shape);

        let (seqlens_q, seqlens_q_layout) = self.seqlens_q.storage_and_layout();
        let seqlens_q = match &*seqlens_q {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?, // Should be i32!
            _ => candle::bail!("seqlens_q must be a cuda tensor"),
        };
        let seqlens_q = match seqlens_q_layout.contiguous_offsets() {
            Some((o1, o2)) => seqlens_q.slice(o1..o2),
            None => candle::bail!("seqlens_q has to be contiguous"),
        };

        let (seqlens_k, seqlens_k_layout) = self.seqlens_k.storage_and_layout();
        let seqlens_k = match &*seqlens_k {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?, // Should be i32!
            _ => candle::bail!("seqlens_k must be a cuda tensor"),
        };
        let seqlens_k = match seqlens_k_layout.contiguous_offsets() {
            Some((o1, o2)) => seqlens_k.slice(o1..o2),
            None => candle::bail!("seqlens_k has to be contiguous"),
        };

        let block_table = if let Some(block_table) = self.block_table.as_ref() {
            let (block_table_storage, block_table_layout) = block_table.storage_and_layout();
            match &*block_table_storage {
                candle::Storage::Cuda(_) => {}
                _ => candle::bail!("block_table must be a cuda tensor"),
            }
            let block_table_stride = block_table_layout.shape().dims2()?.1;
            if block_table_layout.stride().last().copied() != Some(1) {
                candle::bail!("block_table last dimension must be contiguous")
            }
            Some((
                block_table_storage,
                block_table_layout.start_offset(),
                block_table_stride,
            ))
        } else {
            None
        };

        let q = q.as_cuda_slice::<T>()?;
        let k = k.as_cuda_slice::<T>()?;
        let v = v.as_cuda_slice::<T>()?;
        let q = q.slice(q_l.start_offset()..);
        let k = k.slice(k_l.start_offset()..);
        let v = v.slice(v_l.start_offset()..);

        let q_stride = q_l.stride();
        let k_stride = k_l.stride();
        let v_stride = v_l.stride();
        let o_stride = out_l.stride();

        let q_rank = q_stride.len();
        let k_rank = k_stride.len();
        let v_rank = v_stride.len();
        let o_rank = o_stride.len();

        let paged = block_table.is_some();
        if q_rank != 3 || (!paged && k_rank != 3) || (!paged && v_rank != 3) {
            candle::bail!(
                "flash-attn-varlen expects input tensors of rank 3 (q: {q_rank}, k: {k_rank}, v: {v_rank}"
            )
        }
        if paged && (k_rank != 4 || v_rank != 4) {
            candle::bail!(
                "flash-attn-varlen paged expects k/v tensors of rank 4 (k: {k_rank}, v: {v_rank})"
            )
        }
        if q_stride[q_rank - 1] != 1 {
            candle::bail!("the last dim of q must be contiguous {q_stride:?}")
        }
        if k_stride[k_rank - 1] != 1 {
            candle::bail!("the last dim of k must be contiguous {k_stride:?}")
        }
        if v_stride[v_rank - 1] != 1 {
            candle::bail!("the last dim of v must be contiguous {v_stride:?}")
        }

        let (total_q, num_heads, head_size_og) = q_l.shape().dims3()?;
        let (num_heads_k, page_block_size) = if paged {
            let (_, page_block_size, num_heads_k, k_head_size) = k_l.shape().dims4()?;
            let expected_v = k_l.shape().dims4()?;
            if expected_v != v_l.shape().dims4()? {
                candle::bail!("shape mismatch k {:?} and v {:?}", k_l.shape(), v_l.shape())
            }
            if k_head_size != head_size_og {
                candle::bail!("shape mismatch q {:?} and k {:?}", q_l.shape(), k_l.shape())
            }
            let Some(page_block_size_arg) = self.page_block_size else {
                candle::bail!("paged flash-attn requires page_block_size")
            };
            if page_block_size_arg != page_block_size {
                candle::bail!(
                    "page_block_size {page_block_size_arg} does not match k shape {:?}",
                    k_l.shape()
                )
            }
            if page_block_size % 32 != 0 {
                candle::bail!(
                    "paged flash-attn requires page_block_size to be a multiple of 32 (got {page_block_size})"
                )
            }
            if head_size_og > 512 {
                candle::bail!("paged flash-attn supports head sizes up to 512 (got {head_size_og})")
            }
            (num_heads_k, page_block_size_arg)
        } else {
            let (total_k, num_heads_k, _head_size_og) = k_l.shape().dims3()?;
            let expected_kv = (total_k, num_heads_k, head_size_og);
            if expected_kv != k_l.shape().dims3()? {
                candle::bail!("shape mismatch q {:?} and k {:?}", q_l.shape(), k_l.shape())
            }
            if expected_kv != v_l.shape().dims3()? {
                candle::bail!("shape mismatch q {:?} and v {:?}", q_l.shape(), v_l.shape())
            }
            (num_heads_k, 0)
        };
        if head_size_og > 512 {
            candle::bail!("only supports head dimension at most 512 (got {head_size_og})")
        }
        if head_size_og % 8 != 0 {
            // TODO: Handle head sizes that are not a multiple of 8 via some padding.
            candle::bail!("only supports head sizes that are a multiple of 8 (got {head_size_og})")
        }
        if num_heads % num_heads_k != 0 {
            candle::bail!("number of k/v heads {num_heads_k} must divide number of heads in query {num_heads}")
        }

        let nseqlens_q = seqlens_q_layout.shape().dims1()?;
        if nseqlens_q < 2 {
            candle::bail!("seqlens_q should have a len >= 2 {nseqlens_q}")
        }
        let nseqlens_k = seqlens_k_layout.shape().dims1()?;
        if nseqlens_k != nseqlens_q {
            candle::bail!("seqlens_q and seqlens_k should have the same number of elements {nseqlens_q} <> {nseqlens_k}")
        }

        let batch_size = nseqlens_q - 1;
        let mm_prefix_ranges = if let Some(mm_prefix_ranges) = self.mm_prefix_ranges.as_ref() {
            let (storage, layout) = mm_prefix_ranges.storage_and_layout();
            if mm_prefix_ranges.dtype() != DType::I32 {
                candle::bail!(
                    "mm_prefix_ranges must be i32, got {:?}",
                    mm_prefix_ranges.dtype()
                )
            }
            match &*storage {
                candle::Storage::Cuda(_) => {}
                _ => candle::bail!("mm_prefix_ranges must be a cuda tensor"),
            }
            let (mm_batch, max_ranges, two) = layout.shape().dims3()?;
            if mm_batch != batch_size || two != 2 {
                candle::bail!(
                    "mm_prefix_ranges shape must be ({batch_size}, max_ranges, 2), got {:?}",
                    layout.shape()
                )
            }
            if layout.stride().last().copied() != Some(1) {
                candle::bail!("mm_prefix_ranges last dimension must be contiguous")
            }
            Some((
                storage,
                layout.start_offset(),
                layout.stride()[0],
                max_ranges,
            ))
        } else {
            None
        };

        let stream = dev.cuda_stream();
        let alibi_slopes_ptr = if let Some(alibi_slopes) = &self.alibi_slopes {
            if alibi_slopes.dtype() != DType::F32 {
                candle::bail!(
                    "DType mismatch alibi_slopes {:?}, expected {:?}",
                    alibi_slopes.dtype(),
                    DType::F32
                );
            }

            let (alibi_slopes, alibi_slopes_layout) = alibi_slopes.storage_and_layout();

            if num_heads != alibi_slopes_layout.shape().dims1()? {
                candle::bail!(
                    "shape mismatch alibi_slopes {:?}, expected {:?}",
                    alibi_slopes_layout.shape(),
                    (num_heads)
                );
            }

            let alibi_slopes = match &*alibi_slopes {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("alibi_slopes must be a cuda tensor"),
            };

            let alibi_slopes = alibi_slopes.slice(alibi_slopes_layout.start_offset()..);

            // Dropping the guard here doesn't seem very safe.
            let (ptr, _guard) = alibi_slopes.device_ptr(&stream);
            ptr as *const core::ffi::c_void
        } else {
            std::ptr::null()
        };

        // if window_size_left > self.max_seqlen_k or None => -1
        let mut window_size_left = self
            .window_size_left
            .filter(|v| v <= &self.max_seqlen_k)
            .map(|v| v as i32)
            .unwrap_or(-1);

        // if window_size_right > self.max_seqlen_k or None => -1
        let mut window_size_right = self
            .window_size_right
            .filter(|v| v <= &self.max_seqlen_k)
            .map(|v| v as i32)
            .unwrap_or(-1);
        if mm_prefix_ranges.is_some() && window_size_left < 0 && window_size_right == 0 {
            window_size_left = self.max_seqlen_k as i32;
        }

        let head_size = round_multiple(head_size_og, 8);
        let head_size_rounded = round_multiple(head_size, 32);
        let seqlen_q_rounded = round_multiple(self.max_seqlen_q, 128);
        let seqlen_k_rounded = round_multiple(self.max_seqlen_k, 128);

        let elem_count = out_shape.elem_count();
        let mut dst = unsafe { dev.alloc::<T>(elem_count)? };
        let mut softmax_lse = dev.alloc_zeros::<f32>(num_heads * total_q)?;

        let is_bf16 = if is_bf16 { 1 } else { 0 };

        // Causal is the special case where window_size_right == 0 and window_size_left < 0.
        // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
        let is_causal = if window_size_left < 0 && window_size_right == 0 {
            1
        } else {
            0
        };
        if window_size_left < 0 && window_size_right >= 0 {
            window_size_left = self.max_seqlen_k as i32;
        }
        if window_size_left >= 0 && window_size_right < 0 {
            window_size_right = self.max_seqlen_k as i32;
        }

        unsafe {
            let (q_ptr, _guard) = q.device_ptr(&stream);
            let (k_ptr, _guard) = k.device_ptr(&stream);
            let (v_ptr, _guard) = v.device_ptr(&stream);
            let (dst_ptr, _guard) = dst.device_ptr_mut(&stream);
            let (softmax_lse_ptr, _guard) = softmax_lse.device_ptr_mut(&stream);
            let (seqlens_q_ptr, _guard) = seqlens_q.device_ptr(&stream);
            let (seqlens_k_ptr, _guard) = seqlens_k.device_ptr(&stream);
            let (block_table_ptr, block_table_batch_stride) =
                if let Some((block_table, offset, stride)) = block_table.as_ref() {
                    match (&**block_table, self.block_table.as_ref().unwrap().dtype()) {
                        (candle::Storage::Cuda(block_table), DType::U32) => {
                            let block_table = block_table.as_cuda_slice::<u32>()?;
                            let block_table = block_table.slice(*offset..);
                            let (ptr, _guard) = block_table.device_ptr(&stream);
                            (ptr as *const i32, *stride as u32)
                        }
                        (candle::Storage::Cuda(block_table), DType::I32) => {
                            let block_table = block_table.as_cuda_slice::<i32>()?;
                            let block_table = block_table.slice(*offset..);
                            let (ptr, _guard) = block_table.device_ptr(&stream);
                            (ptr as *const i32, *stride as u32)
                        }
                        (_, dtype) => {
                            candle::bail!("block_table must be u32 or i32, got {dtype:?}")
                        }
                    }
                } else {
                    (std::ptr::null(), 0)
                };
            let (mm_prefix_ranges_ptr, mm_prefix_range_batch_stride, max_mm_prefix_ranges) =
                if let Some((storage, offset, stride, max_ranges)) = mm_prefix_ranges.as_ref() {
                    match &**storage {
                        candle::Storage::Cuda(mm_prefix_ranges) => {
                            let mm_prefix_ranges = mm_prefix_ranges.as_cuda_slice::<i32>()?;
                            let mm_prefix_ranges = mm_prefix_ranges.slice(*offset..);
                            let (ptr, _guard) = mm_prefix_ranges.device_ptr(&stream);
                            (ptr as *const i32, *stride as u32, *max_ranges as i32)
                        }
                        _ => unreachable!("mm_prefix_ranges must be a cuda tensor"),
                    }
                } else {
                    (std::ptr::null(), 0, 0)
                };
            ffi::run_mha(
                q_ptr as *const core::ffi::c_void,
                k_ptr as *const core::ffi::c_void,
                v_ptr as *const core::ffi::c_void,
                dst_ptr as *const core::ffi::c_void,
                softmax_lse_ptr as *const core::ffi::c_void,
                /* alibi_slopes_ptr */ alibi_slopes_ptr as *const core::ffi::c_void,
                /* cu_seqlens_q_ptr */ seqlens_q_ptr as *const i32,
                /* cu_seqlens_k_ptr */ seqlens_k_ptr as *const i32,
                /* q_batch_stride */ 0,
                /* k_batch_stride */ if paged { k_stride[0] as u32 } else { 0 },
                /* v_batch_stride */ if paged { v_stride[0] as u32 } else { 0 },
                /* o_batch_stride */ 0,
                /* alibi_slopes_batch_stride */ 0,
                /* q_row_stride   */ q_stride[q_rank - 3] as u32,
                /* k_row_stride   */ k_stride[k_rank - 3] as u32,
                /* v_row_stride   */ v_stride[v_rank - 3] as u32,
                /* o_row_stride   */ o_stride[o_rank - 3] as u32,
                /* q_head_stride  */ q_stride[q_rank - 2] as u32,
                /* k_head_stride  */ k_stride[k_rank - 2] as u32,
                /* v_head_stride  */ v_stride[v_rank - 2] as u32,
                /* o_head_stride  */ o_stride[o_rank - 2] as u32,
                /* b */ batch_size as u32,
                /* h */ num_heads as u32,
                /* h_k */ num_heads_k as u32,
                /* d */ head_size as u32,
                /* d_rounded */ head_size_rounded as u32,
                /* softmax_scale*/ self.softmax_scale,
                /* seqlen_q */ self.max_seqlen_q as u32,
                /* seqlen_k */ self.max_seqlen_k as u32,
                /* seqlen_q_rounded */ seqlen_q_rounded as u32,
                /* seqlen_k_rounded */ seqlen_k_rounded as u32,
                /* total_q */ total_q as u32,
                /* is_bf16 */ is_bf16,
                /* is_causal */ is_causal,
                /* upadded_lse */ 1,
                /* window_size_left */ window_size_left,
                /* window_size_right */ window_size_right,
                /* softcap */ self.softcap.unwrap_or(0.0),
                /* block_table_ptr */ block_table_ptr,
                /* block_table_batch_stride */ block_table_batch_stride,
                /* page_block_size */ page_block_size as i32,
                /* mm_prefix_ranges_ptr */ mm_prefix_ranges_ptr,
                /* mm_prefix_range_batch_stride */ mm_prefix_range_batch_stride,
                /* max_mm_prefix_ranges */ max_mm_prefix_ranges,
                /* stream_ptr */ stream.cu_stream() as *mut core::ffi::c_void,
                // The varlen path never splits the K dimension: splitting
                // interacts with the unpadded LSE accumulator layout and is
                // not exercised upstream, and paged callers already reach the
                // splitkv kernel via block_table with num_splits == 1.
                /* num_splits */
                1,
                /* softmax_lseaccum_ptr */ std::ptr::null(),
                /* oaccum_ptr */ std::ptr::null(),
                /* force_split_kernel */ 0,
            )
        }

        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, out_shape))
    }
}

impl candle::CustomOp3 for FlashAttnVarLen {
    fn name(&self) -> &'static str {
        "flash-attn-varlen"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for flash-attn")
    }

    fn cuda_fwd(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        k: &candle::CudaStorage,
        k_l: &Layout,
        v: &candle::CudaStorage,
        v_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        match q.dtype() {
            candle::DType::F16 => self.cuda_fwd_t::<f16>(q, q_l, k, k_l, v, v_l, false),
            candle::DType::BF16 => self.cuda_fwd_t::<bf16>(q, q_l, k, k_l, v, v_l, true),
            dt => candle::bail!("flash-attn is only supported for f16/bf16 ({dt:?})"),
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with variable-length batching.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
/// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
/// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
/// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
///
/// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
/// `seqlen_1 + seqlen_2`, etc.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
pub fn flash_attn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let window_size_left = None;
    let window_size_right = if causal { Some(0) } else { None };

    let op = FlashAttnVarLen {
        softmax_scale,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        block_table: None,
        mm_prefix_ranges: None,
        page_block_size: None,
        alibi_slopes: None,
        window_size_left,
        window_size_right,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with variable-length batching.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
/// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
/// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
/// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
/// * `window_size_left` - Limit left attention to value tokens.
/// * `window_size_right` - Limit right attention to value tokens.
///
/// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
/// `seqlen_1 + seqlen_2`, etc.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
pub fn flash_attn_varlen_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    let op = FlashAttnVarLen {
        softmax_scale,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        block_table: None,
        mm_prefix_ranges: None,
        page_block_size: None,
        alibi_slopes: None,
        window_size_left,
        window_size_right,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with variable-length batching and paged K/V cache.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
/// * `k` - Paged key tensor with shape `(num_blocks, page_block_size, num_heads_kv, head_size)`.
/// * `v` - Paged value tensor with shape `(num_blocks, page_block_size, num_heads_kv, head_size)`.
/// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
/// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
/// * `block_table` - Per-batch physical block indices with shape `(batch_size, max_blocks)`.
/// * `mm_prefix_ranges` - Optional per-batch ranges with shape `(batch_size, max_ranges, 2)` that bypass the causal/window mask.
/// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
/// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
/// * `window_size_left` - Limit left attention to value tokens.
/// * `window_size_right` - Limit right attention to value tokens.
/// * `page_block_size` - Number of tokens per K/V cache block.
/// * `softcap` - Optional Gemma-style softcap for attention logits before the softmax.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
pub fn flash_attn_varlen_paged_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    block_table: &Tensor,
    mm_prefix_ranges: Option<&Tensor>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
    page_block_size: usize,
    softcap: Option<f32>,
) -> Result<Tensor> {
    let op = FlashAttnVarLen {
        softmax_scale,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        block_table: Some(block_table.clone()),
        mm_prefix_ranges: mm_prefix_ranges.cloned(),
        page_block_size: Some(page_block_size),
        alibi_slopes: None,
        window_size_left,
        window_size_right,
        softcap,
    };
    q.apply_op3(k, v, op)
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with variable-length batching.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `alibi_slopes` - Alibi slopes tensor with shape `(num_heads_q)`.
/// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
/// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
/// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
/// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
///
/// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
/// `seqlen_1 + seqlen_2`, etc.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
pub fn flash_attn_varlen_alibi(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let window_size_left = None;
    let window_size_right = if causal { Some(0) } else { None };

    let op = FlashAttnVarLen {
        softmax_scale,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        block_table: None,
        mm_prefix_ranges: None,
        page_block_size: None,
        alibi_slopes: Some(alibi_slopes.clone()),
        window_size_left,
        window_size_right,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with variable-length batching.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `alibi_slopes` - Alibi slopes tensor with shape `(num_heads_q)`.
/// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
/// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
/// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
/// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
/// * `window_size_left` - Limit left attention to value tokens.
/// * `window_size_right` - Limit right attention to value tokens.
///
/// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
/// `seqlen_1 + seqlen_2`, etc.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
pub fn flash_attn_varlen_alibi_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    let op = FlashAttnVarLen {
        softmax_scale,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        block_table: None,
        mm_prefix_ranges: None,
        page_block_size: None,
        alibi_slopes: Some(alibi_slopes.clone()),
        window_size_left,
        window_size_right,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with variable-length batching.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `alibi_slopes` - Option, alibi slopes tensor with shape `(num_heads_q)`.
/// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
/// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
/// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
/// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
/// * `window_size_left` - Option, limit left attention to value tokens.
/// * `window_size_right` - Option, limit right attention to value tokens.
/// * `softcap` - Gemma style softcap the attention logits before the softmax.
///
/// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
/// `seqlen_1 + seqlen_2`, etc.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
pub fn flash_attn_varlen_alibi_windowed_softcap(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: Option<&Tensor>,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
    softcap: f32,
) -> Result<Tensor> {
    let op = FlashAttnVarLen {
        softmax_scale,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        block_table: None,
        mm_prefix_ranges: None,
        page_block_size: None,
        alibi_slopes: alibi_slopes.cloned(),
        window_size_left,
        window_size_right,
        softcap: Some(softcap),
    };
    q.apply_op3(k, v, op)
}

//! Gemma 4 quantized (GGUF) text model.
//!
//! Gemma 4 is Google's MoE text/vision/audio family; this file implements the
//! GGUF-quantized text decoder (arch `gemma4`).
//!
//! Layers interleave two attention geometries (see `layer_types` in the HF
//! config): `sliding_attention` (local) and `full_attention` (global). The
//! global layers use a larger head dim, fewer KV heads, a proportional RoPE,
//! and share value with key (`attention_k_eq_v`, no separate v_proj).
//!
//! Each layer runs a dense feedforward and (when `enable_moe_block`) a sparse
//! MoE feedforward in parallel: the dense MLP and a 128-expert/8-active router
//! both read the post-attention residual, their outputs are normalized and
//! summed, then scaled by a per-layer scalar. The MoE expert GEMM uses
//! `candle_nn::moe::moe_gemm_gguf`, which is CUDA-only.
//!
//! References:
//! - [Gemma 4](https://blog.google/technology/developers/gemma-4/)

use super::quantized_qwen3::Gguf;
use super::with_tracing::QMatMul;
use crate::quantized_nn::RmsNorm;
use crate::utils::repeat_kv;
use candle::quantized::{gguf_file, QTensor};
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{moe, Activation, Embedding, Linear};
use std::io::{Read, Seek};
use std::sync::Arc;

// Gemma 4 global (full_attention) layers use a proportional RoPE with this
// partial rotary factor; it is an architectural constant (HF config:
// rope_parameters.full_attention.partial_rotary_factor) not stored in GGUF.
const FULL_PARTIAL_ROTARY_FACTOR: f64 = 0.25;

// ── RotaryEmbedding (standard for local layers, proportional for global) ─────

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    /// Standard RoPE over the full head dim (local / sliding_attention layers).
    fn standard(
        dtype: DType,
        head_dim: usize,
        rope_theta: f64,
        max_seq_len: usize,
        dev: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        Self::from_inv_freq(inv_freq, dtype, max_seq_len, dev)
    }

    /// Proportional RoPE: only the first `partial_rotary_factor * head_dim`
    /// dimensions are rotated, the rest are identity (global / full_attention).
    fn proportional(
        dtype: DType,
        head_dim: usize,
        rope_theta: f64,
        partial_rotary_factor: f64,
        max_seq_len: usize,
        dev: &Device,
    ) -> Result<Self> {
        let rope_angles = (partial_rotary_factor * head_dim as f64 / 2.0) as usize;
        let half_dim = head_dim / 2;
        let mut inv_freq = Vec::with_capacity(half_dim);
        for i in 0..rope_angles {
            inv_freq.push(1f32 / (rope_theta as f32).powf((2 * i) as f32 / head_dim as f32));
        }
        // Identity (cos=1, sin=0) on the non-rotated dimensions.
        inv_freq.extend(std::iter::repeat_n(0f32, half_dim - rope_angles));
        Self::from_inv_freq(inv_freq, dtype, max_seq_len, dev)
    }

    fn from_inv_freq(
        inv_freq: Vec<f32>,
        dtype: DType,
        max_seq_len: usize,
        dev: &Device,
    ) -> Result<Self> {
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_b, _h, seq_len, _d) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

/// Weightless RMS normalization over the last dim (Gemma 4 normalizes V and the
/// router input with no learned weight).
fn rms_no_scale(x: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let rms = (x.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?;
    x.broadcast_div(&rms)?.to_dtype(dtype)
}

// ── Attention ───────────────────────────────────────────────────────────────

struct Attention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    // None on full_attention layers: attention_k_eq_v, value = key projection.
    v_proj: Option<QMatMul>,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    num_kv_groups: usize,
    rms_norm_eps: f64,
    rotary_emb: RotaryEmbedding,
    kv_cache: Option<(Tensor, Tensor)>,
    dtype: DType,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn load<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: RotaryEmbedding,
        dtype: DType,
    ) -> Result<Self> {
        let q_proj = gg.qmatmul(&format!("{prefix}.attn_q.weight"))?;
        let k_proj = gg.qmatmul(&format!("{prefix}.attn_k.weight"))?;
        // Global layers omit attn_v (value = key); local layers carry it.
        let v_proj = gg.qmatmul(&format!("{prefix}.attn_v.weight")).ok();
        let o_proj = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;
        let q_norm = gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_norm_eps)?;
        let k_norm = gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_norm_eps)?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            n_head,
            n_kv_head,
            head_dim,
            num_kv_groups: n_head / n_kv_head,
            rms_norm_eps,
            rotary_emb,
            kv_cache: None,
            dtype,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let in_dtype = x.dtype();

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        // attention_k_eq_v: value uses the raw key projection (pre q/k-norm, pre RoPE).
        let v = match &self.v_proj {
            Some(v_proj) => v_proj.forward(x)?,
            None => k.clone(),
        };

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Per-head Q/K RMSNorm, then weightless V norm.
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let v = rms_no_scale(&v, self.rms_norm_eps)?;

        let (q, k) = (q.to_dtype(self.dtype)?, k.to_dtype(self.dtype)?);
        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;
        let v = v.to_dtype(self.dtype)?;

        // offset == 0 starts a fresh sequence: ignore any stale cache.
        let (k, v) = match &self.kv_cache {
            Some((k_cache, v_cache)) if offset > 0 => {
                let k = Tensor::cat(&[k_cache, &k], 2)?;
                let v = Tensor::cat(&[v_cache, &v], 2)?;
                (k, v)
            }
            _ => (k, v),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        // Gemma 4 attention uses scaling = 1.0 (not 1/sqrt(head_dim)); the per-head
        // Q/K RMSNorms control the logit magnitude.
        let mut scores = q.matmul(&k.transpose(2, 3)?)?;
        if let Some(mask) = mask {
            scores = scores.broadcast_add(&mask.to_dtype(scores.dtype())?)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;

        let ctx = ctx
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.n_head * self.head_dim))?;
        self.o_proj.forward(&ctx.to_dtype(in_dtype)?)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

// ── Dense MLP (Phase A feedforward; gemma4 uses gelu_pytorch_tanh GeGLU) ──────

struct Mlp {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
}

impl Mlp {
    fn load<R: Read + Seek>(gg: &mut Gguf<R>, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate_proj: gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?,
            up_proj: gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?,
            down_proj: gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self
            .gate_proj
            .forward(x)?
            .apply(&Activation::GeluPytorchTanh)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ── Sparse MoE feedforward (router + 128 fused experts) ──────────────────────

struct Gemma4Moe {
    // Router (own weightless RMSNorm + learned per-dim scale + scalar_root_size).
    router_proj: Linear,      // ffn_gate_inp.weight  [num_experts, hidden]
    router_scale: Tensor,     // ffn_gate_inp.scale   [hidden]
    per_expert_scale: Tensor, // ffn_down_exps.scale [num_experts]
    scalar_root_size: f64,    // hidden^-0.5
    eps: f64,
    // Experts (quantized, gate/up fused into one tensor per layer).
    gate_up_exps: Arc<QTensor>, // [num_experts, 2*moe_inter, hidden]
    down_exps: Arc<QTensor>,    // [num_experts, hidden, moe_inter]
    moe_inter: usize,
    num_experts_per_tok: usize,
    act: Activation,
    dtype: DType,
}

impl Gemma4Moe {
    fn load<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        hidden_size: usize,
        moe_inter: usize,
        num_experts_per_tok: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Option<Self>> {
        // Present only on MoE (enable_moe_block) layers.
        let gate_up = match gg.tensor(&format!("{prefix}.ffn_gate_up_exps.weight")) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };
        let router_proj_w = gg
            .tensor(&format!("{prefix}.ffn_gate_inp.weight"))?
            .dequantize(device)?
            .to_dtype(DType::F32)?;
        let router_scale = gg
            .tensor(&format!("{prefix}.ffn_gate_inp.scale"))?
            .dequantize(device)?
            .to_dtype(DType::F32)?;
        let per_expert_scale = gg
            .tensor(&format!("{prefix}.ffn_down_exps.scale"))?
            .dequantize(device)?
            .to_dtype(DType::F32)?;
        let down_exps = gg.tensor(&format!("{prefix}.ffn_down_exps.weight"))?;
        Ok(Some(Self {
            router_proj: Linear::new(router_proj_w, None),
            router_scale,
            per_expert_scale,
            scalar_root_size: (hidden_size as f64).powf(-0.5),
            eps: 0.0, // set by caller via with_eps
            gate_up_exps: Arc::new(gate_up),
            down_exps: Arc::new(down_exps),
            moe_inter,
            num_experts_per_tok,
            act: Activation::GeluPytorchTanh,
            dtype,
        }))
    }

    fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// `router_in`: raw post-attention residual (flat `[tokens, hidden]`).
    /// `expert_in`: `pre_feedforward_layernorm_2(residual)` (flat `[tokens, hidden]`).
    fn forward(&self, router_in: &Tensor, expert_in: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let k = self.num_experts_per_tok;

        // Router (in f32 for stability): weightless RMSNorm, per-dim scale,
        // scalar_root_size, softmax over all experts, top-k, renormalize,
        // per-expert scale.
        let h = rms_no_scale(&router_in.to_dtype(DType::F32)?, self.eps)?;
        let h = h.broadcast_mul(&self.router_scale)?;
        let h = (h * self.scalar_root_size)?;
        let scores = self.router_proj.forward(&h)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let topk_idx = probs
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, k)?
            .contiguous()?;
        let topk_w = probs.gather(&topk_idx, D::Minus1)?;
        let topk_w = topk_w.broadcast_div(&topk_w.sum_keepdim(D::Minus1)?)?;
        let pes = self
            .per_expert_scale
            .index_select(&topk_idx.flatten_all()?, 0)?
            .reshape(topk_idx.shape())?;
        let topk_w = (topk_w * pes)?;

        // Experts: grouped quantized GEMM. moe_gemm_gguf takes F32 activations and
        // the model dtype as its compute precision (it dequantizes the experts to
        // that dtype). Fused gate_up -> chunk -> gelu(gate)*up -> down.
        let xs = expert_in.to_dtype(DType::F32)?;
        let num_tokens = xs.dim(0)?;
        let hidden = xs.dim(1)?;
        let (expert_ids, sorted_token_ids) = topk_idx.flatten_all()?.sort_last_dim(true)?;
        let gate_up = moe::moe_gemm_gguf(
            &xs,
            &self.gate_up_exps,
            &None,
            &sorted_token_ids,
            &expert_ids,
            k,
            is_prefill,
            // moe_gemm_gguf's CUDA kernel requires BF16 compute (F32 in/out).
            DType::BF16,
        )?;
        let gate = gate_up.narrow(D::Minus1, 0, self.moe_inter)?;
        let up = gate_up.narrow(D::Minus1, self.moe_inter, self.moe_inter)?;
        let down_in = (up * gate.apply(&self.act)?)?;
        let ys = moe::moe_gemm_gguf(
            &down_in,
            &self.down_exps,
            &Some(topk_w),
            &sorted_token_ids,
            &expert_ids,
            k,
            is_prefill,
            // moe_gemm_gguf's CUDA kernel requires BF16 compute (F32 in/out).
            DType::BF16,
        )?;
        // Sum the per-token top-k expert contributions, back to the model dtype.
        ys.reshape((num_tokens, (), hidden))?
            .sum(D::Minus2)?
            .to_dtype(self.dtype)
    }
}

// ── Decoder layer (attention + dense feedforward, optional sparse MoE) ────────

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    // MoE branch (None on dense layers).
    moe: Option<Gemma4Moe>,
    pre_feedforward_layernorm_2: Option<RmsNorm>,
    post_feedforward_layernorm_1: Option<RmsNorm>,
    post_feedforward_layernorm_2: Option<RmsNorm>,
    layer_scalar: Tensor,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn load<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        hidden_size: usize,
        moe_inter: usize,
        num_experts_per_tok: usize,
        rms_norm_eps: f64,
        rotary_emb: RotaryEmbedding,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let self_attn = Attention::load(
            gg,
            prefix,
            n_head,
            n_kv_head,
            head_dim,
            rms_norm_eps,
            rotary_emb,
            dtype,
        )?;
        let mlp = Mlp::load(gg, prefix)?;
        let input_layernorm = gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
        let post_attention_layernorm = gg.rms_norm(
            &format!("{prefix}.post_attention_norm.weight"),
            rms_norm_eps,
        )?;
        let pre_feedforward_layernorm =
            gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_norm_eps)?;
        let post_feedforward_layernorm =
            gg.rms_norm(&format!("{prefix}.post_ffw_norm.weight"), rms_norm_eps)?;

        let moe = Gemma4Moe::load(
            gg,
            prefix,
            hidden_size,
            moe_inter,
            num_experts_per_tok,
            dtype,
            device,
        )?
        .map(|m| m.with_eps(rms_norm_eps));
        let (
            pre_feedforward_layernorm_2,
            post_feedforward_layernorm_1,
            post_feedforward_layernorm_2,
        ) = if moe.is_some() {
            (
                Some(gg.rms_norm(&format!("{prefix}.pre_ffw_norm_2.weight"), rms_norm_eps)?),
                Some(gg.rms_norm(&format!("{prefix}.post_ffw_norm_1.weight"), rms_norm_eps)?),
                Some(gg.rms_norm(&format!("{prefix}.post_ffw_norm_2.weight"), rms_norm_eps)?),
            )
        } else {
            (None, None, None)
        };
        let layer_scalar = gg
            .tensor(&format!("{prefix}.layer_output_scale.weight"))?
            .dequantize(device)?
            .to_dtype(dtype)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            moe,
            pre_feedforward_layernorm_2,
            post_feedforward_layernorm_1,
            post_feedforward_layernorm_2,
            layer_scalar,
        })
    }

    /// Attention sub-block; returns the post-attention residual (also the Phase A
    /// parity checkpoint, computed before the feedforward block).
    fn forward_attn(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, mask, offset)?;
        let x = self.post_attention_layernorm.forward(&x)?;
        residual + x
    }

    fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let x = self.forward_attn(x, mask, offset)?;
        let residual = &x;

        // Dense feedforward branch.
        let dense = self
            .mlp
            .forward(&self.pre_feedforward_layernorm.forward(&x)?)?;

        let combined = match &self.moe {
            None => dense,
            Some(moe) => {
                // Dual feedforward: dense and MoE both read the post-attention
                // residual; the MoE router reads the raw residual, the experts
                // read pre_feedforward_layernorm_2(residual).
                let (b, seq, hidden) = x.dims3()?;
                let dense1 = self
                    .post_feedforward_layernorm_1
                    .as_ref()
                    .unwrap()
                    .forward(&dense)?;
                let router_in = x.reshape(((), hidden))?;
                let expert_in = self
                    .pre_feedforward_layernorm_2
                    .as_ref()
                    .unwrap()
                    .forward(&x)?
                    .reshape(((), hidden))?;
                let moe_out = moe
                    .forward(&router_in, &expert_in, is_prefill)?
                    .reshape((b, seq, hidden))?;
                let moe2 = self
                    .post_feedforward_layernorm_2
                    .as_ref()
                    .unwrap()
                    .forward(&moe_out)?;
                (dense1 + moe2)?
            }
        };

        let combined = self.post_feedforward_layernorm.forward(&combined)?;
        let x = (residual + combined)?;
        x.broadcast_mul(&self.layer_scalar)
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// ── Model ────────────────────────────────────────────────────────────────────

pub struct ModelWeights {
    tok_embeddings: Embedding,
    embedding_length: usize,
    layers: Vec<DecoderLayer>,
    is_sliding: Vec<bool>,
    sliding_window: usize,
    norm: RmsNorm,
    output: QMatMul,
    final_logit_softcapping: Option<f64>,
    device: Device,
    dtype: DType,
}

impl ModelWeights {
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());
        let md_u32 = |gg: &Gguf<&mut R>, k: &str| -> Result<u32> {
            match gg.metadata().get(k) {
                None => candle::bail!("missing metadata key {k}"),
                Some(v) => v.to_u32(),
            }
        };

        let arch = "gemma4";
        let n_head = md_u32(&gg, &format!("{arch}.attention.head_count"))? as usize;
        let block_count = md_u32(&gg, &format!("{arch}.block_count"))? as usize;
        let embedding_length = md_u32(&gg, &format!("{arch}.embedding_length"))? as usize;
        let key_length = md_u32(&gg, &format!("{arch}.attention.key_length"))? as usize;
        let key_length_swa = md_u32(&gg, &format!("{arch}.attention.key_length_swa"))? as usize;
        let sliding_window = md_u32(&gg, &format!("{arch}.attention.sliding_window"))? as usize;
        let context_length = md_u32(&gg, &format!("{arch}.context_length"))? as usize;
        // MoE: expert intermediate size and active experts per token.
        let moe_inter = md_u32(&gg, &format!("{arch}.expert_feed_forward_length"))? as usize;
        let num_experts_per_tok = md_u32(&gg, &format!("{arch}.expert_used_count"))? as usize;
        let rms_norm_eps = match gg
            .metadata()
            .get(&format!("{arch}.attention.layer_norm_rms_epsilon"))
        {
            Some(v) => v.to_f32()? as f64,
            None => 1e-6,
        };
        let rope_theta = match gg.metadata().get(&format!("{arch}.rope.freq_base")) {
            Some(v) => v.to_f32()? as f64,
            None => 1_000_000.,
        };
        let rope_theta_swa = match gg.metadata().get(&format!("{arch}.rope.freq_base_swa")) {
            Some(v) => v.to_f32()? as f64,
            None => 10_000.,
        };
        let final_logit_softcapping = gg
            .metadata()
            .get(&format!("{arch}.final_logit_softcapping"))
            .and_then(|v| v.to_f32().ok())
            .map(|v| v as f64);

        // Per-layer KV head counts and the sliding/full pattern come as arrays.
        let head_count_kv: Vec<usize> = gg
            .metadata()
            .get(&format!("{arch}.attention.head_count_kv"))
            .and_then(|v| v.to_vec().ok())
            .map(|vs| {
                vs.iter()
                    .filter_map(|x| x.to_u32().ok().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();
        let pattern: Vec<bool> = gg
            .metadata()
            .get(&format!("{arch}.attention.sliding_window_pattern"))
            .and_then(|v| v.to_vec().ok())
            .map(|vs| vs.iter().filter_map(|x| x.to_bool().ok()).collect())
            .unwrap_or_default();
        // is_sliding[i]: true = local/sliding, false = global/full attention.
        // Fallback to the documented every-6th-layer-is-global cadence.
        let is_sliding: Vec<bool> = (0..block_count)
            .map(|i| pattern.get(i).copied().unwrap_or((i + 1) % 6 != 0))
            .collect();

        let tok_embeddings = gg.tensor("token_embd.weight")?.dequantize(device)?;
        let tok_embeddings = Embedding::new(tok_embeddings, embedding_length);
        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        // Tied embeddings: fall back to token_embd if no explicit output.weight.
        let output = match gg.qmatmul("output.weight") {
            Ok(w) => w,
            Err(_) => gg.qmatmul("token_embd.weight")?,
        };

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let sliding = is_sliding[layer_idx];
            let head_dim = if sliding { key_length_swa } else { key_length };
            let n_kv_head =
                head_count_kv
                    .get(layer_idx)
                    .copied()
                    .unwrap_or(if sliding { 8 } else { 2 });
            let rotary_emb = if sliding {
                RotaryEmbedding::standard(dtype, head_dim, rope_theta_swa, context_length, device)?
            } else {
                RotaryEmbedding::proportional(
                    dtype,
                    head_dim,
                    rope_theta,
                    FULL_PARTIAL_ROTARY_FACTOR,
                    context_length,
                    device,
                )?
            };
            layers.push(DecoderLayer::load(
                &mut gg,
                &prefix,
                n_head,
                n_kv_head,
                head_dim,
                embedding_length,
                moe_inter,
                num_experts_per_tok,
                rms_norm_eps,
                rotary_emb,
                dtype,
                device,
            )?);
        }

        Ok(Self {
            tok_embeddings,
            embedding_length,
            layers,
            is_sliding,
            sliding_window,
            norm,
            output,
            final_logit_softcapping,
            device: device.clone(),
            dtype,
        })
    }

    /// Causal mask (additive, 0 keep / -inf masked). `window` bounds the local
    /// context for sliding layers.
    fn causal_mask(
        &self,
        seq_len: usize,
        offset: usize,
        window: Option<usize>,
        dtype: DType,
    ) -> Result<Tensor> {
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..seq_len).map(move |j| {
                    let masked = j > i || window.map(|w| i >= j + w).unwrap_or(false);
                    if masked {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (seq_len, seq_len), &self.device)?;
        let mask = if offset > 0 {
            let zeros = Tensor::zeros((seq_len, offset), DType::F32, &self.device)?;
            Tensor::cat(&[&zeros, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((1, 1, seq_len, seq_len + offset))?
            .to_dtype(dtype)
    }

    fn embed(&self, input: &Tensor) -> Result<Tensor> {
        let xs = self.tok_embeddings.forward(input)?;
        (xs * (self.embedding_length as f64).sqrt())?.to_dtype(self.dtype)
    }

    fn masks(
        &self,
        seq_len: usize,
        offset: usize,
        dtype: DType,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        if seq_len == 1 {
            return Ok((None, None));
        }
        let full = self.causal_mask(seq_len, offset, None, dtype)?;
        let sliding = self.causal_mask(seq_len, offset, Some(self.sliding_window), dtype)?;
        Ok((Some(full), Some(sliding)))
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_b, seq_len) = input.dims2()?;
        let is_prefill = seq_len > 1;
        let mut xs = self.embed(input)?;
        // Masks are additive f32 (-inf / 0); attention casts them to its compute dtype.
        let (full_mask, sliding_mask) = self.masks(seq_len, offset, DType::F32)?;
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let mask = if self.is_sliding[idx] {
                sliding_mask.as_ref()
            } else {
                full_mask.as_ref()
            };
            xs = layer.forward(&xs, mask, offset, is_prefill)?;
        }
        let xs = xs.i((.., seq_len - 1, ..))?;
        let logits = self.output.forward(&self.norm.forward(&xs)?)?;
        match self.final_logit_softcapping {
            None => Ok(logits),
            Some(sc) => (logits / sc)?.tanh()?.affine(sc, 0.0),
        }
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}

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
//! Status: Phase A - attention + dense feedforward only. The sparse MoE branch
//! (router + 128 experts + per-layer scales) is added in Phase B; until then
//! this runs the `enable_moe_block = false` dense path.
//!
//! References:
//! - [Gemma 4](https://blog.google/technology/developers/gemma-4/)

use super::quantized_qwen3::Gguf;
use super::with_tracing::QMatMul;
use crate::quantized_nn::RmsNorm;
use crate::utils::repeat_kv;
use candle::quantized::gguf_file;
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Activation, Embedding};
use std::io::{Read, Seek};

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

/// Weightless RMS normalization (Gemma 4 normalizes V with no learned weight).
fn v_norm(v: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = v.dtype();
    let v = v.to_dtype(DType::F32)?;
    let rms = (v.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?;
    v.broadcast_div(&rms)?.to_dtype(dtype)
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
        let v = v_norm(&v, self.rms_norm_eps)?;

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

// ── Decoder layer (Phase A: dense feedforward only) ──────────────────────────

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
}

impl DecoderLayer {
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
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
        })
    }

    /// Attention sub-block; returns the post-attention residual (Phase A parity
    /// checkpoint, computed before the feedforward block).
    fn forward_attn(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, mask, offset)?;
        let x = self.post_attention_layernorm.forward(&x)?;
        residual + x
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let x = self.forward_attn(x, mask, offset)?;
        // Phase A dense path == gemma4 enable_moe_block=false:
        //   post_ffw_norm(mlp(pre_ffw_norm(x))) + x
        let residual = &x;
        let h = self.pre_feedforward_layernorm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        let h = self.post_feedforward_layernorm.forward(&h)?;
        residual + h
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
                rms_norm_eps,
                rotary_emb,
                dtype,
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
        xs * (self.embedding_length as f64).sqrt()
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
        let mut xs = self.embed(input)?;
        // Masks are additive f32 (-inf / 0); attention casts them to its compute dtype.
        let (full_mask, sliding_mask) = self.masks(seq_len, offset, DType::F32)?;
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let mask = if self.is_sliding[idx] {
                sliding_mask.as_ref()
            } else {
                full_mask.as_ref()
            };
            xs = layer.forward(&xs, mask, offset)?;
        }
        let xs = xs.i((.., seq_len - 1, ..))?;
        let logits = self.output.forward(&self.norm.forward(&xs)?)?;
        match self.final_logit_softcapping {
            None => Ok(logits),
            Some(sc) => (logits / sc)?.tanh()?.affine(sc, 0.0),
        }
    }

    /// Phase A diagnostic: the scaled token embedding (input to layer 0).
    pub fn debug_scaled_embed(&self, input: &Tensor) -> Result<Tensor> {
        self.embed(input)
    }

    /// Phase A verification hook: run a single layer's attention sub-block on a
    /// supplied hidden state `[batch, seq, hidden]`, returning the post-attention
    /// residual. Lets a mid-stack layer (e.g. a global/full layer) be validated
    /// in isolation against a reference, independent of the (Phase A) feedforward.
    pub fn debug_layer_post_attn(
        &mut self,
        hidden: &Tensor,
        layer_idx: usize,
        offset: usize,
    ) -> Result<Tensor> {
        let (_b, seq_len, _) = hidden.dims3()?;
        let (full_mask, sliding_mask) = self.masks(seq_len, offset, DType::F32)?;
        let mask = if self.is_sliding[layer_idx] {
            sliding_mask.as_ref()
        } else {
            full_mask.as_ref()
        };
        self.layers[layer_idx].forward_attn(hidden, mask, offset)
    }

    /// Phase A verification hook: embedding + layer-0 attention sub-block only,
    /// returning the post-attention residual `[batch, seq, hidden]`.
    pub fn debug_layer0_post_attn(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_b, seq_len) = input.dims2()?;
        let xs = self.embed(input)?;
        let (full_mask, sliding_mask) = self.masks(seq_len, offset, DType::F32)?;
        let mask = if self.is_sliding[0] {
            sliding_mask.as_ref()
        } else {
            full_mask.as_ref()
        };
        self.layers[0].forward_attn(&xs, mask, offset)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}

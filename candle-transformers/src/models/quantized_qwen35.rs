//! Qwen3.6 quantized (GGUF) hybrid text model (arch `qwen35`).
//!
//! A hybrid decoder: most layers are Gated DeltaNet (linear attention), and
//! every `full_attention_interval`-th layer is gated full attention; each layer
//! is followed by a dense SwiGLU MLP. The attention layers use partial rotary
//! embeddings and an output gate (Qwen3-Next style).
//!
//! Status: Phase A - GGUF load, the gated full-attention layers, MLP, norms,
//! partial RoPE. The Gated DeltaNet (linear-attention) layers are stubbed as a
//! zero token-mixer (so a forward runs, exercising attention + MLP); the delta
//! rule is added in Phase B.
//!
//! References:
//! - HF `transformers` `qwen3_5`
//! - [Qwen3](https://qwenlm.github.io/)

use super::quantized_qwen3::Gguf;
use super::with_tracing::QMatMul;
use crate::quantized_nn::RmsNorm;
use crate::utils::repeat_kv;
use candle::quantized::gguf_file;
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::Embedding;
use std::io::{Read, Seek};

// ── Partial rotary embedding (rotates the first `dim` head dims, rest pass) ───

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    dim: usize,
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        rope_dim: usize,
        rope_theta: f64,
        max_seq_len: usize,
        dev: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..rope_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / rope_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            dim: rope_dim,
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (_b, _h, seq_len, _d) = x.dims4()?;
        let rot = x.i((.., .., .., ..self.dim))?.contiguous()?;
        let pass = x.i((.., .., .., self.dim..))?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let rot = candle_nn::rotary_emb::rope(&rot, &cos, &sin)?;
        Tensor::cat(&[&rot, &pass], D::Minus1)
    }
}

// ── Gated full attention (Qwen3-Next style) ──────────────────────────────────

struct GatedAttention {
    q_proj: QMatMul, // -> num_heads * head_dim * 2 (query + output gate)
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: RotaryEmbedding,
    kv_cache: Option<(Tensor, Tensor)>,
    dtype: DType,
}

impl GatedAttention {
    #[allow(clippy::too_many_arguments)]
    fn load<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: RotaryEmbedding,
        dtype: DType,
    ) -> Result<Self> {
        let q_proj = gg.qmatmul(&format!("{prefix}.attn_q.weight"))?;
        let k_proj = gg.qmatmul(&format!("{prefix}.attn_k.weight"))?;
        let v_proj = gg.qmatmul(&format!("{prefix}.attn_v.weight"))?;
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
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            rotary_emb,
            kv_cache: None,
            dtype,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let in_dtype = x.dtype();

        // q_proj outputs (query, gate) interleaved per head: reshape to head_dim*2
        // then split the last dim into the query and the output gate.
        let qg = self
            .q_proj
            .forward(x)?
            .reshape((b, l, self.num_heads, 2 * self.head_dim))?;
        let q = qg.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = qg.narrow(D::Minus1, self.head_dim, self.head_dim)?;
        let gate = gate.reshape((b, l, self.num_heads * self.head_dim))?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Per-head Q/K RMSNorm over head_dim.
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let (q, k) = (q.to_dtype(self.dtype)?, k.to_dtype(self.dtype)?);
        let q = self.rotary_emb.apply(&q, offset)?;
        let k = self.rotary_emb.apply(&k, offset)?;
        let v = v.to_dtype(self.dtype)?;

        let (k, v) = match &self.kv_cache {
            Some((kc, vc)) if offset > 0 => {
                (Tensor::cat(&[kc, &k], 2)?, Tensor::cat(&[vc, &v], 2)?)
            }
            _ => (k, v),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(&m.to_dtype(scores.dtype())?)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, l, self.num_heads * self.head_dim))?
            .to_dtype(in_dtype)?;

        // Output gate, then projection.
        let ctx = (ctx * candle_nn::ops::sigmoid(&gate)?)?;
        self.o_proj.forward(&ctx)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

// ── Dense SwiGLU MLP ─────────────────────────────────────────────────────────

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
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ── Token mixer (attention, or a Phase-A stub for DeltaNet layers) ───────────

enum TokenMixer {
    Attn(GatedAttention),
    // Phase A: DeltaNet layers contribute nothing to the residual stream yet.
    LinearStub,
}

// ── Decoder layer ────────────────────────────────────────────────────────────

struct DecoderLayer {
    input_layernorm: RmsNorm,
    mixer: TokenMixer,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl DecoderLayer {
    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let residual = x;
        let h = self.input_layernorm.forward(x)?;
        let h = match &mut self.mixer {
            TokenMixer::Attn(a) => a.forward(&h, mask, offset)?,
            TokenMixer::LinearStub => h.zeros_like()?,
        };
        let x = (residual + h)?;
        let residual = &x;
        let h = self.post_attention_layernorm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        residual + h
    }

    fn clear_kv_cache(&mut self) {
        if let TokenMixer::Attn(a) = &mut self.mixer {
            a.clear_kv_cache();
        }
    }
}

// ── Model ────────────────────────────────────────────────────────────────────

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    output: QMatMul,
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
        let arch = "qwen35";
        let block_count = md_u32(&gg, &format!("{arch}.block_count"))? as usize;
        let num_heads = md_u32(&gg, &format!("{arch}.attention.head_count"))? as usize;
        let num_kv_heads = md_u32(&gg, &format!("{arch}.attention.head_count_kv"))? as usize;
        let head_dim = md_u32(&gg, &format!("{arch}.attention.key_length"))? as usize;
        let embedding_length = md_u32(&gg, &format!("{arch}.embedding_length"))? as usize;
        let context_length = md_u32(&gg, &format!("{arch}.context_length"))? as usize;
        let full_attention_interval =
            md_u32(&gg, &format!("{arch}.full_attention_interval"))? as usize;
        let rope_dim = md_u32(&gg, &format!("{arch}.rope.dimension_count"))? as usize;
        let rms_norm_eps = match gg
            .metadata()
            .get(&format!("{arch}.attention.layer_norm_rms_epsilon"))
        {
            Some(v) => v.to_f32()? as f64,
            None => 1e-6,
        };
        let rope_theta = match gg.metadata().get(&format!("{arch}.rope.freq_base")) {
            Some(v) => v.to_f32()? as f64,
            None => 1e7,
        };

        let tok_embeddings = gg.tensor("token_embd.weight")?.dequantize(device)?;
        let tok_embeddings = Embedding::new(tok_embeddings, embedding_length);
        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        let output = match gg.qmatmul("output.weight") {
            Ok(w) => w,
            Err(_) => gg.qmatmul("token_embd.weight")?,
        };

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let is_attn = (layer_idx + 1) % full_attention_interval == 0;
            let input_layernorm =
                gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
            let post_attention_layernorm = gg.rms_norm(
                &format!("{prefix}.post_attention_norm.weight"),
                rms_norm_eps,
            )?;
            let mixer = if is_attn {
                let rotary_emb =
                    RotaryEmbedding::new(dtype, rope_dim, rope_theta, context_length, device)?;
                TokenMixer::Attn(GatedAttention::load(
                    &mut gg,
                    &prefix,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    rms_norm_eps,
                    rotary_emb,
                    dtype,
                )?)
            } else {
                TokenMixer::LinearStub
            };
            let mlp = Mlp::load(&mut gg, &prefix)?;
            layers.push(DecoderLayer {
                input_layernorm,
                mixer,
                post_attention_layernorm,
                mlp,
            });
        }

        Ok(Self {
            tok_embeddings,
            layers,
            norm,
            output,
            device: device.clone(),
        })
    }

    fn causal_mask(&self, seq_len: usize, offset: usize) -> Result<Option<Tensor>> {
        if seq_len == 1 {
            return Ok(None);
        }
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let mask = Tensor::from_slice(&mask, (seq_len, seq_len), &self.device)?;
        let mask = if offset > 0 {
            let zeros = Tensor::zeros((seq_len, offset), DType::F32, &self.device)?;
            Tensor::cat(&[&zeros, &mask], D::Minus1)?
        } else {
            mask
        };
        Ok(Some(mask.expand((1, 1, seq_len, seq_len + offset))?))
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_b, seq_len) = input.dims2()?;
        let mut xs = self.tok_embeddings.forward(input)?;
        let mask = self.causal_mask(seq_len, offset)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, mask.as_ref(), offset)?;
        }
        let xs = xs.i((.., seq_len - 1, ..))?;
        self.output.forward(&self.norm.forward(&xs)?)
    }

    /// Phase A verification hook: run one decoder layer's forward on a supplied
    /// hidden state, returning the layer output. Lets a single attention layer be
    /// validated in isolation against a reference (teacher-forced).
    pub fn debug_layer(
        &mut self,
        hidden: &Tensor,
        layer_idx: usize,
        offset: usize,
    ) -> Result<Tensor> {
        let (_b, seq_len, _) = hidden.dims3()?;
        let mask = self.causal_mask(seq_len, offset)?;
        self.layers[layer_idx].forward(hidden, mask.as_ref(), offset)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}

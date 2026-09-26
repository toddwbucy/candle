#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle::{test_utils, Device, Tensor};
use candle_nn::{LayerNorm, Module};

#[test]
fn layer_norm() -> Result<()> {
    let device = &Device::Cpu;
    let w = Tensor::new(&[3f32], device)?;
    let b = Tensor::new(&[0.5f32], device)?;
    let ln2 = LayerNorm::new(Tensor::cat(&[&w, &w], 0)?, Tensor::cat(&[&b, &b], 0)?, 1e-8);
    let ln3 = LayerNorm::new(
        Tensor::cat(&[&w, &w, &w], 0)?,
        Tensor::cat(&[&b, &b, &b], 0)?,
        1e-8,
    );
    let ln = LayerNorm::new(w, b, 1e-8);
    assert_eq!(ln.eps(), 1e-8);
    assert!(ln.remove_mean());

    let two = Tensor::new(&[[[2f32]]], device)?;
    let res = ln.forward(&two)?.flatten_all()?;
    assert_eq!(res.to_vec1::<f32>()?, [0.5f32]);

    let inp = Tensor::new(&[[[4f32, 0f32]]], device)?;
    let res = ln2.forward(&inp)?;
    assert_eq!(res.to_vec3::<f32>()?, [[[3.5f32, -2.5]]]);

    let inp = Tensor::new(&[[[1f32, 2., 3.], [4., 5., 6.], [9., 8., 7.]]], device)?;
    let res = ln3.forward(&inp)?;
    assert_eq!(
        test_utils::to_vec3_round(&res, 4)?,
        [[
            [-3.1742, 0.5, 4.1742],
            [-3.1742, 0.5, 4.1742],
            [4.1742, 0.5, -3.1742]
        ]]
    );
    let mean = (res.sum_keepdim(2)? / 3.0)?;
    // The average value should be `b`.
    assert_eq!(
        test_utils::to_vec3_round(&mean, 4)?,
        [[[0.5], [0.5], [0.5]]]
    );
    let std = (res.broadcast_sub(&mean)?.sqr()?.sum_keepdim(2)?.sqrt()? / 3.0)?;
    // The standard deviation should be sqrt(`w`).
    assert_eq!(
        test_utils::to_vec3_round(&std, 4)?,
        [[[1.7321], [1.7321], [1.7321]]]
    );

    // Verify that rms_norm sets remove_mean to false.
    let rms = LayerNorm::rms_norm(Tensor::new(&[1f32], device)?, 1e-5);
    assert_eq!(rms.eps(), 1e-5);
    assert!(!rms.remove_mean());

    Ok(())
}

// Loading through a VarBuilder: a norm without a bias must not need a bias
// tensor, a fresh VarMap must initialise lazily, and the gamma/beta names
// from older checkpoints still resolve.
#[test]
fn layer_norm_loading() -> Result<()> {
    use candle::DType;
    use candle_nn::VarBuilder;
    use std::collections::HashMap;
    let dev = &Device::Cpu;
    let ones = Tensor::ones(4, DType::F32, dev)?;
    let zeros = Tensor::zeros(4, DType::F32, dev)?;

    let only_weight = HashMap::from([("weight".to_string(), ones.clone())]);
    let vb = VarBuilder::from_tensors(only_weight, DType::F32, dev);
    candle_nn::rms_norm(4, 1e-5, vb.clone())?;
    candle_nn::layer_norm_no_bias(4, 1e-5, vb.clone())?;
    assert!(candle_nn::layer_norm(4, 1e-5, vb).is_err());

    let legacy = HashMap::from([
        ("gamma".to_string(), ones.clone()),
        ("beta".to_string(), zeros.clone()),
    ]);
    let vb = VarBuilder::from_tensors(legacy, DType::F32, dev);
    candle_nn::layer_norm(4, 1e-5, vb)?;

    let varmap = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);
    candle_nn::rms_norm(4, 1e-5, vb.pp("rms"))?;
    candle_nn::layer_norm(4, 1e-5, vb.pp("ln"))?;
    assert_eq!(varmap.all_vars().len(), 3);
    Ok(())
}

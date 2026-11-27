# Vendored Zipformer2 (Apache-2.0)

Source: https://github.com/k2-fsa/icefall
Pinned revision: d4d4f281ecefad7b779def552975d2881620a724

zipformer.py, scaling.py, subsampling.py, optim.py:
egs/librispeech/ASR/zipformer/
encoder_interface.py: egs/librispeech/ASR/transducer_stateless/

Original copyright/license headers retained; LICENSE included.
Local changes: relative imports; replace k2 Swoosh kernels with mathematically
equivalent, numerically stable native-PyTorch operations in torch_swoosh.py;
replace the optimizer self-test seed helper's lhotse dependency.
No k2/CUDA extension, lhotse, transducer or FSA is required.

This is the actual upstream Zipformer2 encoder, NOT a renamed Transformer.
Training wrapper, features, CTC normalization, batching and augmentation are
project-specific. This is NOT an exact reproduction of the paper's recipe,
CR-CTC accuracy, or sherpa CPU benchmarks. Native-PyTorch activations can be
slower and differ in floating-point/gradient rounding from k2 fused kernels.

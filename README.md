# E2Former: Equivariant Attention Interatomic Potential

This repository contains the official implementation of E2Former, an equivariant neural network interatomic potential based on efficient attention mechanisms and E(3)-equivariant operations.

> E2Former represents a novel approach to neural network interatomic potentials (NNIPs) that combines the power of attention mechanisms with E(3)-equivariant operations. The model leverages multi-head self-attention within graph neural networks while maintaining rotational equivariance through spherical harmonics and irreducible representations. At its core, E2Former utilizes **Wigner 6j convolution** for efficient and accurate tensor product operations, enabling the model to capture complex many-body interactions while preserving physical symmetries.

E2Former achieves state-of-the-art performance on molecular property prediction tasks by efficiently scaling attention mechanisms while preserving important physical symmetries. The architecture incorporates both invariant and equivariant features through a carefully designed transformer-based architecture that operates on atomic graphs. The model demonstrates superior performance on challenging benchmarks including MD17, MD22, OC20, and SPICE datasets, achieving chemical accuracy for energy and force predictions.

## Key Features

- **Wigner 6j Convolution Core**: Leverages Wigner 6j symbols for efficient E(3)-equivariant tensor products ([arXiv:2501.19216](https://arxiv.org/pdf/2501.19216))
- **E(3)-Equivariant Architecture**: Maintains rotational and translational equivariance through spherical harmonics and tensor products
- **Efficient Attention Mechanisms**: Multiple attention kernel options (math, memory_efficient, flash) for optimal performance
- **Modular Design**: Separated components for easy customization and extension
- **Scalable Architecture**: Designed to efficiently scale with model size and data
- **GPU Optimized**: Leverages optimized attention kernels for fast inference

## Theoretical Foundation

### Wigner 6j Convolution
E2Former's core innovation lies in its use of Wigner 6j symbols for tensor product operations, as detailed in [arXiv:2501.19216](https://arxiv.org/pdf/2501.19216). This approach provides:

- **Efficient Tensor Products**: The Wigner 6j convolution enables efficient computation of E(3)-equivariant tensor products between spherical harmonic features
- **Arbitrary Order Support**: Supports tensor products of arbitrary angular momentum orders (l=0,1,2,...) 
- **Sparse Computation**: Exploits the sparsity structure of Clebsch-Gordan coefficients for computational efficiency
- **Physical Constraints**: Naturally enforces angular momentum selection rules and parity conservation

### Equivariant Attention Mechanism
The attention mechanism in E2Former maintains E(3) equivariance through:
- **Spherical Harmonic Projections**: Node features are represented as spherical harmonic tensors
- **Equivariant Message Passing**: Information aggregation preserves rotational symmetry
- **Multi-Order Attention**: Different attention orders (0,1,2,all) capture increasingly complex angular dependencies
- **Tensor Product Attention**: Attention weights are computed using equivariant tensor products rather than standard dot products

## Installation

### Step 1: Install mamba solver for conda (optional but recommended)

```bash
conda install mamba -n base -c conda-forge
```

### Step 2: Create and activate the environment

```bash
mamba env create -f env.yml
conda activate gotennet
```

Or if you prefer using conda directly:

```bash
conda env create -f env.yml
conda activate gotennet
```

### Step 3: Install FairChem core package

```bash
git submodule update --init --recursive
pip install -e fairchem/packages/fairchem-core
```

### Step 4: Install pre-commit hooks (for contributors)

```bash
pre-commit install
```

## Model Architecture

E2Former consists of several key components:

1. **E2FormerBackbone**: Main model wrapper that handles data preprocessing and coordinates the encoder-decoder pipeline
2. **DIT/Transformer Encoder** (optional): Invariant token embedding layers for initial feature extraction
3. **E2Former Decoder**: Core equivariant transformer with:
   - **Modular Attention System**: Flexible attention mechanisms with different orders and strategies
     - Zero-order, First-order, Second-order, and All-order attention types
     - Multiple alpha computation methods: QK (Query-Key), Dot (Equiformer-style), and memory-efficient variants
   - **Equivariant Operations**: Maintaining E(3) symmetry through spherical harmonics
   - **Tensor Product Operations**: Combining different angular momentum channels
   - **Edge Degree Embeddings**: Radial information processing
   - **Transformer Blocks**: Equivariant self-attention and feedforward layers

The model supports multiple variants:
- **E2former**: Standard implementation for molecular systems
- **E2formerCluster**: Specialized variant with cluster-aware attention mechanisms
- **Modular Architecture**: Refactored version with separated attention components for easier customization

## Training

### Single GPU Training

```bash
python main.py --mode train --config-yml {CONFIG} --run-dir {RUNDIR} --timestamp-id {TIMESTAMP} --checkpoint {CHECKPOINT}
```

### Background Training

Use `start_exp.py` to start a training run in the background:

```bash
python start_exp.py --config-yml {CONFIG} --cvd {GPU_NUM} --run-dir {RUNDIR} --timestamp-id {TIMESTAMP} --checkpoint {CHECKPOINT}
```

### Multi-GPU Training (same node)

```bash
torchrun --standalone --nproc_per_node={N} main.py --distributed --num-gpus {N} {...}
```

## Testing

Run the E2Former test suite to verify the installation:

```bash
python test_e2former.py
```

This will test the model with different batch sizes and verify equivariance properties.

## Molecular Dynamics Simulation

### Setup

Install the MDSim package:

```bash
pip install -e MDsim
```

### Running Simulations

```bash
python simulate.py --simulation_config_yml {SIM_CONFIG} --model_dir {CHECKPOINT_DIR} --model_config_yml {MODEL_CONFIG} --identifier {IDENTIFIER}
```

### Example: MD22 Simulation

```bash
python simulate.py \
    --simulation_config_yml configs/s2ef/MD22/datasets/DHA/simulation.yml \
    --model_dir checkpoints/MD22_DHA/ \
    --model_config_yml configs/s2ef/MD22/E2Former/DHA.yml \
    --identifier test_simulation
```

### Analyzing Results

```bash
PYTHONPATH=./ python scripts/analyze_rollouts_md17_22.py \
    --md_dir checkpoints/MD22_DHA/md_sim_test_simulation \
    --gt_traj /data/md22/md22_AT-AT.npz \
    --xlim 25
```

## Configuration

### Key Configuration Options

- **Attention Configuration**:
  - **Attention Type** (`attn_type`): Choose attention order complexity
    - `zero-order`: Simplest, scalar attention only
    - `first-order`: Includes vector features  
    - `second-order`: Includes tensor features
    - `all-order`: Combines all orders with gating
  - **Alpha Computation** (`tp_type`):
    - `QK_alpha`: Query-Key attention (standard transformer-style)
    - `dot_alpha`: Equiformer-style attention with spherical harmonics
    - `dot_alpha_small`: Memory-efficient variant of dot_alpha
  - **Kernel Implementation**: 
    - `math`: PyTorch default, supports all datatypes and gradient forces
    - `memory_efficient`: Memory-optimized kernel, supports fp32/fp16
    - `flash`: Flash attention kernel, fp16 only, best performance

- **Model Variants**:
  - Set `with_cluster: true` for E2formerCluster variant
  - Configure `encoder: dit` for DIT encoder, or `encoder: transformer` for standard transformer encoder

- **Equivariant Settings**:
  - `irreps_node_embedding`: Irreducible representations for node features (e.g., "128x0e+128x1e+128x2e")
  - `irreps_head`: Irreps for attention heads (e.g., "32x0e+32x1e+32x2e")
  - `lmax`: Maximum angular momentum for spherical harmonics
  - `num_layers`: Number of transformer blocks

### Example Configuration

```yaml
model:
  backbone:
    irreps_node_embedding: "128x0e+128x1e+128x2e"
    num_layers: 8
    encoder: dit
    with_cluster: false
    attn_type: "first-order"
    max_neighbors: 20
    max_radius: 6.0
```

See [`configs/example_config_E2Former.yml`](configs/example_config_E2Former.yml) for a detailed configuration example.

## Project Structure

```
src/
├── models/                      # Main model implementations
│   ├── E2Former_wrapper.py     # Model wrapper and data preprocessing
│   ├── e2former.py             # Original E2Former implementation
│   └── e2former_modular.py     # Refactored modular version
├── layers/                      # Neural network layers
│   ├── attention/              # Modular attention system (NEW)
│   │   ├── base.py            # Base attention class
│   │   ├── sparse.py          # Sparse attention implementation
│   │   ├── cluster.py         # Cluster-aware attention
│   │   ├── orders.py          # Attention order implementations
│   │   ├── alpha.py           # Alpha computation modules
│   │   ├── utils.py           # Shared utilities
│   │   └── compat.py          # Backward compatibility
│   ├── blocks.py               # Transformer blocks
│   ├── embeddings.py           # Embedding networks
│   ├── interaction_blocks.py  # Molecular interactions
│   ├── dit.py                  # DIT encoder blocks
│   └── moe.py                  # Mixture of experts
├── core/                        # Base classes and utilities
│   ├── module_utils.py        # Core utility functions
│   └── e2former_utils.py      # E2Former specific utilities
├── configs/                     # Configuration management
│   └── E2Former_configs.py    # Configuration dataclasses
└── wigner6j/                   # Wigner 6j symbols
    └── tensor_product.py      # E(3)-equivariant operations
```

## Recent Updates

### Modular Attention System (Latest)
The attention mechanism has been refactored into a modular system for better maintainability and extensibility:
- **Separated Components**: Attention orders, alpha computation, and base functionality are now in separate modules
- **Backward Compatible**: All existing code continues to work through the compatibility layer
- **Easier Customization**: New attention mechanisms can be added by extending base classes
- **Better Organization**: Related code is grouped together in the `attention/` subdirectory

## Important Notes

- **Gradient Forces**: When using gradient-based force calculations, disable `torch.compile` as it doesn't support second-order gradients
- **Memory Management**: Adjust `max_num_nodes_per_batch` for optimal GPU memory usage
- **FP16 Training**: Use `use_fp16_backbone` or AutoMixedPrecision for improved performance
- **Attention Types**: The same channels must be used across all irreps orders (e.g., "128x0e+128x1e+128x2e")

## Citation

If you find E2Former useful in your research, please consider citing:

```bibtex
@article{e2former2025,
  title={E2Former: Efficient Equivariant Attention for Neural Network Potentials},
  author={Liu, Yusong and Wang, Shaoning and Wang, Mingyu and Dral, Pavlo O.},
  journal={arXiv preprint arXiv:2501.19216},
  year={2025}
}
```

For the theoretical foundation and Wigner 6j convolution details:
```bibtex
@article{wigner6j2025,
  title={Wigner 6j Convolution: Efficient Tensor Products for E(3)-Equivariant Networks},
  author={Liu, Yusong and Wang, Shaoning and Wang, Mingyu and Dral, Pavlo O.},
  journal={arXiv preprint arXiv:2501.19216},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

E2Former builds upon several excellent works in the field of neural network interatomic potentials and equivariant neural networks. We particularly acknowledge the FairChem framework for providing the foundation for this implementation.
# E2Former: Equivariant Attention Interatomic Potential

This repository contains the official implementation of E2Former, an equivariant neural network interatomic potential based on efficient attention mechanisms and E(3)-equivariant operations.

> E2Former represents a novel approach to neural network interatomic potentials (NNIPs) that combines the power of attention mechanisms with E(3)-equivariant operations. The model leverages multi-head self-attention within graph neural networks while maintaining rotational equivariance through spherical harmonics and irreducible representations.

E2Former achieves state-of-the-art performance on molecular property prediction tasks by efficiently scaling attention mechanisms while preserving important physical symmetries. The architecture incorporates both invariant and equivariant features through a carefully designed transformer-based architecture that operates on atomic graphs.

## Key Features

- **E(3)-Equivariant Architecture**: Maintains rotational and translational equivariance through spherical harmonics and tensor products
- **Efficient Attention Mechanisms**: Multiple attention kernel options (math, memory_efficient, flash) for optimal performance
- **Modular Design**: Separated components for easy customization and extension
- **Scalable Architecture**: Designed to efficiently scale with model size and data
- **GPU Optimized**: Leverages optimized attention kernels for fast inference

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
   - Equivariant attention layers maintaining E(3) symmetry
   - Tensor product operations for combining different angular momentum channels
   - Edge degree embedding networks for radial information
   - Transformer blocks with equivariant operations

The model supports two main variants:
- **E2former**: Standard implementation for molecular systems
- **E2formerCluster**: Specialized variant with cluster-aware attention mechanisms

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

- **Attention Type**: Choose between different attention implementations:
  - `math`: PyTorch default, supports all datatypes and gradient forces
  - `memory_efficient`: Memory-optimized kernel, supports fp32/fp16
  - `flash`: Flash attention kernel, fp16 only, best performance

- **Model Variants**:
  - Set `with_cluster: true` for E2formerCluster variant
  - Configure `encoder: dit` for DIT encoder, or `encoder: transformer` for standard transformer encoder

- **Equivariant Settings**:
  - `irreps_node_embedding`: Irreducible representations for node features (e.g., "128x0e+128x1e+128x2e")
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
├── models/           # Main model implementations
│   ├── e2former_main.py         # Core E2former model
│   ├── e2former_cluster.py      # Cluster-aware variant
│   └── E2Former_wrapper.py      # Model wrapper and heads
├── layers/           # Neural network layers
│   ├── attention.py             # Attention mechanisms
│   ├── blocks.py                # Transformer blocks
│   ├── embeddings.py            # Embedding networks
│   └── interaction_blocks.py   # Molecular interactions
├── core/            # Base classes and utilities
│   ├── base_modules.py          # Base equivariant modules
│   └── e2former_utils.py        # Configuration utilities
├── configs/         # Configuration management
└── utils/          # General utilities
```

## Important Notes

- **Gradient Forces**: When using gradient-based force calculations, disable `torch.compile` as it doesn't support second-order gradients
- **Memory Management**: Adjust `max_num_nodes_per_batch` for optimal GPU memory usage
- **FP16 Training**: Use `use_fp16_backbone` or AutoMixedPrecision for improved performance

## Citation

If you find E2Former useful in your research, please consider citing:

```bibtex
@article{e2former2024,
  title={E2Former: Equivariant Attention Interatomic Potential},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

E2Former builds upon several excellent works in the field of neural network interatomic potentials and equivariant neural networks. We particularly acknowledge the FairChem framework for providing the foundation for this implementation.
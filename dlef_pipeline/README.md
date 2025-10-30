## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone repository
git clone git@github.com:AI-ON-Laboratory/MIL.git
cd mil

conda create -n mil python=3.9
conda activate mil

# Install dependencies
pip install -r requirements.txt

# Configure environment
export MPLBACKEND=Agg  # Required for headless plotting
```

### 3. Train Model

```bash
python pipeline/train.py \
  --config configs/config_survival_cv.yaml \
  --base_dir ./results
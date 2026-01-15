# AMB-DSGDN: Adaptive Modality-Balanced Dynamic Semantic Graph Differential Network for Multimodal Emotion Recognition

## I. Environment Configuration
### 1. Prerequisites
- Anaconda is installed.
- The server is equipped with an NVIDIA GPU supporting CUDA 11.7.
- The AMB-DSGDN project code has been downloaded.

### 2. Operation Steps
```bash
# 1. Create and activate the conda environment
conda create -n AMB-DSGDN python=3.10.13 -y
conda activate AMB-DSGDN

# 2. Install dependency packages
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install pandas==2.2.0 numpy==1.26.3 thop

# 3. Navigate to the project directory
cd AMB-DSGDN

# 4. Grant execution permissions to the scripts
chmod +x run_iemocap.sh
chmod +x run_meld.sh
```

## II. Run Experiments
```bash
# Run experiments on the IEMOCAP dataset
./run_iemocap.sh

# Run experiments on the MELD dataset
./run_meld.sh
```

### Key Notes for Translation Consistency:
1. **Term Standardization**:
   - "环境配置" → "Environment Configuration" (consistent with technical documentation conventions)
   - "前置条件" → "Prerequisites" (standard for software setup guides)
   - "操作步骤" → "Operation Steps" (clear and concise for technical instructions)
   - "赋予脚本执行权限" → "Grant execution permissions to the scripts" (accurate technical expression)
2. **Code Preservation**:
   - All bash commands, package names, version numbers, and script filenames are retained unchanged to ensure functional consistency.
3. **Grammatical Rigor**:
   - Passive voice is used for prerequisite descriptions (e.g., "Anaconda is installed") to maintain formality.
   - Imperative mood is used for operation steps (e.g., "Create and activate...", "Install...") to align with technical documentation norms.
4. **Cultural Adaptation**:
   - "服务器配备" → "The server is equipped with" (natural expression in English technical writing)
   - "数据集实验" → "experiments on the [Dataset Name] dataset" (redundancy avoided while retaining clarity)

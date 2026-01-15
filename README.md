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



中文说明：

# AMB-DSGDN: Adaptive Modality-Balanced Dynamic Semantic Graph Differential Network for Multimodal Emotion Recognition

## 一、环境配置
### 1. 前置条件
- 已安装 Anaconda 包管理工具
- 服务器配备支持 CUDA 11.7 的 NVIDIA GPU（保证模型训练的硬件需求）
- 已下载 AMB-DSGDN 项目代码
- 已下载实验所需数据集：
  - IEMOCAP 数据集
  - MELD 数据集
  - 数据集下载地址：[Google Drive](https://drive.google.com/drive/folders/1J1mvbqQmVodNBzbiOIxRiWOtkP6qqP-K)

### 2. 操作步骤
```bash
# 1. 创建并激活conda环境
conda create -n AMB-DSGDN python=3.10.13 -y
conda activate AMB-DSGDN

# 2. 安装依赖包
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install pandas==2.2.0 numpy==1.26.3 thop

# 3. 进入项目目录
cd AMB-DSGDN

# 4. 赋予脚本执行权限
chmod +x run_iemocap.sh
chmod +x run_meld.sh
```

## 二、运行实验
```bash
# 运行IEMOCAP数据集实验
./run_iemocap.sh

# 运行MELD数据集实验
./run_meld.sh
```


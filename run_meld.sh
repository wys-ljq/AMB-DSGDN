python train.py \
  --lr=0.00005 \
  --dropout=0.11 \
  --l2=0.000075 \
  --batch-size=64 \
  --hidden_dim=512 \
  --n_head=8 \
  --epochs=20 \
  --windows=6 \
  --Dataset=MELD \
  --save_model_path=./MELD \
  --q_base=0.2 \
  --lam=0.9 \
  --p_exe=0.2 \
  --warm_up_epochs=0


# ./run_meld.sh
# seed = 2094
# Namespace(no_cuda=False, lr=5e-05, dropout=0.11, l2=7.5e-05, batch_size=64, hidden_dim=512, n_head=8, epochs=20, windows=6, class_weight=True, Dataset='MELD', save_model_path='./MELD', disable_modality_dropout=False, q_base=0.2, lam=0.9, p_exe=0.2, warm_up_epochs=0)
# Running on GPU
# total parameters: 12481973
# training parameters: 12481973
# 模型总参数数量: 12,481,973
# 使用thop计算的FLOPs: 9,108,253,184.0 (9.108G)
# epoch: 1, train_loss: 2.2335, train_acc: 49.42, train_fscore: 43.41, test_loss: 1.8327, test_acc: 65.13, test_fscore: 59.61, epoch_time: 8.86 sec
#   计算开销指标 - 训练时间/epoch: 8.8627 sec, 训练时间/batch: 0.2066 sec, 推理时间: 0.0002 sec, 显存占用: 2390.92 MB, FLOPs: 9
# epoch: 2, train_loss: 1.7212, train_acc: 67.76, train_fscore: 63.86, test_loss: 1.6963, test_acc: 66.28, test_fscore: 62.44, epoch_time: 8.57 sec
#   计算开销指标 - 训练时间/epoch: 8.5742 sec, 训练时间/batch: 0.2095 sec, 推理时间: 0.0002 sec, 显存占用: 4567.75 MB, FLOPs: 9
# epoch: 3, train_loss: 1.6002, train_acc: 69.42, train_fscore: 66.14, test_loss: 1.6673, test_acc: 66.63, test_fscore: 63.38, epoch_time: 8.65 sec
#   计算开销指标 - 训练时间/epoch: 8.6487 sec, 训练时间/batch: 0.2046 sec, 推理时间: 0.0001 sec, 显存占用: 4567.75 MB, FLOPs: 9
# epoch: 4, train_loss: 1.5451, train_acc: 70.49, train_fscore: 67.64, test_loss: 1.6474, test_acc: 66.59, test_fscore: 63.88, epoch_time: 9.39 sec
#   计算开销指标 - 训练时间/epoch: 9.3900 sec, 训练时间/batch: 0.2105 sec, 推理时间: 0.0001 sec, 显存占用: 4567.75 MB, FLOPs: 9
# epoch: 5, train_loss: 1.5582, train_acc: 69.94, train_fscore: 67.39, test_loss: 1.6524, test_acc: 66.51, test_fscore: 63.75, epoch_time: 9.17 sec
#   计算开销指标 - 训练时间/epoch: 9.1717 sec, 训练时间/batch: 0.2112 sec, 推理时间: 0.0001 sec, 显存占用: 4568.24 MB, FLOPs: 9
# epoch: 6, train_loss: 1.5416, train_acc: 70.53, train_fscore: 68.21, test_loss: 1.6336, test_acc: 67.13, test_fscore: 64.82, epoch_time: 9.33 sec
#   计算开销指标 - 训练时间/epoch: 9.3318 sec, 训练时间/batch: 0.2106 sec, 推理时间: 0.0002 sec, 显存占用: 6743.90 MB, FLOPs: 9
# epoch: 7, train_loss: 1.5671, train_acc: 70.01, train_fscore: 67.74, test_loss: 1.6305, test_acc: 67.01, test_fscore: 64.89, epoch_time: 9.50 sec
#   计算开销指标 - 训练时间/epoch: 9.4967 sec, 训练时间/batch: 0.2112 sec, 推理时间: 0.0002 sec, 显存占用: 4566.47 MB, FLOPs: 9
# epoch: 8, train_loss: 1.5359, train_acc: 70.26, train_fscore: 68.04, test_loss: 1.6253, test_acc: 67.09, test_fscore: 65.07, epoch_time: 9.46 sec
#   计算开销指标 - 训练时间/epoch: 9.4576 sec, 训练时间/batch: 0.2118 sec, 推理时间: 0.0002 sec, 显存占用: 4566.56 MB, FLOPs: 9
# epoch: 9, train_loss: 1.4246, train_acc: 73.48, train_fscore: 71.88, test_loss: 1.6296, test_acc: 67.13, test_fscore: 65.02, epoch_time: 9.20 sec
#   计算开销指标 - 训练时间/epoch: 9.1980 sec, 训练时间/batch: 0.2169 sec, 推理时间: 0.0001 sec, 显存占用: 4564.91 MB, FLOPs: 9
# epoch: 10, train_loss: 1.4983, train_acc: 71.14, train_fscore: 69.28, test_loss: 1.6266, test_acc: 67.24, test_fscore: 65.31, epoch_time: 9.07 sec
#   计算开销指标 - 训练时间/epoch: 9.0665 sec, 训练时间/batch: 0.2094 sec, 推理时间: 0.0002 sec, 显存占用: 6740.02 MB, FLOPs: 9
# epoch: 11, train_loss: 1.4883, train_acc: 71.5, train_fscore: 69.64, test_loss: 1.6279, test_acc: 67.28, test_fscore: 65.11, epoch_time: 9.42 sec
#   计算开销指标 - 训练时间/epoch: 9.4198 sec, 训练时间/batch: 0.2124 sec, 推理时间: 0.0001 sec, 显存占用: 4565.35 MB, FLOPs: 9
# epoch: 12, train_loss: 1.4151, train_acc: 73.5, train_fscore: 72.0, test_loss: 1.6326, test_acc: 67.28, test_fscore: 65.12, epoch_time: 9.91 sec
#   计算开销指标 - 训练时间/epoch: 9.9118 sec, 训练时间/batch: 0.2079 sec, 推理时间: 0.0002 sec, 显存占用: 6739.88 MB, FLOPs: 9
# epoch: 13, train_loss: 1.3828, train_acc: 73.96, train_fscore: 72.65, test_loss: 1.6554, test_acc: 67.05, test_fscore: 64.47, epoch_time: 9.50 sec
#   计算开销指标 - 训练时间/epoch: 9.4993 sec, 训练时间/batch: 0.2108 sec, 推理时间: 0.0002 sec, 显存占用: 6739.73 MB, FLOPs: 9
# epoch: 14, train_loss: 1.4142, train_acc: 73.49, train_fscore: 71.99, test_loss: 1.6283, test_acc: 67.2, test_fscore: 65.36, epoch_time: 9.45 sec
#   计算开销指标 - 训练时间/epoch: 9.4455 sec, 训练时间/batch: 0.2172 sec, 推理时间: 0.0002 sec, 显存占用: 6739.89 MB, FLOPs: 9
# epoch: 15, train_loss: 1.3819, train_acc: 73.56, train_fscore: 72.12, test_loss: 1.6357, test_acc: 67.39, test_fscore: 65.4, epoch_time: 9.56 sec
#   计算开销指标 - 训练时间/epoch: 9.5640 sec, 训练时间/batch: 0.2219 sec, 推理时间: 0.0002 sec, 显存占用: 4566.03 MB, FLOPs: 9
# epoch: 16, train_loss: 1.3649, train_acc: 74.34, train_fscore: 73.1, test_loss: 1.6356, test_acc: 67.62, test_fscore: 65.73, epoch_time: 9.34 sec
#   计算开销指标 - 训练时间/epoch: 9.3445 sec, 训练时间/batch: 0.2253 sec, 推理时间: 0.0002 sec, 显存占用: 4566.56 MB, FLOPs: 9
# epoch: 17, train_loss: 1.3694, train_acc: 73.95, train_fscore: 72.61, test_loss: 1.6479, test_acc: 66.82, test_fscore: 64.75, epoch_time: 9.37 sec
#   计算开销指标 - 训练时间/epoch: 9.3744 sec, 训练时间/batch: 0.2175 sec, 推理时间: 0.0001 sec, 显存占用: 4564.99 MB, FLOPs: 9
# epoch: 18, train_loss: 1.3937, train_acc: 73.61, train_fscore: 72.14, test_loss: 1.6343, test_acc: 67.32, test_fscore: 65.5, epoch_time: 9.66 sec
#   计算开销指标 - 训练时间/epoch: 9.6566 sec, 训练时间/batch: 0.2211 sec, 推理时间: 0.0002 sec, 显存占用: 6740.66 MB, FLOPs: 9
# epoch: 19, train_loss: 1.3455, train_acc: 74.82, train_fscore: 73.53, test_loss: 1.6288, test_acc: 67.51, test_fscore: 66.06, epoch_time: 9.61 sec
#   计算开销指标 - 训练时间/epoch: 9.6053 sec, 训练时间/batch: 0.2260 sec, 推理时间: 0.0001 sec, 显存占用: 6739.95 MB, FLOPs: 9
# epoch: 20, train_loss: 1.428, train_acc: 72.65, train_fscore: 71.07, test_loss: 1.6398, test_acc: 67.09, test_fscore: 65.51, epoch_time: 9.43 sec
#   计算开销指标 - 训练时间/epoch: 9.4336 sec, 训练时间/batch: 0.2093 sec, 推理时间: 0.0001 sec, 显存占用: 4562.12 MB, FLOPs: 9
# ================================================================================
# Model Performance:
# Best_Test-FScore-epoch_index: 19
# Best_Test_F-Score: 66.06
#               precision    recall  f1-score   support

#            0     0.7493    0.8567    0.7994    1256.0
#            1     0.5948    0.5694    0.5818     281.0
#            2     0.2857    0.1200    0.1690      50.0
#            3     0.5763    0.3269    0.4172     208.0
#            4     0.6460    0.6219    0.6337     402.0
#            5     0.4000    0.2059    0.2718      68.0
#            6     0.5465    0.5449    0.5457     345.0

#     accuracy                         0.6751    2610.0
#    macro avg     0.5427    0.4637    0.4884    2610.0
# weighted avg     0.6582    0.6751    0.6606    2610.0


# ================================================================================
# 最终计算开销统计:
# --------------------------------------------------------------------------------
# 1. 训练时间:
#    - 平均每个epoch: 9.5278 秒
#    - 平均每个batch: 0.2123 秒
#    - 总训练时间: 285.83 秒 (4.76 分钟)

# 2. 推理时间:
#    - 平均推理时间: 0.0001 秒/样本

# 3. 显存占用:
#    - 平均显存占用: 5724.74 MB
#    - 峰值显存占用: 6743.90 MB

# 4. 计算量 (FLOPs):
#    - 模型FLOPs: 9
# ================================================================================
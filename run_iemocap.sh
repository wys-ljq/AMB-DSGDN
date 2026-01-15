python train.py \
--lr=0.000068 \
--dropout=0.5 \
--l2=0.00005 \
--batch-size=16 \
--hidden_dim=512 \
--n_head=8 \
--epochs=61 \
--windows=20 \
--class-weight \
--Dataset=IEMOCAP \
--save_model_path=./IEMOCAP \
--q_base=0.3 \
--lam=0.9 \
--p_exe=0.5 \
--warm_up_epochs=60

# ./run_iemocap.sh
# seed = 2094
# Namespace(no_cuda=False, lr=6.8e-05, dropout=0.5, l2=5e-05, batch_size=16, hidden_dim=512, n_head=8, epochs=61, windows=20, class_weight=True, Dataset='IEMOCAP', save_model_path='./IEMOCAP', disable_modality_dropout=False, q_base=0.3, lam=0.9, p_exe=0.5, warm_up_epochs=60)
# Running on GPU
# total parameters: 13133234
# training parameters: 13133234
# 模型总参数数量: 13,133,234
# 使用thop计算的FLOPs: 9,561,135,744.0 (9.561G)
# epoch: 1, train_loss: 2.573, train_acc: 29.54, train_fscore: 28.51, test_loss: 2.1179, test_acc: 50.52, test_fscore: 49.8, epoch_time: 17.79 sec
#   计算开销指标 - 训练时间/epoch: 17.7910 sec, 训练时间/batch: 0.1979 sec, 推理时间: 0.0001 sec, 显存占用: 1333.69 MB, FLOPs: 9
# epoch: 2, train_loss: 1.9948, train_acc: 50.65, train_fscore: 50.46, test_loss: 1.9137, test_acc: 55.7, test_fscore: 56.0, epoch_time: 20.03 sec
#   计算开销指标 - 训练时间/epoch: 20.0304 sec, 训练时间/batch: 0.2096 sec, 推理时间: 0.0001 sec, 显存占用: 2422.90 MB, FLOPs: 9
# epoch: 3, train_loss: 1.7861, train_acc: 56.01, train_fscore: 55.85, test_loss: 1.8075, test_acc: 56.19, test_fscore: 56.98, epoch_time: 19.86 sec
#   计算开销指标 - 训练时间/epoch: 19.8613 sec, 训练时间/batch: 0.2068 sec, 推理时间: 0.0001 sec, 显存占用: 2420.65 MB, FLOPs: 9
# epoch: 4, train_loss: 1.6781, train_acc: 58.64, train_fscore: 58.88, test_loss: 1.7483, test_acc: 58.53, test_fscore: 59.12, epoch_time: 19.37 sec
#   计算开销指标 - 训练时间/epoch: 19.3689 sec, 训练时间/batch: 0.2019 sec, 推理时间: 0.0001 sec, 显存占用: 2419.09 MB, FLOPs: 9
# epoch: 5, train_loss: 1.6092, train_acc: 61.53, train_fscore: 61.25, test_loss: 1.705, test_acc: 60.26, test_fscore: 60.02, epoch_time: 19.96 sec
#   计算开销指标 - 训练时间/epoch: 19.9563 sec, 训练时间/batch: 0.1993 sec, 推理时间: 0.0001 sec, 显存占用: 2421.35 MB, FLOPs: 9
# epoch: 6, train_loss: 1.5627, train_acc: 63.8, train_fscore: 63.58, test_loss: 1.6735, test_acc: 63.34, test_fscore: 63.25, epoch_time: 19.20 sec
#   计算开销指标 - 训练时间/epoch: 19.2035 sec, 训练时间/batch: 0.1971 sec, 推理时间: 0.0001 sec, 显存占用: 2421.99 MB, FLOPs: 9
# epoch: 7, train_loss: 1.4795, train_acc: 65.9, train_fscore: 65.73, test_loss: 1.646, test_acc: 60.94, test_fscore: 61.19, epoch_time: 19.99 sec
#   计算开销指标 - 训练时间/epoch: 19.9931 sec, 训练时间/batch: 0.1882 sec, 推理时间: 0.0001 sec, 显存占用: 2432.41 MB, FLOPs: 9
# epoch: 8, train_loss: 1.4297, train_acc: 67.16, train_fscore: 66.91, test_loss: 1.5716, test_acc: 64.7, test_fscore: 64.75, epoch_time: 19.57 sec
#   计算开销指标 - 训练时间/epoch: 19.5713 sec, 训练时间/batch: 0.1942 sec, 推理时间: 0.0001 sec, 显存占用: 3521.44 MB, FLOPs: 9
# epoch: 9, train_loss: 1.4025, train_acc: 67.87, train_fscore: 67.86, test_loss: 1.5322, test_acc: 64.57, test_fscore: 65.05, epoch_time: 20.56 sec
#   计算开销指标 - 训练时间/epoch: 20.5577 sec, 训练时间/batch: 0.2068 sec, 推理时间: 0.0001 sec, 显存占用: 2421.38 MB, FLOPs: 9
# epoch: 10, train_loss: 1.3634, train_acc: 68.3, train_fscore: 68.17, test_loss: 1.4821, test_acc: 66.36, test_fscore: 66.36, epoch_time: 20.36 sec
#   计算开销指标 - 训练时间/epoch: 20.3556 sec, 训练时间/batch: 0.2053 sec, 推理时间: 0.0001 sec, 显存占用: 2416.79 MB, FLOPs: 9
# epoch: 11, train_loss: 1.316, train_acc: 69.81, train_fscore: 69.64, test_loss: 1.4716, test_acc: 66.97, test_fscore: 67.43, epoch_time: 19.21 sec
#   计算开销指标 - 训练时间/epoch: 19.2113 sec, 训练时间/batch: 0.1898 sec, 推理时间: 0.0001 sec, 显存占用: 2417.26 MB, FLOPs: 9
# epoch: 12, train_loss: 1.2816, train_acc: 71.1, train_fscore: 71.01, test_loss: 1.4326, test_acc: 67.53, test_fscore: 67.92, epoch_time: 20.61 sec
#   计算开销指标 - 训练时间/epoch: 20.6136 sec, 训练时间/batch: 0.2012 sec, 推理时间: 0.0001 sec, 显存占用: 2420.88 MB, FLOPs: 9
# epoch: 13, train_loss: 1.2656, train_acc: 71.53, train_fscore: 71.34, test_loss: 1.4154, test_acc: 68.76, test_fscore: 68.89, epoch_time: 18.96 sec
#   计算开销指标 - 训练时间/epoch: 18.9628 sec, 训练时间/batch: 0.1927 sec, 推理时间: 0.0001 sec, 显存占用: 2432.67 MB, FLOPs: 9
# epoch: 14, train_loss: 1.2438, train_acc: 72.46, train_fscore: 72.14, test_loss: 1.4103, test_acc: 69.07, test_fscore: 69.18, epoch_time: 19.16 sec
#   计算开销指标 - 训练时间/epoch: 19.1583 sec, 训练时间/batch: 0.2080 sec, 推理时间: 0.0001 sec, 显存占用: 2433.14 MB, FLOPs: 9
# epoch: 15, train_loss: 1.2157, train_acc: 72.69, train_fscore: 72.53, test_loss: 1.4185, test_acc: 68.52, test_fscore: 68.96, epoch_time: 21.54 sec
#   计算开销指标 - 训练时间/epoch: 21.5356 sec, 训练时间/batch: 0.1848 sec, 推理时间: 0.0001 sec, 显存占用: 2431.90 MB, FLOPs: 9
# epoch: 16, train_loss: 1.1923, train_acc: 73.3, train_fscore: 73.21, test_loss: 1.396, test_acc: 69.32, test_fscore: 69.46, epoch_time: 20.11 sec
#   计算开销指标 - 训练时间/epoch: 20.1119 sec, 训练时间/batch: 0.2049 sec, 推理时间: 0.0001 sec, 显存占用: 3523.00 MB, FLOPs: 9
# epoch: 17, train_loss: 1.163, train_acc: 74.97, train_fscore: 74.77, test_loss: 1.3823, test_acc: 70.06, test_fscore: 70.05, epoch_time: 19.26 sec
#   计算开销指标 - 训练时间/epoch: 19.2568 sec, 训练时间/batch: 0.2076 sec, 推理时间: 0.0001 sec, 显存占用: 2424.88 MB, FLOPs: 9
# epoch: 18, train_loss: 1.1571, train_acc: 74.49, train_fscore: 74.35, test_loss: 1.3744, test_acc: 70.18, test_fscore: 70.43, epoch_time: 19.67 sec
#   计算开销指标 - 训练时间/epoch: 19.6716 sec, 训练时间/batch: 0.2137 sec, 推理时间: 0.0001 sec, 显存占用: 2421.35 MB, FLOPs: 9
# epoch: 19, train_loss: 1.1274, train_acc: 76.37, train_fscore: 76.34, test_loss: 1.372, test_acc: 70.98, test_fscore: 70.91, epoch_time: 19.12 sec
#   计算开销指标 - 训练时间/epoch: 19.1193 sec, 训练时间/batch: 0.2095 sec, 推理时间: 0.0001 sec, 显存占用: 2423.52 MB, FLOPs: 9
# epoch: 20, train_loss: 1.1254, train_acc: 75.97, train_fscore: 75.68, test_loss: 1.3637, test_acc: 70.24, test_fscore: 70.31, epoch_time: 18.93 sec
#   计算开销指标 - 训练时间/epoch: 18.9348 sec, 训练时间/batch: 0.1984 sec, 推理时间: 0.0001 sec, 显存占用: 2426.16 MB, FLOPs: 9
# epoch: 21, train_loss: 1.113, train_acc: 76.4, train_fscore: 76.38, test_loss: 1.3547, test_acc: 70.67, test_fscore: 71.01, epoch_time: 18.36 sec
#   计算开销指标 - 训练时间/epoch: 18.3623 sec, 训练时间/batch: 0.2015 sec, 推理时间: 0.0001 sec, 显存占用: 3515.01 MB, FLOPs: 9
# epoch: 22, train_loss: 1.0917, train_acc: 77.26, train_fscore: 77.07, test_loss: 1.3516, test_acc: 71.35, test_fscore: 71.15, epoch_time: 18.90 sec
#   计算开销指标 - 训练时间/epoch: 18.8993 sec, 训练时间/batch: 0.1966 sec, 推理时间: 0.0001 sec, 显存占用: 2423.57 MB, FLOPs: 9
# epoch: 23, train_loss: 1.0872, train_acc: 76.88, train_fscore: 76.8, test_loss: 1.3485, test_acc: 69.99, test_fscore: 70.31, epoch_time: 18.74 sec
#   计算开销指标 - 训练时间/epoch: 18.7389 sec, 训练时间/batch: 0.2089 sec, 推理时间: 0.0001 sec, 显存占用: 2421.65 MB, FLOPs: 9
# epoch: 24, train_loss: 1.0623, train_acc: 77.68, train_fscore: 77.52, test_loss: 1.3389, test_acc: 71.9, test_fscore: 71.7, epoch_time: 19.37 sec
#   计算开销指标 - 训练时间/epoch: 19.3750 sec, 训练时间/batch: 0.2232 sec, 推理时间: 0.0001 sec, 显存占用: 3508.69 MB, FLOPs: 9
# epoch: 25, train_loss: 1.0492, train_acc: 77.81, train_fscore: 77.68, test_loss: 1.3353, test_acc: 69.99, test_fscore: 70.12, epoch_time: 18.20 sec
#   计算开销指标 - 训练时间/epoch: 18.1952 sec, 训练时间/batch: 0.1867 sec, 推理时间: 0.0001 sec, 显存占用: 2419.03 MB, FLOPs: 9
# epoch: 26, train_loss: 1.0335, train_acc: 78.06, train_fscore: 77.93, test_loss: 1.3209, test_acc: 70.98, test_fscore: 71.06, epoch_time: 19.17 sec
#   计算开销指标 - 训练时间/epoch: 19.1666 sec, 训练时间/batch: 0.1970 sec, 推理时间: 0.0001 sec, 显存占用: 3506.01 MB, FLOPs: 9
# epoch: 27, train_loss: 1.0145, train_acc: 78.71, train_fscore: 78.56, test_loss: 1.3303, test_acc: 70.43, test_fscore: 70.47, epoch_time: 19.84 sec
#   计算开销指标 - 训练时间/epoch: 19.8437 sec, 训练时间/batch: 0.1883 sec, 推理时间: 0.0001 sec, 显存占用: 3501.14 MB, FLOPs: 9
# epoch: 28, train_loss: 0.9978, train_acc: 79.83, train_fscore: 79.76, test_loss: 1.3097, test_acc: 71.47, test_fscore: 71.64, epoch_time: 19.05 sec
#   计算开销指标 - 训练时间/epoch: 19.0538 sec, 训练时间/batch: 0.1960 sec, 推理时间: 0.0001 sec, 显存占用: 3503.19 MB, FLOPs: 9
# epoch: 29, train_loss: 0.9959, train_acc: 78.3, train_fscore: 78.08, test_loss: 1.3129, test_acc: 70.86, test_fscore: 70.92, epoch_time: 20.10 sec
#   计算开销指标 - 训练时间/epoch: 20.0996 sec, 训练时间/batch: 0.2016 sec, 推理时间: 0.0001 sec, 显存占用: 3511.64 MB, FLOPs: 9
# epoch: 30, train_loss: 0.974, train_acc: 79.67, train_fscore: 79.55, test_loss: 1.3298, test_acc: 71.35, test_fscore: 71.48, epoch_time: 19.43 sec
#   计算开销指标 - 训练时间/epoch: 19.4339 sec, 训练时间/batch: 0.1957 sec, 推理时间: 0.0001 sec, 显存占用: 3513.38 MB, FLOPs: 9
# epoch: 31, train_loss: 0.9716, train_acc: 79.98, train_fscore: 79.91, test_loss: 1.3121, test_acc: 71.04, test_fscore: 71.22, epoch_time: 20.96 sec
#   计算开销指标 - 训练时间/epoch: 20.9564 sec, 训练时间/batch: 0.2045 sec, 推理时间: 0.0001 sec, 显存占用: 3510.22 MB, FLOPs: 9
# epoch: 32, train_loss: 0.9545, train_acc: 80.34, train_fscore: 80.21, test_loss: 1.3116, test_acc: 71.78, test_fscore: 71.73, epoch_time: 19.20 sec
#   计算开销指标 - 训练时间/epoch: 19.1964 sec, 训练时间/batch: 0.2034 sec, 推理时间: 0.0001 sec, 显存占用: 3508.77 MB, FLOPs: 9
# epoch: 33, train_loss: 0.9438, train_acc: 80.69, train_fscore: 80.62, test_loss: 1.3104, test_acc: 71.66, test_fscore: 71.75, epoch_time: 20.55 sec
#   计算开销指标 - 训练时间/epoch: 20.5526 sec, 训练时间/batch: 0.2111 sec, 推理时间: 0.0001 sec, 显存占用: 2428.95 MB, FLOPs: 9
# epoch: 34, train_loss: 0.937, train_acc: 80.93, train_fscore: 80.87, test_loss: 1.2797, test_acc: 72.4, test_fscore: 72.39, epoch_time: 19.93 sec
#   计算开销指标 - 训练时间/epoch: 19.9333 sec, 训练时间/batch: 0.1982 sec, 推理时间: 0.0001 sec, 显存占用: 2428.76 MB, FLOPs: 9
# epoch: 35, train_loss: 0.9225, train_acc: 80.64, train_fscore: 80.52, test_loss: 1.3164, test_acc: 71.29, test_fscore: 71.13, epoch_time: 21.36 sec
#   计算开销指标 - 训练时间/epoch: 21.3627 sec, 训练时间/batch: 0.2042 sec, 推理时间: 0.0001 sec, 显存占用: 2430.74 MB, FLOPs: 9
# epoch: 36, train_loss: 0.9133, train_acc: 81.76, train_fscore: 81.71, test_loss: 1.2802, test_acc: 71.78, test_fscore: 71.95, epoch_time: 21.92 sec
#   计算开销指标 - 训练时间/epoch: 21.9161 sec, 训练时间/batch: 0.2046 sec, 推理时间: 0.0001 sec, 显存占用: 3518.62 MB, FLOPs: 9
# epoch: 37, train_loss: 0.8997, train_acc: 81.38, train_fscore: 81.28, test_loss: 1.3052, test_acc: 72.09, test_fscore: 71.97, epoch_time: 21.37 sec
#   计算开销指标 - 训练时间/epoch: 21.3652 sec, 训练时间/batch: 0.2126 sec, 推理时间: 0.0001 sec, 显存占用: 3515.64 MB, FLOPs: 9
# epoch: 38, train_loss: 0.8854, train_acc: 81.79, train_fscore: 81.71, test_loss: 1.2887, test_acc: 72.15, test_fscore: 72.1, epoch_time: 21.32 sec
#   计算开销指标 - 训练时间/epoch: 21.3239 sec, 训练时间/batch: 0.2107 sec, 推理时间: 0.0001 sec, 显存占用: 3511.12 MB, FLOPs: 9
# epoch: 39, train_loss: 0.8744, train_acc: 82.72, train_fscore: 82.63, test_loss: 1.2842, test_acc: 72.21, test_fscore: 72.23, epoch_time: 18.76 sec
#   计算开销指标 - 训练时间/epoch: 18.7565 sec, 训练时间/batch: 0.2164 sec, 推理时间: 0.0001 sec, 显存占用: 3517.84 MB, FLOPs: 9
# epoch: 40, train_loss: 0.8773, train_acc: 81.82, train_fscore: 81.73, test_loss: 1.2757, test_acc: 72.64, test_fscore: 72.71, epoch_time: 18.29 sec
#   计算开销指标 - 训练时间/epoch: 18.2933 sec, 训练时间/batch: 0.2050 sec, 推理时间: 0.0001 sec, 显存占用: 3517.93 MB, FLOPs: 9
# epoch: 41, train_loss: 0.8637, train_acc: 82.43, train_fscore: 82.36, test_loss: 1.263, test_acc: 72.83, test_fscore: 72.95, epoch_time: 19.62 sec
#   计算开销指标 - 训练时间/epoch: 19.6233 sec, 训练时间/batch: 0.2089 sec, 推理时间: 0.0001 sec, 显存占用: 2423.58 MB, FLOPs: 9
# epoch: 42, train_loss: 0.8532, train_acc: 82.32, train_fscore: 82.25, test_loss: 1.2664, test_acc: 72.89, test_fscore: 72.87, epoch_time: 18.87 sec
#   计算开销指标 - 训练时间/epoch: 18.8673 sec, 训练时间/batch: 0.2193 sec, 推理时间: 0.0001 sec, 显存占用: 2428.56 MB, FLOPs: 9
# epoch: 43, train_loss: 0.8466, train_acc: 83.2, train_fscore: 83.13, test_loss: 1.2668, test_acc: 72.46, test_fscore: 72.54, epoch_time: 19.31 sec
#   计算开销指标 - 训练时间/epoch: 19.3112 sec, 训练时间/batch: 0.2108 sec, 推理时间: 0.0001 sec, 显存占用: 3517.67 MB, FLOPs: 9
# epoch: 44, train_loss: 0.8259, train_acc: 83.39, train_fscore: 83.27, test_loss: 1.2715, test_acc: 73.14, test_fscore: 73.12, epoch_time: 20.35 sec
#   计算开销指标 - 训练时间/epoch: 20.3487 sec, 训练时间/batch: 0.2071 sec, 推理时间: 0.0001 sec, 显存占用: 3519.79 MB, FLOPs: 9
# epoch: 45, train_loss: 0.83, train_acc: 83.41, train_fscore: 83.36, test_loss: 1.2882, test_acc: 72.95, test_fscore: 72.94, epoch_time: 20.80 sec
#   计算开销指标 - 训练时间/epoch: 20.8025 sec, 训练时间/batch: 0.2099 sec, 推理时间: 0.0001 sec, 显存占用: 2434.25 MB, FLOPs: 9
# epoch: 46, train_loss: 0.8246, train_acc: 83.22, train_fscore: 83.11, test_loss: 1.2297, test_acc: 73.69, test_fscore: 73.86, epoch_time: 20.21 sec
#   计算开销指标 - 训练时间/epoch: 20.2073 sec, 训练时间/batch: 0.2111 sec, 推理时间: 0.0001 sec, 显存占用: 3522.84 MB, FLOPs: 9
# epoch: 47, train_loss: 0.8169, train_acc: 83.94, train_fscore: 83.9, test_loss: 1.2806, test_acc: 72.15, test_fscore: 72.1, epoch_time: 19.36 sec
#   计算开销指标 - 训练时间/epoch: 19.3639 sec, 训练时间/batch: 0.2158 sec, 推理时间: 0.0001 sec, 显存占用: 2423.66 MB, FLOPs: 9
# epoch: 48, train_loss: 0.8106, train_acc: 83.44, train_fscore: 83.34, test_loss: 1.2618, test_acc: 73.94, test_fscore: 74.0, epoch_time: 18.36 sec
#   计算开销指标 - 训练时间/epoch: 18.3648 sec, 训练时间/batch: 0.2092 sec, 推理时间: 0.0001 sec, 显存占用: 3510.76 MB, FLOPs: 9
# epoch: 49, train_loss: 0.7911, train_acc: 84.3, train_fscore: 84.24, test_loss: 1.2413, test_acc: 74.12, test_fscore: 74.16, epoch_time: 19.86 sec
#   计算开销指标 - 训练时间/epoch: 19.8583 sec, 训练时间/batch: 0.2095 sec, 推理时间: 0.0001 sec, 显存占用: 2417.13 MB, FLOPs: 9
# epoch: 50, train_loss: 0.7872, train_acc: 84.37, train_fscore: 84.3, test_loss: 1.2576, test_acc: 73.57, test_fscore: 73.55, epoch_time: 18.80 sec
#   计算开销指标 - 训练时间/epoch: 18.8041 sec, 训练时间/batch: 0.1993 sec, 推理时间: 0.0001 sec, 显存占用: 2418.85 MB, FLOPs: 9
# epoch: 51, train_loss: 0.7797, train_acc: 84.6, train_fscore: 84.5, test_loss: 1.2515, test_acc: 74.06, test_fscore: 74.25, epoch_time: 18.73 sec
#   计算开销指标 - 训练时间/epoch: 18.7319 sec, 训练时间/batch: 0.2079 sec, 推理时间: 0.0001 sec, 显存占用: 3508.76 MB, FLOPs: 9
# epoch: 52, train_loss: 0.7817, train_acc: 84.41, train_fscore: 84.36, test_loss: 1.2743, test_acc: 73.38, test_fscore: 73.09, epoch_time: 19.91 sec
#   计算开销指标 - 训练时间/epoch: 19.9149 sec, 训练时间/batch: 0.2000 sec, 推理时间: 0.0001 sec, 显存占用: 2422.72 MB, FLOPs: 9
# epoch: 53, train_loss: 0.7721, train_acc: 84.8, train_fscore: 84.71, test_loss: 1.2242, test_acc: 73.81, test_fscore: 73.97, epoch_time: 18.82 sec
#   计算开销指标 - 训练时间/epoch: 18.8218 sec, 训练时间/batch: 0.2198 sec, 推理时间: 0.0001 sec, 显存占用: 3511.68 MB, FLOPs: 9
# epoch: 54, train_loss: 0.7635, train_acc: 85.08, train_fscore: 85.01, test_loss: 1.2182, test_acc: 74.25, test_fscore: 74.23, epoch_time: 20.02 sec
#   计算开销指标 - 训练时间/epoch: 20.0210 sec, 训练时间/batch: 0.2072 sec, 推理时间: 0.0001 sec, 显存占用: 3507.45 MB, FLOPs: 9
# epoch: 55, train_loss: 0.75, train_acc: 85.18, train_fscore: 85.09, test_loss: 1.2912, test_acc: 73.69, test_fscore: 73.75, epoch_time: 21.42 sec
#   计算开销指标 - 训练时间/epoch: 21.4171 sec, 训练时间/batch: 0.2226 sec, 推理时间: 0.0001 sec, 显存占用: 3506.07 MB, FLOPs: 9
# epoch: 56, train_loss: 0.75, train_acc: 84.89, train_fscore: 84.85, test_loss: 1.2273, test_acc: 73.2, test_fscore: 72.92, epoch_time: 19.02 sec
#   计算开销指标 - 训练时间/epoch: 19.0155 sec, 训练时间/batch: 0.2013 sec, 推理时间: 0.0001 sec, 显存占用: 3510.01 MB, FLOPs: 9
# epoch: 57, train_loss: 0.7464, train_acc: 85.39, train_fscore: 85.29, test_loss: 1.2204, test_acc: 74.55, test_fscore: 74.79, epoch_time: 19.89 sec
#   计算开销指标 - 训练时间/epoch: 19.8924 sec, 训练时间/batch: 0.2126 sec, 推理时间: 0.0001 sec, 显存占用: 3510.54 MB, FLOPs: 9
# epoch: 58, train_loss: 0.7294, train_acc: 85.27, train_fscore: 85.18, test_loss: 1.2999, test_acc: 74.06, test_fscore: 73.78, epoch_time: 19.21 sec
#   计算开销指标 - 训练时间/epoch: 19.2060 sec, 训练时间/batch: 0.2041 sec, 推理时间: 0.0001 sec, 显存占用: 2426.34 MB, FLOPs: 9
# epoch: 59, train_loss: 0.7375, train_acc: 84.99, train_fscore: 84.94, test_loss: 1.1866, test_acc: 74.31, test_fscore: 74.52, epoch_time: 22.43 sec
#   计算开销指标 - 训练时间/epoch: 22.4291 sec, 训练时间/batch: 0.2107 sec, 推理时间: 0.0001 sec, 显存占用: 3515.33 MB, FLOPs: 9
# epoch: 60, train_loss: 0.7192, train_acc: 85.58, train_fscore: 85.49, test_loss: 1.2535, test_acc: 73.63, test_fscore: 73.56, epoch_time: 19.84 sec
#   计算开销指标 - 训练时间/epoch: 19.8435 sec, 训练时间/batch: 0.1958 sec, 推理时间: 0.0001 sec, 显存占用: 3513.24 MB, FLOPs: 9
# epoch: 61, train_loss: 0.971, train_acc: 78.33, train_fscore: 78.28, test_loss: 1.2085, test_acc: 75.48, test_fscore: 75.67, epoch_time: 18.86 sec
#   计算开销指标 - 训练时间/epoch: 18.8607 sec, 训练时间/batch: 0.2357 sec, 推理时间: 0.0001 sec, 显存占用: 3514.64 MB, FLOPs: 9

# ================================================================================
# Model Performance:
# Best_Test-FScore-epoch_index: 61
# Best_Test_F-Score: 75.67
#               precision    recall  f1-score   support

#            0     0.5838    0.7500    0.6565     144.0
#            1     0.8326    0.8122    0.8223     245.0
#            2     0.7344    0.7995    0.7656     384.0
#            3     0.6952    0.7647    0.7283     170.0
#            4     0.8975    0.7324    0.8066     299.0
#            5     0.7486    0.6877    0.7168     381.0

#     accuracy                         0.7548    1623.0
#    macro avg     0.7487    0.7578    0.7494    1623.0
# weighted avg     0.7652    0.7548    0.7567    1623.0
# train.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np, argparse, time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset, DailyDialogDataset
from model import MaskedNLLLoss, Transformer_Based_Model
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import random
# from vision import TwoD_Tsne, sampleCount, confuPLT

# 用于计算FLOPs的库
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("警告: thop库未安装，FLOPs计算将使用简化方法。安装命令: pip install thop")

seed = 2094
print('seed = {}'.format(seed))

def seed_everything():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_everything()

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False, windows=5):
    trainset = MELDDataset('data/meld_multimodal_features.pkl', windows=windows)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset('data/meld_multimodal_features.pkl', train=False, windows=windows)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             # 随机样本可视化注意力系数时使用
                             # shuffle=True,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=True, windows=5):
    trainset = IEMOCAPDataset("data/iemocap_multimodal_features.pkl", windows=windows)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset("data/iemocap_multimodal_features.pkl", train=False, windows=windows)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             # 随机样本可视化注意力系数时使用
                             # shuffle=True,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader

def get_DailyDialog_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=True, windows=5):
    trainset = DailyDialogDataset("data/dd_multimodal_features.pkl", windows=windows)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = DailyDialogDataset("data/dd_multimodal_features.pkl", train=False, windows=windows)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             # 随机样本可视化注意力系数时使用
                             # shuffle=True,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader

# model, loss_function, kl_loss, train_loader, e, optimizer, True
def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False, dataset=None, warm_up_epochs=60):
    losses, preds, labels, masks, all_transformer_outs, umasks = [], [], [], [], [], []
    warm_up = 1 if not train or epoch < warm_up_epochs else 0

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    # 用于记录时间指标
    batch_times = []
    inference_times = []
    total_samples = 0  # 用于计算每个样本的平均推理时间
    
    # 重置显存统计
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    for data in dataloader:
        batch_start_time = time.time()
        if train:
            optimizer.zero_grad()

        if  dataset=="DailyDialog":
            textf, qmask, umask, label, Self_semantic_adj, Cross_semantic_adj, Semantic_adj\
                    = [d.cuda() for d in data] if cuda else data
            visuf = None
            acouf = None
        else:
            textf, visuf, acouf, qmask, umask, label, Self_semantic_adj, Cross_semantic_adj, Semantic_adj\
                = [d.cuda() for d in data] if cuda else data

        umasks.append(umask)
        qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        
        # 记录batch中的样本数量（有效utterance数量）
        batch_sample_count = umask.sum().item()
        total_samples += batch_sample_count
        
        # 记录前向推理时间
        if not train:
            inference_start = time.time()
        
        log_prob, all_log_prob, all_prob, all_final_out, update_flag = \
            model(textf, visuf, acouf, umask, qmask, lengths, Self_semantic_adj, Cross_semantic_adj, Semantic_adj, label=label if train else None, warm_up=warm_up)
        
        if not train:
            inference_time = time.time() - inference_start
            # 记录每个样本的平均推理时间（秒/样本）
            # 确保batch_sample_count > 0，避免除零错误
            if batch_sample_count > 0:
                inference_times.append(inference_time / batch_sample_count)
            else:
                # 如果batch_sample_count为0，使用batch_size作为fallback
                batch_size_actual = textf.size(0) if textf is not None else 1
                if batch_size_actual > 0:
                    inference_times.append(inference_time / batch_size_actual)
        lp_all = all_log_prob.view(-1, all_log_prob.size()[-1])
        labels_ = label.view(-1)
        umask_ = umask.view(-1)

        #融合损失
        fusion_loss = loss_function(lp_all, labels_, umask_)
        if dataset=="DailyDialog":
            loss = fusion_loss
            t_loss = loss_function(log_prob[0].view(-1, log_prob[0].size()[-1]), labels_, umask_)
        else:
            # 单模态损失
            t_loss = loss_function(log_prob[0].view(-1, log_prob[0].size()[-1]), labels_, umask_)
            a_loss = loss_function(log_prob[1].view(-1, log_prob[1].size()[-1]), labels_, umask_)
            v_loss = loss_function(log_prob[2].view(-1, log_prob[2].size()[-1]), labels_, umask_)
            #总损失
            loss = fusion_loss + t_loss * (t_loss / 10) + a_loss * (a_loss / 10) + v_loss * (v_loss / 10)

        lp_ = all_prob.view(-1, all_prob.size()[-1])
        pred_ = torch.argmax(lp_, 1)
        preds.append(pred_.data.cpu().numpy())
        all_transformer_outs.append(all_final_out)
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask_.cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            optimizer.step()
        
        # 记录每个batch的时间
        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
    
    # 计算显存峰值（MB）
    peak_memory_mb = 0
    if cuda:
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory_mb = peak_memory_bytes / (1024 ** 2)  # 转换为MB
    
    if preds!=[]:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), [], [], 0, 0, 0

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
    
    # 计算平均时间指标
    avg_batch_time = np.mean(batch_times) if batch_times else 0
    # 平均推理时间（秒/样本）
    if inference_times:
        avg_inference_time = np.mean(inference_times)
    else:
        # 如果没有记录到推理时间，尝试从batch时间估算
        # 这通常发生在训练模式下，或者数据有问题时
        avg_inference_time = 0
        if not train and batch_times:
            # 在评估模式下，如果没有单独的推理时间记录，使用batch时间估算
            # 假设batch时间主要包含推理时间
            avg_batch_time_for_inference = np.mean(batch_times)
            # 估算每个样本的推理时间（需要知道平均batch大小）
            if total_samples > 0:
                avg_inference_time = avg_batch_time_for_inference / (total_samples / len(batch_times)) if len(batch_times) > 0 else 0
    
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, all_transformer_outs, umasks, avg_batch_time, avg_inference_time, peak_memory_mb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.000068, metavar='LR', help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--l2', type=float, default=0.00005, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=512, metavar='hidden_dim', help='output hidden size')
    parser.add_argument('--n_head', type=int, default=8, metavar='n_head', help='number of heads')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--windows', type=int, default=20, help='number of windows')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')
    parser.add_argument('--save_model_path', default='./IEMOCAP', type=str, help='模型输出路径')
    # 默认启用模态丢弃，如果指定 --disable_modality_dropout 则禁用
    parser.add_argument('--disable_modality_dropout', action='store_true', default=False, help='disable modality dropout')
    parser.add_argument('--q_base', type=float, default=0.3, help='q_base parameter')
    parser.add_argument('--lam', type=float, default=0.9, help='lam parameter')
    parser.add_argument('--p_exe', type=float, default=0.5, help='p_exe parameter')
    parser.add_argument('--warm_up_epochs', type=int, default=60, help='warm up epochs')
    args = parser.parse_args()
    print(args)
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')
    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    feat2dim = {'IS10':1582, 'denseface':342, 'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024
    n_speakers = 9 if args.Dataset == 'MELD' else 2
    n_classes = 6 if args.Dataset=='IEMOCAP' else 7

    model = Transformer_Based_Model(args.Dataset, D_text, D_visual, D_audio, args.n_head,
                                    n_classes=n_classes,
                                    hidden_dim=args.hidden_dim,
                                    n_speakers=n_speakers,
                                    dropout=args.dropout,
                                    use_adam_drop=not args.disable_modality_dropout,
                                    q_base=args.q_base,
                                    lam=args.lam,
                                    p_exe=args.p_exe)


    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters: {}'.format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('training parameters: {}'.format(total_trainable_params))
    if torch.cuda.device_count()>0:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    if args.Dataset == 'MELD':
        loss_weights = torch.FloatTensor([1/0.481226,
                                          1/0.107663,
                                          1/0.191571,
                                          1/0.079669,
                                          1/0.154023,
                                          1/0.026054,
                                          1/0.132184
                                            ])
        loss_function = MaskedNLLLoss()
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                     batch_size=batch_size,
                                                                    num_workers=0, windows=args.windows)
    elif args.Dataset == 'IEMOCAP':
        # 这个权重是每一类的样本数与总样本数的比例的倒数，这样定义loss的作用是对样本不均衡的类别给予更高的权重
        loss_weights = torch.FloatTensor([1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668])
        loss_function = MaskedNLLLoss(loss_weights.cuda())
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=0, windows=args.windows)

    else:
        loss_weights = torch.FloatTensor([1 / 0.0017,
                                      1 / 0.0034,
                                      1 / 0.1251,
                                      1 / 0.831,
                                      1 / 0.0099,
                                      1 / 0.0177,
                                      1 / 0.0112])
        loss_function = MaskedNLLLoss()
        train_loader, valid_loader, test_loader = get_DailyDialog_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=0, windows=args.windows)

    best_fscore, best_loss, best_label, best_pred, best_mask, best_feature, best_umasks= None, None, None, None, None, None, None
    all_valid_fscore, all_fscore, all_acc, all_loss = [], [], [], []
    
    # 用于记录所有epoch的指标
    all_epoch_train_times = []
    all_epoch_batch_times = []
    all_epoch_inference_times = []
    all_epoch_memory_usage = []
    all_epoch_flops = []
    
    # 计算模型FLOPs（仅计算一次，使用第一个batch的样本）
    model_flops = 0
    model.eval()
    with torch.no_grad():
        try:
            # 获取一个样本用于FLOPs计算
            sample_data = next(iter(train_loader))
            if args.Dataset == "DailyDialog":
                if cuda:
                    textf, qmask, umask, label, Self_semantic_adj, Cross_semantic_adj, Semantic_adj = [d.cuda() for d in sample_data]
                else:
                    textf, qmask, umask, label, Self_semantic_adj, Cross_semantic_adj, Semantic_adj = sample_data
                visuf = None
                acouf = None
            else:
                if cuda:
                    textf, visuf, acouf, qmask, umask, label, Self_semantic_adj, Cross_semantic_adj, Semantic_adj = [d.cuda() for d in sample_data]
                else:
                    textf, visuf, acouf, qmask, umask, label, Self_semantic_adj, Cross_semantic_adj, Semantic_adj = sample_data
            qmask = qmask.permute(1, 0, 2)
            lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
            
            # 先计算总参数数量，用于验证
            total_params = sum(p.numel() for p in model.parameters())
            print(f"模型总参数数量: {total_params:,}")
            
            if THOP_AVAILABLE:
                try:
                    flops, params = profile(model, inputs=(textf, visuf, acouf, umask, qmask, lengths, Self_semantic_adj, Cross_semantic_adj, Semantic_adj, None, 1), verbose=False)
                    model_flops = flops
                    print(f"使用thop计算的FLOPs: {model_flops:,} ({model_flops/1e9:.3f}G)")
                except Exception as e:
                    print(f"警告: thop计算FLOPs失败: {e}")
                    # 使用参数数量估算（更准确的估算方法）
                    # 对于Transformer模型，每个参数在前向传播中约对应2-4次浮点运算
                    # 考虑到矩阵乘法和注意力机制，使用更合理的估算
                    model_flops = total_params * 2  # 保守估算：每个参数2次FLOPs
                    print(f"使用参数数量估算FLOPs: {model_flops:,} ({model_flops/1e9:.3f}G)")
            else:
                # 简化方法：使用参数数量估算
                # 对于Transformer模型，每个参数在前向传播中约对应2-4次浮点运算
                model_flops = total_params * 2  # 保守估算：每个参数2次FLOPs
                print(f"使用参数数量估算FLOPs: {model_flops:,} ({model_flops/1e9:.3f}G)")
        except Exception as e:
            print(f"警告: FLOPs计算失败: {e}")
            import traceback
            traceback.print_exc()
            total_params = sum(p.numel() for p in model.parameters())
            # 使用更保守的估算
            model_flops = total_params * 2
            print(f"使用fallback方法估算FLOPs: {model_flops:,} ({model_flops/1e9:.3f}G)")
    
    # 验证FLOPs值是否合理
    if model_flops <= 0 or model_flops < 1000:
        print(f"警告: FLOPs值异常 ({model_flops})，使用参数数量重新估算")
        total_params = sum(p.numel() for p in model.parameters())
        model_flops = total_params * 2
        print(f"重新估算的FLOPs: {model_flops:,} ({model_flops/1e9:.3f}G)")
    
    for e in range(n_epochs):
        epoch_start_time = time.time()
        train_loss, train_acc, _, _, _, train_fscore, _, _, train_batch_time, _, train_memory = train_or_eval_model(model, loss_function, train_loader, e, optimizer, True, dataset=args.Dataset, warm_up_epochs=args.warm_up_epochs)
        valid_loss, valid_acc, _, _, _, valid_fscore, _, _, _, valid_inference_time, valid_memory = train_or_eval_model(model, loss_function, valid_loader, e, train=False, dataset=args.Dataset, warm_up_epochs=args.warm_up_epochs)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, all_transformer_outs, umasks, _, test_inference_time, test_memory = train_or_eval_model(model, loss_function, test_loader, e, train=False, dataset=args.Dataset, warm_up_epochs=args.warm_up_epochs)
        all_valid_fscore.append(test_fscore)
        all_fscore.append(test_fscore)
        
        epoch_time = time.time() - epoch_start_time
        all_epoch_train_times.append(epoch_time)
        all_epoch_batch_times.append(train_batch_time)
        all_epoch_inference_times.append(test_inference_time)
        all_epoch_memory_usage.append(max(train_memory, test_memory))

        if best_fscore == None or best_fscore < test_fscore:
            best_fscore = test_fscore
            best_label, best_pred, best_mask, best_feature = test_label, test_pred, test_mask, all_transformer_outs
        
        # 格式化FLOPs输出
        if model_flops > 0:
            if THOP_AVAILABLE:
                try:
                    flops_str = clever_format([model_flops], "%.3f")[0]
                except:
                    # 如果clever_format失败，手动格式化
                    if model_flops >= 1e9:
                        flops_str = f"{model_flops / 1e9:.3f}G"
                    elif model_flops >= 1e6:
                        flops_str = f"{model_flops / 1e6:.3f}M"
                    elif model_flops >= 1e3:
                        flops_str = f"{model_flops / 1e3:.3f}K"
                    else:
                        flops_str = f"{model_flops:.0f}"
            else:
                # 手动格式化
                if model_flops >= 1e9:
                    flops_str = f"{model_flops / 1e9:.3f}G"
                elif model_flops >= 1e6:
                    flops_str = f"{model_flops / 1e6:.3f}M"
                elif model_flops >= 1e3:
                    flops_str = f"{model_flops / 1e3:.3f}K"
                else:
                    flops_str = f"{model_flops:.0f}"
        else:
            flops_str = "N/A"
        
        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, epoch_time: {:.2f} sec'.\
                format(e+1, train_loss, train_acc, train_fscore, test_loss, test_acc, test_fscore, epoch_time))
        print('  计算开销指标 - 训练时间/epoch: {:.4f} sec, 训练时间/batch: {:.4f} sec, 推理时间: {:.4f} sec, 显存占用: {:.2f} MB, FLOPs: {}'.\
                format(epoch_time, train_batch_time, test_inference_time, max(train_memory, test_memory), flops_str))

    save_path = os.path.join("./"+args.Dataset, "bestModel.pth")
    torch.save(model, save_path)
    print('\n' + '='*80)
    print('Model Performance:')
    print('Best_Test-FScore-epoch_index: {}'.format(all_fscore.index(max(all_fscore))+1))
    print('Best_Test_F-Score: {}'.format(max(all_fscore)))
    print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4, zero_division=0))
    
    # 输出最终的计算开销统计
    print('\n' + '='*80)
    print('最终计算开销统计:')
    print('-'*80)
    avg_epoch_time = np.mean(all_epoch_train_times)
    avg_batch_time = np.mean(all_epoch_batch_times)
    avg_inference_time = np.mean(all_epoch_inference_times)
    avg_memory = np.mean(all_epoch_memory_usage)
    max_memory = np.max(all_epoch_memory_usage)
    
    # 格式化FLOPs输出
    if model_flops > 0:
        if THOP_AVAILABLE:
            try:
                flops_str = clever_format([model_flops], "%.3f")[0]
            except:
                # 如果clever_format失败，手动格式化
                if model_flops >= 1e9:
                    flops_str = f"{model_flops / 1e9:.3f}G"
                elif model_flops >= 1e6:
                    flops_str = f"{model_flops / 1e6:.3f}M"
                elif model_flops >= 1e3:
                    flops_str = f"{model_flops / 1e3:.3f}K"
                else:
                    flops_str = f"{model_flops:.0f}"
        else:
            # 手动格式化
            if model_flops >= 1e9:
                flops_str = f"{model_flops / 1e9:.3f}G"
            elif model_flops >= 1e6:
                flops_str = f"{model_flops / 1e6:.3f}M"
            elif model_flops >= 1e3:
                flops_str = f"{model_flops / 1e3:.3f}K"
            else:
                flops_str = f"{model_flops:.0f}"
    else:
        flops_str = "N/A"
    
    print(f'1. 训练时间:')
    print(f'   - 平均每个epoch: {avg_epoch_time:.4f} 秒')
    print(f'   - 平均每个batch: {avg_batch_time:.4f} 秒')
    print(f'   - 总训练时间: {sum(all_epoch_train_times):.2f} 秒 ({sum(all_epoch_train_times)/60:.2f} 分钟)')
    print(f'\n2. 推理时间:')
    print(f'   - 平均推理时间: {avg_inference_time:.4f} 秒/样本')
    print(f'\n3. 显存占用:')
    print(f'   - 平均显存占用: {avg_memory:.2f} MB')
    print(f'   - 峰值显存占用: {max_memory:.2f} MB')
    print(f'\n4. 计算量 (FLOPs):')
    print(f'   - 模型FLOPs: {flops_str}')
    print('='*80)
    
    try:
        confuPLT(confusion_matrix(best_label, best_pred, sample_weight=best_mask).astype(int), args.Dataset)
    except:
        pass
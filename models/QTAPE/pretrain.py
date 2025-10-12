import os
current_dir = os.getcwd()
parent_parent_dir = os.path.dirname(os.path.dirname(current_dir))
target_folder_path = os.path.join(parent_parent_dir, "dataset_generation")
print(target_folder_path)


import utils
import pandas as pd 
import numpy as np
from decoder import Decoder
import embedding 
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def main(qubits_num):
    utils.fix_seed(2024)
    try:
        dataset_path = "/dataset_results/heisenberg_1d/n{samples_num}|X(coupling, meas{shots})_y(energy,entropy,corrs)_q{q}.csv".format(samples_num=samples_num, shots=shots_num, q=qubits_num)
        df = pd.read_csv(target_folder_path + dataset_path)
    except:
        raise FileNotFoundError("Dataset not found")

    meas_records = np.array([utils.read_matrix_v2(x) for x in df['measurement_samples'].values]) # shape (samples_num, shots_num * qubits_num)
    conditions = np.array([utils.read_matrix_v2(x) for x in df['coupling_matrix'].values]) # shape (samples_num, qubits_num * qubits_num)

    meas_records = meas_records.reshape(-1, shots_num, qubits_num) # shape (samples_num, shots_num, qubits_num)
    meas_records = meas_records.reshape(-1, qubits_num) # shape (samples_num * shots_num, qubits_num)

    new_conditions = []
    for i in range(samples_num):
        for _ in range(shots_num):
            new_conditions.append(conditions[i])
    new_conditions = np.array(new_conditions) # shape (samples_num * shots_num, qubits_num * qubits_num)

    # GPU设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    embedding_dim = 512
    seq_len = qubits_num + 1 # +1 for the CLS token, and the CLS token is at the first position of the sequence

    batch_conditions, batch_measures = [], []
    sample_idx = np.random.choice(range(samples_num*shots_num), batch_size, replace=False)
    batch_measures = meas_records[sample_idx]   # shape (batch_size, qubits_num)
    batch_conditions = new_conditions[sample_idx]   # shape (batch_size, qubits_num * qubits_num)
    #？？？？？？？？？？？？？这里为什么只用1个batch进行训练？？？？？？？？？？？？？？？
    all_embeddings, token_embedding = embedding.get_embedding(batch_size, seq_len, embedding_dim, batch_measures, batch_conditions)
    #shape: (batch_size, seq_len, embedding_dim)


    # 将embeddings和labels从计算图中分离，避免DataLoader错误
    embeddings = all_embeddings.detach()
    labels = F.softmax(token_embedding, dim=-1).detach()
    dataset = TensorDataset(embeddings, labels)
    # 优化DataLoader: pin_memory加速CPU到GPU传输, num_workers并行加载数据
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, 
                          pin_memory=True if device.type == 'cuda' else False,
                          num_workers=4)
    #？？？？？？？？？？？为什么数据集这样安排，特征是位置 实验条件 和测量结果的嵌入，标签是测量结果的概率分布？？？？？？？？？？

    # 初始化模型
    decoder = Decoder(embedding_dim, seq_len, embedding_dim, ffn_hidden=128, n_head=8, n_layers=qubits_num, drop_prob=0.1, device=device.type)
    decoder = decoder.to(device)
    
    # 多GPU并行训练
    if num_gpus > 1:
        print(f"Using DataParallel with {num_gpus} GPUs")
        decoder = nn.DataParallel(decoder)
    
    # output: (batch_size, seq_len, embedding_dim) 最后一维是词表大小，概率分布形式
    criterion = nn.KLDivLoss(reduction='batchmean') # 对批量和序列长度维度求平均，保留嵌入维度的差异。
    optimizer = optim.Adam(decoder.parameters(), lr=0.001)

    # Early stopping parameters
    epochs = 1000
    patience = 10  # 连续10个epoch没有改善就停止
    min_delta = 1e-4  # 最小改善阈值
    best_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    print(f"Training with early stopping: patience={patience}, min_delta={min_delta}")
    
    for epoch in tqdm(range(epochs)):
        decoder.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            # 将数据移动到GPU
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 创建mask并移动到GPU
            # mask shape需要是(batch_size, 1, seq_len, seq_len)以匹配multi-head attention
            actual_batch_size = inputs.size(0)
            trg_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))  # 下三角矩阵
            trg_mask = trg_mask.unsqueeze(0).unsqueeze(0)  # 添加batch和head维度 -> (1, 1, seq_len, seq_len)
            trg_mask = trg_mask.expand(actual_batch_size, 1, seq_len, seq_len)  # 扩展到batch size
            
            outputs = decoder(inputs, trg_mask) # shape: (batch_size, seq_len, embedding_dim)概率分布

            loss = criterion(outputs.contiguous().view(-1, embeddings.size(-1)), 
                           targets.contiguous().view(-1, embeddings.size(-1)))
            # 展开成二维张量（batch_size * seq_len, embedding_dim），逐个位置计算KL散度
            
            # 使用标准的backward而不是手动处理梯度
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Calculate average loss for this epoch
        avg_loss = running_loss / len(dataloader)
        
        # Print progress every epoch (or change to every N epochs if needed)
        if (epoch+1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.8f}, Best Loss: {best_loss:.8f}')
        
        # Early stopping check
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model (处理DataParallel的情况)
            best_model_path = "save/pretrain/best_pretrain_q{}_s{}_sn{}.pt".format(qubits_num, shots_num, samples_num)
            model_to_save = decoder.module if isinstance(decoder, nn.DataParallel) else decoder
            torch.save(model_to_save.state_dict(), best_model_path)
            print(f'*** New best model saved at epoch {best_epoch} with loss {best_loss:.8f} ***')
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter} epochs (patience: {patience})')
        
        # Check if we should stop
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered at epoch {epoch+1}')
            print(f'Best loss {best_loss:.8f} was at epoch {best_epoch}')
            break

    # save the final model (处理DataParallel的情况)
    final_epoch = epoch + 1
    pretrain_path = "save/pretrain/final_pretrain_q{}_s{}_sn{}_ep{}.pt".format(qubits_num, shots_num, samples_num, final_epoch)
    model_to_save = decoder.module if isinstance(decoder, nn.DataParallel) else decoder
    torch.save(model_to_save.state_dict(), pretrain_path)
    print(f"Final model saved at {pretrain_path}")
    print(f"Best model was saved at epoch {best_epoch} with loss {best_loss:.8f}")
    
if __name__ == "__main__":
    #qubits_list = [127] #[8, 10, 12]
    samples_num = 10
    shots_num_list = [3]
    qubits_num = 2 # L
    
    for shots_num in shots_num_list:
        batch_size = samples_num*shots_num
        main(qubits_num)
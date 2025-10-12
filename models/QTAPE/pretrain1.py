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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_dim = 512
    seq_len = qubits_num + 1 # +1 for the CLS token, and the CLS token is at the first position of the sequence

    # batch_conditions, batch_measures = [], []
    # sample_idx = np.random.choice(range(samples_num*shots_num), batch_size, replace=False)
    # batch_measures = meas_records[sample_idx]   # shape (batch_size, qubits_num)
    # batch_conditions = new_conditions[sample_idx]   # shape (batch_size, qubits_num * qubits_num)
    #？？？？？？？？？？？？？这里为什么只用1个batch进行训练？？？？？？？？？？？？？？？
    #all_embeddings, token_embedding = embedding.get_embedding(batch_size, seq_len, embedding_dim, batch_measures, batch_conditions)
    # shape: (batch_size, seq_len, embedding_dim)
    # 使用全部数据生成embeddings
    all_embeddings, token_embedding = embedding.get_embedding(
        samples_num * shots_num,  # 使用全部样本
        seq_len, 
        embedding_dim, 
        meas_records,  # 全部测量记录
        new_conditions  # 全部条件
    )

    embeddings = all_embeddings
    labels = F.softmax(token_embedding, dim=-1)
    dataset = TensorDataset(embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    #？？？？？？？？？？？为什么数据集这样安排，特征是位置 实验条件 和测量结果的嵌入，标签是测量结果的概率分布？？？？？？？？？？

    decoder = Decoder(embedding_dim, seq_len, embedding_dim, ffn_hidden=128, n_head=8, n_layers=qubits_num, drop_prob=0.1, device='cuda')
    # output: (batch_size, seq_len, embedding_dim) 最后一维是词表大小，概率分布形式
    criterion = nn.KLDivLoss(reduction='batchmean') # 对批量和序列长度维度求平均，保留嵌入维度的差异。
    optimizer = optim.Adam(decoder.parameters(), lr=0.001)

    epochs = 1000
    for epoch in tqdm(range(epochs)):
        decoder.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            trg_mask = torch.ones(8, all_embeddings.shape[1], all_embeddings.shape[1])
            trg_mask = torch.triu(trg_mask, diagonal=1)
            trg_mask = 1-trg_mask
            #trg_mask = trg_mask.masked_fill(trg_mask == 1, float(0))
            #?????????????????????????变为全0矩阵了，实际上应该是上三角部分为-inf，下三角部分为0的矩阵??????????????????????
            #trg_mask = trg_mask.to(device)
            outputs = decoder(inputs, trg_mask) # shape: (batch_size, seq_len, embedding_dim)概率分布

            loss = criterion(outputs.contiguous().view(-1, all_embeddings.size(-1)), targets.contiguous().view(-1, all_embeddings.size(-1)))
            # 展开成二维张量（batch_size * seq_len, embedding_dim），逐个位置计算KL散度
            #？？？？？？？？？？？？？？这里为什么要手动处理，而不是loss.backward()？？？？？？？？？？？？？？？
            gradients = torch.autograd.grad(loss, decoder.parameters(), retain_graph=True)
            for param, grad in zip(decoder.parameters(), gradients):
                param.grad = grad
            # loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.8f}')

    # save the decoder model
    pretrain_path = "save/pretrain_q{}_s{}_bs{}_ep{}.pt".format(qubits_num, shots_num, batch_size, epochs)
    torch.save(decoder.state_dict(), pretrain_path)
    print("Pretrained model saved at", pretrain_path)
    
if __name__ == "__main__":
    #qubits_list = [127] #[8, 10, 12]
    samples_num = 100
    shots_num_list = [64, 128, 256, 512]
    qubits_num = 127 # L
    batch_size = 64 #100
    for shots_num in shots_num_list:
        main(qubits_num)


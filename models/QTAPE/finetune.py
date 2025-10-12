import os
current_dir = os.getcwd()
parent_parent_dir = os.path.dirname(os.path.dirname(current_dir))
target_folder_path = os.path.join(parent_parent_dir, "dataset_generation")


import pandas as pd 
import numpy as np
from decoder import Decoder
import embedding 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import utils
import loss
import models
from tqdm import tqdm


def main(qubits_num, train_samples):#, test_samples):
    utils.fix_seed(2025)
    
    # GPU设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    try:
        finetune_path = "/dataset_results/heisenberg_1d/n{samples_num}|X(coupling, meas{shots})_y(energy,entropy,corrs)_q{q}.csv".format(samples_num=samples_num, shots=shots_num, q=qubits_num)
        df = pd.read_csv(target_folder_path + finetune_path)
    except:
        raise FileNotFoundError("Dataset not found")

    embedding_dim = 512
    hidden_dim = 128
    seq_len = qubits_num + 1
    #batch_size = samples_num
    meas_records = np.array([utils.read_matrix_v2(x) for x in df['measurement_samples'].values]).reshape(-1, qubits_num, shots_num) 
    # shape: (samples_num, qubits_num, shots_num)
    conditions = np.array([utils.read_matrix_v2(x) for x in df['coupling_matrix'].values])
    # shape: (samples_num, qubits_num*qubits_num)
    all_idx = np.random.choice(range(samples_num), batch_size, replace=False)
    batch_measures = meas_records[all_idx] # shape: (batch_size, qubits_num, shots_num)
    batch_conditions = conditions[all_idx] # shape: (batch_size, qubits_num*qubits_num)
    cls_token = torch.zeros((batch_size, shots_num, 1), dtype=torch.long) # shape: (batch_size, shots_num, 1)
    batch_measures = torch.cat((cls_token, torch.tensor(batch_measures).permute(0, 2, 1).long()), dim=2).permute(0, 2, 1).float()
    # shape: (batch_size, qubits_num+1, shots_num)
    y_approx_corr = torch.tensor([utils.read_matrix_v2(x) for x in df['approx_correlation'].values])
    y_exact_corr = torch.tensor([utils.read_matrix_v2(x) for x in df['exact_correlation'].values])
    #？？？？？？？？？？？？？？？？？这里为什么要用lstm来生成token_embedding？？？？？？？？？？？？？？？？？
    rnn = nn.LSTM(shots_num, embedding_dim, 1)
    # shape: (batch_size, qubits_num+1, embedding_dim)
    token_embedding_ft, _ = rnn(batch_measures)
    all_embedding = token_embedding_ft + embedding.get_embedding_ft(batch_size, seq_len, embedding_dim, batch_conditions)
    # shape: (batch_size, qubits_num+1, embedding_dim)  

    # 将embeddings从计算图中分离，避免DataLoader错误
    all_embedding = all_embedding.detach()
    y_approx_corr = y_approx_corr.detach()
    y_exact_corr = y_exact_corr.detach()
    
    #test_samples = int(samples_num * test_size)
    #train_samples = samples_num - test_samples
    #？？？？？？？？？？？？样本划分，是否取重合？？？？？？？？？？？？
    train_sample_idx = np.random.choice(range(samples_num), train_samples, replace=False)
    test_sample_idx = np.array([i for i in range(samples_num) if i not in train_sample_idx])
    #test_sample_idx = np.arange(10, 100, 1)
    #？？？？？？？？？？？？这里为什么要在近似关联函数上做微调，精确关联函数上做测试？？？？？？？？？？？？
    X_train = all_embedding[train_sample_idx]
    y_train = y_exact_corr[train_sample_idx]
    X_test = all_embedding[test_sample_idx]
    y_test = y_exact_corr[test_sample_idx]

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test) 
    decoder = Decoder(embedding_dim, seq_len, embedding_dim, ffn_hidden=128, n_head=8, n_layers=qubits_num, drop_prob=0.1, device=device.type)
    decoder = decoder.to(device)  # 将模型移动到GPU
    
    # 多GPU并行训练
    if num_gpus > 1:
        print(f"Using DataParallel with {num_gpus} GPUs")
        decoder = nn.DataParallel(decoder)
    
    try:
        # 处理DataParallel的情况
        model_to_load = decoder.module if isinstance(decoder, nn.DataParallel) else decoder
        model_to_load.load_state_dict(torch.load("save/pretrain/best_pretrain_q{}_s{}_sn{}.pt".format(qubits_num, shots_num, samples_num), weights_only=True))
        print("Pretrained model loaded.")
    except:
        raise FileNotFoundError("Pretrained model not found.")

    # supervised fine-tuning
    optimizer = optim.Adam(decoder.parameters(), lr=0.001)

    finetune_model = models.FinetuneDecoder(decoder, None, embedding_dim, hidden_dim, embedding_dim, qubits_num*qubits_num)
    finetune_model = finetune_model.to(device)  # 将微调模型移动到GPU
    # shape: (batch_size, qubits_num*qubits_num)
    
    # 优化DataLoader: pin_memory加速CPU到GPU传输, num_workers并行加载数据
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, 
                              pin_memory=True if device.type == 'cuda' else False,
                              num_workers=4)  # 创建数据加载器，使用合适的batch size和shuffle，确保每个epoch看到不同的数据组合
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             pin_memory=True if device.type == 'cuda' else False,
                             num_workers=4)
    
    # Early stopping parameters
    epochs = 1000
    patience = 10  # 连续10个epoch没有改善就停止
    min_delta = 1e-4  # 最小改善阈值
    best_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    print(f"Training with early stopping: patience={patience}, min_delta={min_delta}")
    
    for i in tqdm(range(epochs)):
        finetune_model.train()  # 设置为训练模式
        total_loss = 0.0
        for input, target in train_loader:  # 遍历训练数据的所有batch
            # 将数据移动到GPU
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            train_loss = 0.0
            optimizer.zero_grad()
            output = finetune_model(input)
            train_loss = loss.rmse_loss(output, target)
            # 使用标准的backward而不是手动处理梯度
            train_loss.backward()
            optimizer.step()
            total_loss += train_loss.item()
        
        # Calculate average loss for this epoch
        avg_loss = total_loss / len(train_loader)
        
        # Print progress every epoch
        if (i+1) % 1 == 0:
            print(f'Epoch [{i+1}/{epochs}], Loss: {avg_loss:.8f}, Best Loss: {best_loss:.8f}')
        
        # Early stopping check
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            best_epoch = i + 1
            patience_counter = 0
            # Save best model (处理DataParallel的情况)
            # best_model_path = "save/finetune/best_finetune_q{}_s{}_sn{}_trs{}_tes{}.pt".format(qubits_num, shots_num, samples_num, train_samples, test_samples)
            # model_to_save = finetune_model.module if isinstance(finetune_model, nn.DataParallel) else finetune_model
            # torch.save(model_to_save.state_dict(), best_model_path)
            print(f'*** New best model saved at epoch {best_epoch} with loss {best_loss:.8f} ***')
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter} epochs (patience: {patience})')
        
        # Check if we should stop
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered at epoch {i+1}')
            print(f'Best loss {best_loss:.8f} was at epoch {best_epoch}')
            break

    result = 0.0
    # evaluation
    finetune_model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for input, target in test_loader:
            # 将数据移动到GPU
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            output = finetune_model(input)
            test_loss = loss.rmse_loss(output, target)
        print("Test Loss: {}".format(test_loss.item()))
        result = test_loss.item()
    
    # 保存最终模型
    # final_epoch = i + 1
    # final_model_path = "save/finetune/final_finetune_q{}_s{}_sn{}_trs{}_tes{}_ep{}.pt".format(qubits_num, shots_num, samples_num, train_samples, test_samples, final_epoch)
    # model_to_save = finetune_model.module if isinstance(finetune_model, nn.DataParallel) else finetune_model
    # torch.save(model_to_save.state_dict(), final_model_path)
    # print(f"Final model saved at {final_model_path}")
    print(f"Best model was saved at epoch {best_epoch} with loss {best_loss:.8f}")
    
    return result

if __name__ == "__main__":
    qubits_list = [127] #[8, 10, 12]
    train_samples_list = [20, 50, 90]
    shots_num_list = [64,128,256,512]
    samples_num = 100
    batch_size = samples_num
    #qubits_num = 8
    #test_size = 0.8
     
    for qubits_num in qubits_list:
        for shots_num in shots_num_list:
            for train_samples in train_samples_list:
                test_samples = samples_num - train_samples
                tloss = main(qubits_num, train_samples) #, test_samples)
                print("qubits: {}, train_samples: {}, test loss: {}".format(qubits_num, train_samples, tloss))
                with open("results/heisenberg_1d_correlation_rmse.txt", "a") as f:
                    f.write("test loss: {}, qubits: {}, shots_num: {}, samples_num: {}, train_samples: {}, test_samples: {}\n".format(tloss, qubits_num, shots_num, samples_num, train_samples, test_samples))
                    f.close()
        
    


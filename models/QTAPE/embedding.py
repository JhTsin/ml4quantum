import torch
import torch.nn as nn
import torch.optim as optim
import math
from models import SimpleAutoencoder
from sklearn.preprocessing import MinMaxScaler

def get_token_embedding(batch_size, batch_measures, embedding_dim, num_embeddings=6):
    #num_embeddings：嵌入字典的大小，默认值为6（表示测量结果的可能取值数量，0-5），字典有点小，而且不含内在信息！！！
    embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
    cls_token = torch.zeros(batch_size, 1, dtype=torch.long)
    token_embedding = embedding_layer(torch.cat((cls_token, torch.tensor(batch_measures).long()), dim=1))
    return token_embedding  # shape: (batch_size, seq_len, embedding_dim) seq_len -> qubits_num + 1


def get_positional_embedding(batch_size, seq_len, embedding_dim):
    # same method as in "Attention is All You Need"
    positional_embedding = torch.zeros(seq_len, embedding_dim)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    # shape: (seq_len, 1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim)) 
    # shape: (embedding_dim/2,)
    positional_embedding[:, 0::2] = torch.sin(position * div_term) # broadcasting
    positional_embedding[:, 1::2] = torch.cos(position * div_term)
    # shape: (seq_len, embedding_dim)
    return positional_embedding  # shape: (seq_len, embedding_dim)

#??????????????????为什么要用autoencoder来做condition embedding？？？？
def get_condition_embedding(batch_size, batch_conditions, embedding_dim):
    # Instantiate the model
    ae = SimpleAutoencoder(input_dim=batch_conditions.shape[1], hidden_dim=48, embed_dim=embedding_dim)
    optimizer = optim.Adam(ae.parameters(), lr=0.001)
    loss_function = nn.MSELoss()  # Reconstruction loss

    epochs = 1000
    data_loader = torch.utils.data.DataLoader(torch.tensor(batch_conditions).float(), batch_size=batch_size, shuffle=True)
    # shape of data_loader: (num_batches, batch_size, num_conditions)
    # Example training loop
    for epoch in range(epochs):
        for inputs in data_loader:  #shape of inputs: (batch_size, num_conditions)
            optimizer.zero_grad()
            condition_embedding, reconstructed = ae(inputs)  
            loss = loss_function(reconstructed, inputs) 
            loss.backward()  
            optimizer.step()  
            # print('Epoch %d, Loss: %.4f' % (epoch, loss.item()))

    scaler = MinMaxScaler()
    condition_embedding_norm = scaler.fit_transform(condition_embedding.detach().numpy()) 
    # shape: (batch_size, embedding_dim)
    condition_embedding_norm = torch.tensor(condition_embedding_norm)
    return condition_embedding_norm # shape: (batch_size, embedding_dim)



def get_embedding(batch_size, seq_len, embedding_dim, batch_measures, batch_conditions):
    token_embedding = get_token_embedding(batch_size, batch_measures, embedding_dim) # shape: (batch_size, seq_len, embedding_dim)
    positional_embedding = get_positional_embedding(batch_size, seq_len, embedding_dim) # shape: (seq_len, embedding_dim)
    condition_embedding = get_condition_embedding(batch_size, batch_conditions, embedding_dim) # shape: (batch_size, embedding_dim)

    # broadcasting summation
    condition_embedding_expanded = condition_embedding.unsqueeze(1).expand(-1, seq_len, -1)
    positional_embedding_expanded = positional_embedding.unsqueeze(0).expand(batch_size, -1, -1)
    all_embeddings = token_embedding + positional_embedding_expanded + condition_embedding_expanded
    
    return all_embeddings, token_embedding


#为什么没有token embedding？？ft文件中另外加了
## finetune embedding
def get_embedding_ft(batch_size, seq_len, embedding_dim, batch_conditions):
    positional_embedding = get_positional_embedding(batch_size, seq_len, embedding_dim)
    condition_embedding = get_condition_embedding(batch_size, batch_conditions, embedding_dim)
    condition_embedding_expanded = condition_embedding.unsqueeze(1).expand(-1, seq_len, -1)
    positional_embedding_expanded = positional_embedding.unsqueeze(0).expand(batch_size, -1, -1)
    all_embeddings = positional_embedding_expanded + condition_embedding_expanded
    
    return all_embeddings
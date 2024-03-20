import torch
from torch.utils.data import TensorDataset, DataLoader

class NanoGPT_Dataset:
    def __init__(self, tokenizer, train_percent=0.9):
        t = torch.tensor(tokenizer.encode(tokenizer.get_data()), dtype=torch.long)
        train_num = int(train_percent * t.shape[0])

        self._train_data = t[:train_num]
        self._val_data = t[train_num:]
        print(f"TrainNum: {train_num}")
    
    def create_dataset(self, t, block_size):
        length = t.shape[0] - block_size
        tensor_x = torch.arange(start=0, end=length)
        tensor_y = torch.arange(start=1, end=length+1)

        b = torch.arange(block_size).view(1, -1)
        x_view = tensor_x.view(-1, 1) + b
        x_data = t[x_view]

        y_view = tensor_y.view(-1, 1) + b
        y_data = t[y_view]
        
        return TensorDataset(x_data, y_data)

    def create_data_loader(self, t, batch_size, block_size):
        dataset = self.create_dataset(t, block_size=block_size)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def get_train_val_dl(self, batch_size, block_size):
        torch.manual_seed(1337)
        train_dl = self.create_data_loader(self._train_data, batch_size=batch_size, block_size=block_size)
        val_dl = self.create_data_loader(self._val_data, batch_size=batch_size, block_size=block_size)
        return (train_dl, val_dl)

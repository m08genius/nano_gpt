import torch
import pickle
from typing import Optional, List


class ModelCheckpoint:
    def __init__(self, filepath, model_name: str, block_size: int, epoch_id: Optional[int] = None):
        if epoch_id is not None:
            assert epoch_id >= 1

        self._epoch_id = 0 if epoch_id is None else epoch_id

        self._template = "{}/Mn{}_Seq{}_Type{}_Epoch{}.{}"
        self._model_file_template = self._template.format(filepath, model_name, block_size, "MODEL", "{}", "pt")
        self._global_train_loss_template = self._template.format(filepath, model_name, block_size, "TRAIN_LOSS", "{}", "pkl")
        self._global_val_loss_template = self._template.format(filepath, model_name, block_size, "VAL_LOSS", "{}", "pkl")

        print(self._model_file_template)
        print(self._global_train_loss_template)
        print(self._global_val_loss_template)
    
    def load_model_params(self, model):
        if self._epoch_id == 0:
            return

        model.load_state_dict(torch.load(self._model_file_template.format(self._epoch_id)))
        return
    
    def load_loss(self):
        if self._epoch_id == 0:
            return ([], [])
            
        with open(self._global_train_loss_template.format(self._epoch_id), 'rb') as f:
            train_loss = pickle.load(f)

        with open(self._global_val_loss_template.format(self._epoch_id), 'rb') as f:
            val_loss = pickle.load(f)

        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        return (train_loss, val_loss)
    
    def update_epoch(self):
        self._epoch_id = self._epoch_id + 1
        return
    
    def get_epoch(self):
        return self._epoch_id
    
    def save(self, model, train_loss: List, val_loss: List):
        torch.save(model.state_dict(), self._model_file_template.format(self._epoch_id))
        with open(self._global_train_loss_template.format(self._epoch_id), "wb") as fp:
            pickle.dump(train_loss, fp)
        with open(self._global_val_loss_template.format(self._epoch_id), "wb") as fp:
            pickle.dump(val_loss, fp)

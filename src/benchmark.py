from typing import List, Callable, Any, Dict
from torch.utils.data import DataLoader
from datasets import load_dataset
import time

def dataset_process(dataset):
    return dataset["train"].select(range(100))

class Timer():
    
    def __init__(self):
        
        self._start = None
        
        self.take()
        
    def take(self):
        
        self._start = time.time()
        
    def stop(self):
        
        return  time.time() - self._start


class DatasetItemCallbackNLP():
    
    def __init__(self, tokenizer):
        
        self._tokenizer = tokenizer
        
    def process(self, batch: List[Dict[str, Any]]):
        
        texts = [x["text"] for x in batch]
        
        tokens = self._tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            truncation=True
        )
        
        del tokens["token_type_ids"]
        
        return tokens

class Benchmark():
    
    def __init__(
        self, dataset: str,
        batch_sizes: List[int],
        data_item_process_callback: Callable = None,
        dataset_process_callback: Callable = dataset_process,
    ):
        
        self._batch_sizes = batch_sizes
        self._data_item_process_callback = data_item_process_callback
        self._dataset_process_callback = dataset_process_callback
        self._dataset = load_dataset(dataset)
        
    def _setup_dataloader(self, batch_size: int):
        
        if self._dataset_process_callback is not None:
            dataset = self._dataset_process_callback(self._dataset)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._data_item_process_callback,
        )
    
    def excecute(self, model, name: str = None):
        
        durations = []
        
        name = str(model.__class__) if name is None else name
        
        print(name)
        
        for batch_size in self._batch_sizes:
            
            data_loader = self._setup_dataloader(batch_size)
            
            for batch in data_loader:
                
                timer = Timer()

                model(batch)
            
                durations.append(
                    {
                        "duration": timer.stop(),
                        "batch_size": batch_size,
                        "name": name
                    }
                )
            
            # Delete the last one, we can not be sure that we have a complete batch
            durations.pop()
            
        return durations
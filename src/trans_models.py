
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from benchmarks_models import BenchmarkModel


class TextClassification(BenchmarkModel):
    
    def __init__(
        self,
        model_name: str = "nateraw/bert-base-uncased-imdb",
        **kwargs
    ):
        
        self._tokenizer = AutoTokenizer.from_pretrained(
           model_name
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **kwargs
        )
        
        super().__init__(model, model_name.replace("/", "-"))
        
    def export_to_onnx(self, device: str,  dynamic_axis: bool = True):
        
        dummy_input = {
            "input_ids": torch.Tensor([[1, 2, 3]]).long(),
            "attention_mask": torch.Tensor([[0, 1, 0]]).long()
        }
        
        if dynamic_axis:
            dynamic_axis = {
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size'}
            }
        else:
            dynamic_axis = None
            
        return super().export_to_onnx(
            dummy_input,
            dynamic_axis,
            device=device,
        )
    
    def export_to_tensorrt(
        self,
        max_batch_size: int = 100,
        device: str = "cuda",
    ):
        
        min_shape = [
            [1, 1], [1, 1]
        ]
        
        max_shape = [
            [max_batch_size, self._tokenizer.model_max_length],
            [max_batch_size, self._tokenizer.model_max_length]
        ]

        import math

        opt_shape = [
            [math.ceil(max_batch_size/2), int(self._tokenizer.model_max_length / 2)],
            [math.ceil(max_batch_size/2), int(self._tokenizer.model_max_length / 2)]
        ]
        
        return super().export_to_tensorrt(
            min_shape=min_shape,
            opt_shape=opt_shape,
            max_shape=max_shape,
            names=["input_ids", "attention_mask"],
        )
    
    def export_to_traced_jit(self, device: str):

        dummy_input = {
            "input_ids": torch.Tensor([[1, 2, 3]]).long(),
            "attention_mask": torch.Tensor([[0, 1, 0]]).long()
        }
        
        return super().export_to_traced_jit(
            dummy_input,
            device=device,
        )
    
    def export_to_native(self, device: str):
        
        return super().export_to_native(
            device=device,
        )
            
    def forward_text(self, text: str):
        
        tokens = self._tokenizer(text, return_tensors="pt")
        
        return self.forward(**tokens)
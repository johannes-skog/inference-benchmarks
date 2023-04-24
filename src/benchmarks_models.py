import os
from typing import Dict, List
import torch
import onnx
import numpy as np
import onnxruntime
from tensor_rt import TensorRTBackend, Engine

class BenchmarkModelBase():

    def __call__(self, *args, **kwargs):
        
        return self.forward(*args, **kwargs)

class BenchmarkModelNative(BenchmarkModelBase):
    
    def __init__(self, model, device: str = "cpu") -> None:

        super().__init__()

        self._device = device
    
        self._model = model.to(device)
        
    def forward(self, inputs: Dict[str, torch.Tensor]):

        with torch.no_grad():

            inputs = {
                k: v.to(self._device) for k, v in inputs.items()
            }
                
            return self._model(**inputs)

class BenchmarkModelTracedJIT(BenchmarkModelBase):
    
    def __init__(self, model_path, device: str):

        super().__init__()

        self._device = device
        
        self._model = torch.jit.load(model_path).to(device)
    
    def forward(self, inputs: Dict[str, torch.Tensor]):
        
        with torch.no_grad():
            inputs = {
                k: v.to(self._device) for k, v in inputs.items()
            }
            
            return self._model(**inputs)

class BenchmarkModelTensorRT(BenchmarkModelBase):
    
    def __init__(self, run_engine):

        super().__init__()
        
        self._run_engine = run_engine
    
    def forward(self, inputs: Dict[str, torch.Tensor]):
        
        tokens = {k: v.numpy() for k,v in inputs.items()}
        
        return self._run_engine.run(list(tokens.values()))
    
class BenchmarkModelTensorONNX(BenchmarkModelBase):
    
    def __init__(self, onnx_path, device: str):

        super().__init__()
        
        self._onnx_path = onnx_path

        providers = []

        if device == 'cuda':
            providers.append('CUDAExecutionProvider')
        elif device == "tensorrt":
            providers.append('TensorrtExecutionProvider')
        else:
            providers.append('CPUExecutionProvider')

        self._session = onnxruntime.InferenceSession(
            onnx_path,
            providers=providers,
            verbose=True,
        )

        self._input_names = [x.name for x in self._session.get_inputs()]
        
        self._output_name = self._session.get_outputs()[0].name
        
    def forward(self, inputs):

        inputs = {
            k: v.cpu().numpy().astype(np.int64) for k, v in inputs.items() if k in self._input_names
        }
        
        return self._session.run([self._output_name], inputs)[0]

class BenchmarkModel(torch.nn.Module):
    
    def __init__(self, model, model_path: str):
        
        super().__init__()
        
        self._model = model
        self._model_path = model_path 
        
        os.makedirs(model_path, exist_ok=True)
        
    def forward(self, *args, **kwargs):
        
        return self._model.forward(*args, **kwargs)
    
    def export_to_onnx(
        self,
        dummy_input: Dict[str, torch.Tensor],
        dynamic_axis: Dict,
        device: str,
    ):
        
        self._model.eval()

        self._model.cpu()
        
        onnx_path = os.path.join(self._model_path, "onnx")

        torch.onnx.export(
            self._model,
            dummy_input,
            onnx_path,
            output_names=["output"],
            input_names=list(dummy_input.keys()),
            opset_version=12,
            dynamic_axes=dynamic_axis,
        )
        
        return BenchmarkModelTensorONNX(onnx_path, device=device)
    
    def export_to_traced_jit(
        self,
        inputs: Dict[str, torch.Tensor],
        device: str = "cpu",
    ):
        
        self._model.cpu()
        
        module = torch.jit.trace(
            self._model, example_kwarg_inputs=inputs
        )
        
        path = os.path.join(self._model_path, "traced")
        
        torch.jit.save(module, path)
        
        return BenchmarkModelTracedJIT(path, device)

    def export_to_tensorrt(
        self,
        min_shape: List[int],
        max_shape: List[int],
        opt_shape: List[int],
        names: List[str],
    ):
        
        self._model.eval()

        self._model.cpu()

        onnx_model = self.export_to_onnx(
            dynamic_axis=True,
            device="cpu",
        )

        backend = TensorRTBackend(
            model=onnx.load(onnx_model._onnx_path),
            verbose=False,
        )

        engine = backend.build_engine(
            min_shapes=min_shape,
            opt_shapes=opt_shape,
            max_shapes=max_shape,
            names=names, 
        )

        run_engine = Engine(engine)

        return BenchmarkModelTensorRT(run_engine)
    
    def export_to_native(self, device: str):

        self._model.cpu()

        return BenchmarkModelNative(self._model, device=device)
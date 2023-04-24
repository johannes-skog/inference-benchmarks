import tensorrt as trt
import pycuda.driver as cuda
import pycuda.driver
import pycuda.gpuarray
import pycuda.autoinit
import numpy as np
from six import string_types

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TensorRTBackend():
    
    def __init__(
        self,
        model,
        max_workspace_size=None,
        serialize_engine=False,
        verbose=False,
        **kwargs
    ):
        
        self._logger = TRT_LOGGER
        
        self.builder = trt.Builder(self._logger)
        
        self.config = self.builder.create_builder_config()
        
        self.network = self.builder.create_network(
            flags=1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH),
            
        )
        self.parser = trt.OnnxParser(self.network, self._logger)
        
        self.serialize_engine = serialize_engine
        
        self.verbose = verbose
        
        if self.verbose:
            print(f'\nRunning {model.graph.name}...')
            TRT_LOGGER.min_severity = trt.Logger.VERBOSE

        model_str = model.SerializeToString()
        
        if not self.parser.parse(model_str):
            error = self.parser.get_error(0)
            msg = "While parsing node number %i:\n" % error.node()
            msg += ("%s:%i In function %s:\n[%i] %s" %
                    (error.file(), error.line(), error.func(),
                     error.code(), error.desc()))
            raise RuntimeError(msg)
            
        self.config.max_workspace_size = (
            max_workspace_size if max_workspace_size is not None else (1 << 32)
        )

        self.num_inputs = self.network.num_inputs
        
        if self.verbose:
            for layer in self.network:
                print(layer)
            print(f'Output shape: {self.network[-1].get_output(0).shape}')
        
        self._output_shapes = {}
        self._output_dtype = {}
        
        for output in model.graph.output:
            dims = output.type.tensor_type.shape.dim
            output_shape = tuple([dim.dim_value for dim in dims])
            self._output_shapes[output.name] = output_shape
            self._output_dtype[output.name] = output.type.tensor_type.elem_type

    def build_engine(
        self,
        min_shapes,
        max_shapes,
        opt_shapes,
        names
    ):
        
        profile = self.builder.create_optimization_profile()

        print(min_shapes)
        
        for i, (min_shape, max_shape, opt_shape, name) in enumerate(zip(min_shapes, max_shapes, opt_shapes, names)):
            print(f'Binding {name} with min_shape: {min_shape}, max_shape: {max_shape}, opt_shape: {opt_shape}')
            profile.set_shape(
                name,
                min=min_shape,
                opt=opt_shape,
                max=max_shape,
            )
        
        self.config.add_optimization_profile(profile)

        trt_engine = self.builder.build_engine(self.network, self.config)

        if trt_engine is None:
            raise RuntimeError("Failed to build TensorRT engine from network")
        if self.serialize_engine:
            trt_engine = self._serialize_deserialize(trt_engine)
        
        self.engine = trt_engine
        
        return self.engine

class DynamicBinding(object):
    def __init__(self, engine, idx_or_name):
        if isinstance(idx_or_name, string_types):
            self.name = idx_or_name
            self.index = engine.get_binding_index(self.name)
            if self.index == -1:
                raise IndexError("Binding name not found: %s" % self.name)
        else:
            self.index = idx_or_name
            self.name = engine.get_binding_name(self.index)
            if self.name is None:
                raise IndexError("Binding index out of range: %i" % self.index)
        self.is_input = engine.binding_is_input(self.index)

        dtype = engine.get_binding_dtype(self.index)
        dtype_map = {trt.DataType.FLOAT: np.float32,
                     trt.DataType.HALF: np.float16,
                     trt.DataType.INT8: np.int8,
                     trt.DataType.BOOL: np.bool}
        if hasattr(trt.DataType, 'INT32'):
            dtype_map[trt.DataType.INT32] = np.int32

        self.dtype = dtype_map[dtype]
        self._host_buf = None
        self._device_buf = None

    def allocate_buffers(self, shape):
        
        self.shape = tuple(shape)
        # Must allocate a buffer of size 1 for empty inputs / outputs
        if 0 in self.shape:
            self.empty = True
            # Save original shape to reshape output binding when execution is done
            self.empty_shape = self.shape
            self.shape = tuple([1])
        else:
            self.empty = False

        self._host_buf = pycuda.driver.pagelocked_empty(self.shape, self.dtype)
        self._device_buf = pycuda.gpuarray.empty(self.shape, self.dtype)

    @property
    def host_buffer(self):
        return self._host_buf

    @property
    def device_buffer(self):
        return self._device_buf

    def get_async(self, stream):
        src = self.device_buffer
        dst = self.host_buffer
        src.get_async(stream, dst)
        return dst

def squeeze_hw(x):
    if x.shape[-2:] == (1, 1):
        x = x.reshape(x.shape[:-2])
    elif x.shape[-1] == 1:
        x = x.reshape(x.shape[:-1])
    return x

def check_input_validity(input_idx, input_array, input_binding):
    # Check shape
    trt_shape = tuple(input_binding.shape)
    onnx_shape    = tuple(input_array.shape)

    if onnx_shape != trt_shape:
        if not (trt_shape == (1,) and onnx_shape == ()) :
            raise ValueError("Wrong shape for input %i. Expected %s, got %s." %
                            (input_idx, trt_shape, onnx_shape))

    # Check dtype
    if input_array.dtype != input_binding.dtype:
        #TRT does not support INT64, need to convert to INT32
        if input_array.dtype == np.int64 and input_binding.dtype == np.int32:
            casted_input_array = np.array(input_array, copy=True, dtype=np.int32)
            if np.equal(input_array, casted_input_array).all():
                input_array = casted_input_array
            else:
                raise TypeError("Wrong dtype for input %i. Expected %s, got %s. Cannot safely cast." %
                            (input_idx, input_binding.dtype, input_array.dtype))
        else:
            raise TypeError("Wrong dtype for input %i. Expected %s, got %s." %
                            (input_idx, input_binding.dtype, input_array.dtype))
    return input_array


class Engine(object):
    def __init__(self, trt_engine):
        self.engine = trt_engine
        nbinding = self.engine.num_bindings

        self.bindings = [DynamicBinding(self.engine, i) for i in range(nbinding)]

        self.inputs = [b for b in self.bindings if b.is_input]
        self.outputs = [b for b in self.bindings if not b.is_input]
        
        self.context = self.engine.create_execution_context()
        self.context.active_optimization_profile = 0  # Add this line
        
        self.stream = pycuda.driver.Stream()

    def __del__(self):
        if self.engine is not None:
            del self.engine

    def allocate_buffers(self, input_shapes):
        for input_binding, shape in zip(self.inputs, input_shapes):
            input_binding.allocate_buffers(shape)
            self.context.set_binding_shape(input_binding.index, shape)

        for output_binding in self.outputs:
            output_index = output_binding.index
            output_shape = self.context.get_binding_shape(output_index)

            if any(dim < 0 for dim in output_shape):
                raise ValueError(f"Invalid output shape: {output_shape}. Please make sure the optimization profiles are configured correctly.")

            output_binding.allocate_buffers(output_shape)


    def run(self, inputs):
        if len(inputs) < len(self.inputs):
            raise ValueError("Not enough inputs. Expected %i, got %i." %
                             (len(self.inputs), len(inputs)))
        if isinstance(inputs, dict):
            inputs = [inputs[b.name] for b in self.inputs]

        # Update input shapes and allocate buffers
        input_shapes = [input_array.shape for input_array in inputs]
        self.allocate_buffers(input_shapes)

        binding_addrs = [b.device_buffer.ptr for b in self.bindings]

        for i, (input_array, input_binding) in enumerate(zip(inputs, self.inputs)):
            input_array = check_input_validity(i, input_array, input_binding)
            input_binding_array = input_binding.device_buffer
            input_binding_array.set_async(input_array, self.stream)

        self.context.execute_async_v2(binding_addrs, self.stream.handle)

        results = [output.get_async(self.stream) for output in self.outputs]

        # For any empty bindings, update the result shape to the expected empty shape
        for i, (output_array, output_binding) in enumerate(zip(results, self.outputs)):
            if output_binding.empty:
                results[i] = np.empty(shape=output_binding.empty_shape, dtype=output_binding.dtype)

        self.stream.synchronize()
        return results

    def run_no_dma(self, batch_size):
        self.context.execute_async(
            batch_size, self.binding_addrs, self.stream.handle
        )

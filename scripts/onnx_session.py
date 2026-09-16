import onnxruntime as ort
import numpy as np


class OnnxSession:
    def __init__(self, model_path, device, cuda_required, logger, cuda_library=None):
        self._logger = logger

        if device not in ('cpu', 'gpu'):
            raise ValueError(f'Invalid device: {device}. Use "cpu" or "gpu".')

        if device == 'gpu':
            providers = ['CUDAExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        session_options = ort.SessionOptions()

        if cuda_required:
            if not cuda_library:
                raise ValueError('cuda_required is true but cuda_library is empty.')

            session_options.register_custom_ops_library(cuda_library)
            self._logger.info(f'Loaded CUDA custom ops: {cuda_library}')

        self._session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=providers
        )

        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        self._logger.info(f'Loaded ONNX model: {model_path}')
        self._logger.info(f'ONNX providers: {self._session.get_providers()}')

        self._logger.info(
            f"ONNX input: {self._input_name}, "
            f"shape: {self._session.get_inputs()[0].shape}, "
            f"type: {self._session.get_inputs()[0].type}"
            f"ONNX output: {self._output_name}, "
            f"shape: {self._session.get_outputs()[0].shape}, "
            f"type: {self._session.get_outputs()[0].type}"
        )

    def run(self, input_array):
        outputs = self._session.run([self._output_name], {self._input_name: input_array})
        return np.asarray(outputs[0])
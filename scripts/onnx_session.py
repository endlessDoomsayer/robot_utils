
import onnxruntime as ort
from pathlib import Path
import numpy as np

class OnnxSession:
    def __init__(self, model_path: str, device: str, logger) -> None:
        self._logger = logger
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'ONNX model not found: {path}')

        providers = ['CPUExecutionProvider']
        if device in ('auto', 'cuda') and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self._session = ort.InferenceSession(str(path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._logger.info(f'Loaded ONNX model: {path}')

    def run(self, input_array: np.ndarray) -> np.ndarray:
        outputs = self._session.run([self._output_name], {self._input_name: input_array})
        return np.asarray(outputs[0])
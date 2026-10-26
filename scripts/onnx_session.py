"""ONNX Runtime implementation of the generic model-session interface."""
import os
import numpy as np
import onnxruntime as ort

from robot_utils.model_session import ModelSession


class OnnxSession(ModelSession):
    def __init__(
        self,
        model_path,
        device,
        cuda_library_required=False,
        logger=None,
        cuda_library_path=None,
    ):
        # Keep the legacy positional order (model_path, device, flag, logger,
        # library path) while exposing the new parameter names.
        if logger is None:
            raise ValueError("logger must be provided.")
        self._logger = logger
        requested_device = str(device).strip().lower()
        if requested_device == "gpu":
            requested_device = "cuda"
        if requested_device not in {"cpu", "cuda"}:
            raise ValueError(
                f"Unsupported ONNX device '{device}'. Use 'cpu' or 'cuda'."
            )

        # Do not silently run on CPU when CUDA was explicitly requested.
        providers = (
            ["CUDAExecutionProvider"]
            if requested_device == "cuda"
            else ["CPUExecutionProvider"]
        )
        options = ort.SessionOptions()

        if cuda_library_required:
            if not cuda_library_path:
                raise ValueError(
                    "cuda_library_required is true but cuda_library_path is empty."
                )
            if not os.path.isfile(cuda_library_path):
                raise FileNotFoundError(
                    f"ONNX custom-op library does not exist: {cuda_library_path}"
                )
            options.register_custom_ops_library(cuda_library_path)
            self._logger.info(
                f"Registered ONNX custom-op library: {cuda_library_path}"
            )

        self._session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=providers,
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if not inputs or not outputs:
            raise RuntimeError(f"ONNX model has no inputs or outputs: {model_path}")

        self._input_name = inputs[0].name
        self._output_names = [item.name for item in outputs]
        self._logger.info(f"Loaded ONNX model: {model_path}")
        self._logger.info(f"ONNX providers: {self._session.get_providers()}")
        self._logger.info(
            f"ONNX input: {inputs[0].name}, shape={inputs[0].shape}, type={inputs[0].type}"
        )
        self._logger.info(
            "ONNX outputs: " + ", ".join(
                f"{item.name} shape={item.shape} type={item.type}" for item in outputs
            )
        )

    def run(self, input_data):
        if isinstance(input_data, dict):
            raise TypeError("ONNX session currently expects a single NumPy input array.")
        output = self._session.run(
            self._output_names,
            {self._input_name: np.asarray(input_data)},
        )
        return np.asarray(output[0])

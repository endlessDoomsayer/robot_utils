"""Backend-neutral inference session interface and factory."""
from abc import ABC, abstractmethod


class ModelSession(ABC):
    """Common inference API: run(input_data) -> numpy.ndarray."""

    @abstractmethod
    def run(self, input_data):
        """Run inference and return the primary model output as a NumPy array."""
        raise NotImplementedError


def create_model_session(
    model_format,
    model_path,
    device,
    logger,
    cuda_library_required=False,
    cuda_library_path=None,
    model_builder=None,
):
    """Create a backend session from ROS-configured model parameters.

    model_builder is an optional callable used by non-serialized PyTorch
    architectures (e.g. OpenPoints), and must return a ready-to-load nn.Module.
    """
    normalized_format = str(model_format).strip().lower()
    if normalized_format == "onnx":
        from robot_utils.onnx_session import OnnxSession
        return OnnxSession(
            model_path=model_path,
            device=device,
            logger=logger,
            cuda_library_required=cuda_library_required,
            cuda_library_path=cuda_library_path,
        )
    if normalized_format in {"torch", "pytorch"}:
        from robot_utils.torch_session import TorchSession
        return TorchSession(
            model_path=model_path,
            device=device,
            logger=logger,
            model_builder=model_builder,
        )
    raise ValueError(
        f"Unsupported model format '{model_format}'. Expected 'onnx' or 'torch'."
    )

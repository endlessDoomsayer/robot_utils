"""PyTorch implementation of the generic model-session interface."""
from pathlib import Path
import numpy as np
import torch

from robot_utils.model_session import ModelSession


class TorchSession(ModelSession):
    """Runs either a TorchScript artifact or a model supplied by model_builder."""

    def __init__(self, model_path, device, logger, model_builder=None):
        self._logger = logger
        self.device = self._resolve_device(device)

        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PyTorch model/checkpoint does not exist: {path}")

        if model_builder is not None:
            model = model_builder()
        else:
            try:
                model = torch.jit.load(str(path), map_location=self.device)
            except Exception as jit_exc:
                # A full nn.Module serialized with torch.save is also supported.
                # State dictionaries still require an architecture-specific builder.
                try:
                    try:
                        loaded = torch.load(
                            str(path), map_location=self.device, weights_only=False
                        )
                    except TypeError:
                        loaded = torch.load(str(path), map_location=self.device)
                except Exception as load_exc:
                    raise RuntimeError(
                        f"Could not load '{path}' as TorchScript or a serialized "
                        "PyTorch module. For a state-dict checkpoint, provide a "
                        "model_builder that constructs the architecture and loads "
                        "its weights."
                    ) from load_exc
                if not isinstance(loaded, torch.nn.Module):
                    raise RuntimeError(
                        f"'{path}' is not a serialized torch.nn.Module. "
                        "A state-dict checkpoint requires a model_builder."
                    ) from jit_exc
                model = loaded

        if not isinstance(model, torch.nn.Module):
            raise TypeError("PyTorch model builder must return torch.nn.Module.")
        self.model = model.to(self.device)
        self.model.eval()
        self._logger.info(f"Loaded PyTorch model: {path} on {self.device}")

    @staticmethod
    def _resolve_device(device):
        requested = str(device or "cpu").strip().lower()
        if requested == "gpu":
            requested = "cuda"
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"PyTorch device '{requested}' requested but CUDA is unavailable."
            )
        if requested != "cpu" and not requested.startswith("cuda"):
            raise ValueError(f"Unsupported PyTorch device '{device}'. Use 'cpu', 'cuda', or 'auto'.")
        return torch.device(requested)

    @staticmethod
    def _primary_tensor(output):
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, dict):
            for key in ("logits", "seg_logits", "cls_logits", "pred"):
                value = output.get(key)
                if isinstance(value, torch.Tensor):
                    return value
            for value in output.values():
                if isinstance(value, torch.Tensor):
                    return value
        if isinstance(output, (tuple, list)):
            for value in output:
                if isinstance(value, torch.Tensor):
                    return value
        raise TypeError(
            "PyTorch model output must be a tensor or contain a tensor "
            "in a dictionary/tuple/list."
        )

    def run(self, input_data):
        if isinstance(input_data, dict):
            model_input = {
                key: value.to(self.device) if isinstance(value, torch.Tensor)
                else torch.as_tensor(value, device=self.device)
                for key, value in input_data.items()
            }
        elif isinstance(input_data, torch.Tensor):
            model_input = input_data.to(self.device)
        else:
            model_input = torch.as_tensor(np.asarray(input_data), device=self.device)

        with torch.inference_mode():
            output = self._primary_tensor(self.model(model_input))
        return output.detach().cpu().numpy()

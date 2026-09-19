"""Selected upstream EdgeTAM backend; HTTP/ROS service code is not included."""
from __future__ import annotations
import threading
import gc
import math
import numpy as np

class TrackingFailure(RuntimeError):
    pass

class EdgeTamBackend:
    """Lazily loaded official Transformers EdgeTAM streaming backend."""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self._load_lock = threading.Lock()
        self._inference_lock = threading.RLock()
        self._torch = None
        self._processor = None
        self._model = None
        self._device = None
        self._dtype = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            import torch
            from transformers import EdgeTamVideoModel, Sam2VideoProcessor

            device = torch.device(self.config.device)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "EDGETAM_DEVICE requests CUDA, but torch.cuda.is_available() is false",
                )
            if device.type == "cuda" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif device.type == "cuda":
                dtype = torch.float16
            else:
                dtype = torch.float32
            processor = Sam2VideoProcessor.from_pretrained(self.config.model_id)
            model = EdgeTamVideoModel.from_pretrained(self.config.model_id)
            model = model.to(device=device, dtype=dtype)
            model.eval()
            self._torch = torch
            self._device = device
            self._dtype = dtype
            self._processor = processor
            self._model = model

    def initialize(
        self,
        image: np.ndarray,
        bbox_xyxy: tuple[int, int, int, int],
    ) -> tuple[object, np.ndarray, float]:
        self._load()
        with self._inference_lock, self._torch.inference_mode():
            state = self._processor.init_video_session(
                inference_device=self._device,
                dtype=self._dtype,
                max_vision_features_cache_size=self.config.vision_cache_frames,
            )
            try:
                frame, original_size = self._prepare_frame(image)
                self._processor.add_inputs_to_inference_session(
                    inference_session=state,
                    frame_idx=0,
                    obj_ids=1,
                    input_boxes=[[[float(item) for item in bbox_xyxy]]],
                    original_size=original_size,
                )
                output = self._model(
                    inference_session=state,
                    frame_idx=0,
                    frame=frame,
                )
                mask, score = self._result(output, original_size)
                return state, mask, score
            except Exception:
                reset = getattr(state, "reset_inference_session", None)
                if callable(reset):
                    reset()
                raise

    def update(
        self,
        state: object,
        image: np.ndarray,
        frame_seq: int,
    ) -> tuple[np.ndarray, float]:
        self._load()
        with self._inference_lock, self._torch.inference_mode():
            frame, original_size = self._prepare_frame(image)
            output = self._model(
                inference_session=state,
                frame_idx=frame_seq,
                frame=frame,
            )
            result = self._result(output, original_size)
            self._prune_streaming_state(state, frame_seq)
            return result

    def reset(self, state: object) -> None:
        with self._inference_lock:
            reset = getattr(state, "reset_inference_session", None)
            if callable(reset):
                reset()
            processed_frames = getattr(state, "processed_frames", None)
            if isinstance(processed_frames, dict):
                processed_frames.clear()
            # Transformers keeps the CUDA caching allocator populated even
            # after its session dictionaries are cleared.  Releasing those
            # blocks here lets an isolated grasp model share the GPU.
            gc.collect()
            if (
                self._torch is not None
                and self._device is not None
                and self._device.type == "cuda"
            ):
                self._torch.cuda.empty_cache()

    def _stream_history_limit(self) -> int:
        """Keep every model dependency while bounding streaming state."""
        model_config = getattr(self._model, "config", None)
        required = max(
            int(getattr(model_config, "num_maskmem", 1)),
            int(getattr(model_config, "max_object_pointers_in_encoder", 1)),
        )
        return max(self.config.stream_history_frames, required)

    def _prune_streaming_state(self, state: object, frame_seq: int) -> None:
        """Discard history that EdgeTAM can no longer consult in streaming mode.

        The upstream inference session retains all frames and all per-frame
        outputs by default.  EdgeTAM's streaming forward pass only consults a
        bounded recent window plus conditioning frames, so retaining older
        non-conditioning entries causes linear GPU growth without changing a
        future result.
        """
        history_limit = self._stream_history_limit()
        cutoff = frame_seq - history_limit + 1
        if cutoff <= 0:
            return

        def prune_indexed(mapping: object, *, before: int = cutoff) -> None:
            if not isinstance(mapping, dict):
                return
            for index in tuple(mapping):
                if isinstance(index, int) and index < before:
                    mapping.pop(index, None)

        # A processed frame is only needed while its vision features are being
        # computed.  Retaining the same bounded history is conservative and
        # also supports the model's small feature cache.
        prune_indexed(getattr(state, "processed_frames", None))
        for outputs in getattr(state, "output_dict_per_obj", {}).values():
            if isinstance(outputs, dict):
                prune_indexed(outputs.get("non_cond_frame_outputs"))
        for tracked in getattr(state, "frames_tracked_per_obj", {}).values():
            prune_indexed(tracked)

    def _prepare_frame(self, image: np.ndarray) -> tuple[object, tuple[int, int]]:
        inputs = self._processor(
            images=image,
            device=self._device,
            return_tensors="pt",
        )
        original = inputs.original_sizes[0]
        if hasattr(original, "detach"):
            original = original.detach().cpu().tolist()
        original_size = (int(original[0]), int(original[1]))
        return inputs.pixel_values[0], original_size

    def _result(self, output: object, original_size: tuple[int, int]) -> tuple[np.ndarray, float]:
        logits = getattr(output, "pred_masks", None)
        if logits is None:
            raise TrackingFailure("EdgeTAM returned no mask logits")
        restored = self._processor.post_process_masks(
            [logits],
            original_sizes=[original_size],
            binarize=False,
        )[0]
        if hasattr(restored, "detach"):
            restored = restored.detach().float().cpu().numpy()
        array = np.asarray(restored, dtype=np.float32)
        while array.ndim > 2 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2:
            raise TrackingFailure(f"unexpected EdgeTAM mask shape {array.shape}")
        mask = array > 0.0
        pixels = int(mask.sum())
        if pixels < self.config.min_mask_pixels:
            raise TrackingFailure("EdgeTAM returned an empty or too-small mask")
        # EdgeTAM Video exposes mask logits but no object-score field.  Mean
        # foreground probability is therefore the service confidence metric.
        foreground_logits = np.clip(array[mask], -30.0, 30.0)
        score = float(np.mean(1.0 / (1.0 + np.exp(-foreground_logits))))
        if not math.isfinite(score) or score < self.config.min_score:
            raise TrackingFailure("EdgeTAM confidence is below the configured threshold")
        return mask, score

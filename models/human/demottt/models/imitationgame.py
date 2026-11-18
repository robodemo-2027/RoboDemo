"""Benchmark-specific I/O for the RoboDemo policy."""

from collections import OrderedDict

import torch
import torch.nn.functional as F

from demottt.models.policy import DemoTTTPolicy
from demottt.encoders.vision import PackedVisionTokenBatch, pack_post_merger_tokens
from demottt._vendor.starvla.model.framework.BackbonePolicy import _BackbonePolicyBase


def promote_trainable_parameters(module):
    """Keep optimizer/master parameters in FP32; autocast still controls compute."""
    counts = {}
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is not None:
            raise RuntimeError("Precision must be selected before training/optimizer construction")
        key = str(parameter.dtype)
        counts[key] = counts.get(key, 0) + parameter.numel()
        if parameter.is_floating_point() and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
    return counts


class ImitationGamePolicy(DemoTTTPolicy):
    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        from demottt.encoders.temporal import FrozenVJEPA2Encoder
        from transformers.modeling_utils import no_init_weights

        spec = config.framework.demottt.retrieval_temporal_encoder
        print("ImitationGame: loading frozen V-JEPA2 checkpoint", flush=True)
        with no_init_weights():
            self.human_temporal_encoder = FrozenVJEPA2Encoder(
                spec.repository, spec.checkpoint, reference_frames=16,
            )
        self._human_temporal_cache = OrderedDict()
        self._human_semantic_cache = OrderedDict()
        memory = config.framework.demottt
        self.reference_cache_size = int(memory.get("reference_cache_size", 4))
        self.cache_reference_semantics = bool(memory.get("cache_reference_semantics", False))
        self.reference_cache_device = str(memory.get("reference_cache_device", "cpu"))
        if self.reference_cache_device not in {"cpu", "cuda"} or self.reference_cache_size < 1:
            raise ValueError("Invalid frozen-reference cache configuration")
        if bool(memory.get("trainable_parameters_fp32", False)):
            previous = promote_trainable_parameters(self)
            print(f"ImitationGame: FP32 trainable parameters; previous dtype counts {previous}", flush=True)
        print("ImitationGame: model assets loaded", flush=True)

    def _messages(self, images, instruction, actual_length, training):
        if len(images) != 1:
            raise ValueError("IG-10K zed2i profile requires exactly one real camera, not duplicated views")
        return _BackbonePolicyBase._messages(self, images, instruction, actual_length, training)

    @torch.no_grad()
    def _encode_uncached_visual_sequences(self, sequences):
        if not sequences or not sequences[0] or any(len(seq) != len(sequences[0]) for seq in sequences):
            raise ValueError("Visual sequences must be nonempty and equally long")
        visual = self.backbone.vlm.model.visual
        if visual.training or any(p.requires_grad for p in visual.parameters()):
            raise RuntimeError("Writer-token encoding requires a frozen, eval-mode vision tower")
        images, owners = [], []
        for batch, sequence in enumerate(sequences):
            for time, step in enumerate(sequence):
                for view, source in enumerate(self.writer_views):
                    images.append(self._vision_messages(step["image"][source])[0]["content"][0]["image"])
                    owners.append((batch, time, view))
        # Writer features contain no text: bypass ragged chat-token construction.
        processed = self.collator.processor.image_processor(
            images=images, do_resize=False, return_tensors="pt",
        )
        device = next(self.backbone.parameters()).device
        grids = processed["image_grid_thw"]
        features, _ = self.backbone.vlm.get_image_features(
            processed["pixel_values"].to(device), grids.to(device),
        )
        return pack_post_merger_tokens(
            features, image_grid_thw=grids.cpu(), owners=torch.tensor(owners),
            batch_size=len(sequences), time_steps=len(sequences[0]),
            view_count=len(self.writer_views), spatial_merge_size=int(visual.spatial_merge_size),
        )

    def _encode_visual_sequences(self, sequences):
        keys = [sequence[0].get("reference_cache_key") for sequence in sequences]
        if not self.cache_reference_semantics or not all(keys):
            return self._encode_uncached_visual_sequences(sequences)
        if len(set(keys)) > self.reference_cache_size:
            raise ValueError("Reference cache is smaller than the active batch")
        unique = {}
        for key, sequence in zip(keys, sequences, strict=True):
            if any(step.get("reference_cache_key") != key for step in sequence):
                raise ValueError("Mixed reference identities within a visual sequence")
            unique[key] = sequence
        missing = [key for key in unique if key not in self._human_semantic_cache]
        if missing:
            encoded = self._encode_uncached_visual_sequences([unique[key] for key in missing])
            cache_device = encoded.tokens.device if self.reference_cache_device == "cuda" else "cpu"
            for row, key in enumerate(missing):
                self._human_semantic_cache[key] = PackedVisionTokenBatch(
                    *(value[row:row + 1].detach().to(device=cache_device, copy=True) for value in encoded)
                )
        selected = [self._human_semantic_cache[key] for key in keys]
        for key in keys:
            self._human_semantic_cache.move_to_end(key)
        while len(self._human_semantic_cache) > self.reference_cache_size:
            self._human_semantic_cache.popitem(last=False)
        maximum = max(value.tokens.shape[-2] for value in selected)
        device = next(self.backbone.parameters()).device
        return PackedVisionTokenBatch(
            torch.cat([F.pad(v.tokens, (0, 0, 0, maximum - v.tokens.shape[-2])) for v in selected]).to(device),
            torch.cat([F.pad(v.mask, (0, maximum - v.mask.shape[-1])) for v in selected]).to(device),
            torch.cat([F.pad(v.patch_positions, (0, 0, 0, maximum - v.tokens.shape[-2])) for v in selected]).to(device),
            torch.cat([v.view_ids for v in selected]).to(device),
        )

    def _retrieval_phase_writers(self, visual, sequences, examples, phase_membership):
        enriched = []
        for example in examples:
            if "retrieval_temporal_tokens" in example:
                enriched.append(example)
                continue
            key = example["reference_cache_key"]
            if key not in self._human_temporal_cache:
                clips = example["retrieval_temporal_frames"]
                if clips.ndim != 5 or clips.shape[:3] != (4, 16, 3):
                    raise ValueError("Human temporal clips must have shape [4,16,3,H,W]")
                device = next(self.human_temporal_encoder.parameters()).device
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    tokens = torch.cat([
                        self.human_temporal_encoder(clip[None].to(device))
                        for clip in clips
                    ])
                # Convert inference tensors before the trainable cross-attention consumes them.
                cache_device = device if self.reference_cache_device == "cuda" else "cpu"
                self._human_temporal_cache[key] = tokens.to(device=cache_device, dtype=torch.bfloat16).clone()
                while len(self._human_temporal_cache) > self.reference_cache_size:
                    self._human_temporal_cache.popitem(last=False)
            tokens = self._human_temporal_cache[key]
            self._human_temporal_cache.move_to_end(key)
            enriched.append({
                **example, "retrieval_temporal_tokens": tokens,
                "retrieval_temporal_mask": torch.ones(tokens.shape[:2], dtype=torch.bool),
            })
        return super()._retrieval_phase_writers(visual, sequences, enriched, phase_membership)

"""Pinned, frozen encoders. No fallback weights or silently truncated inputs."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
import re

from PIL import Image
import torch
from torch import nn

from .features import CandidateFeatures, InitialFeatures
from .preprocessing import (
    content_rows, instruction_span, last_valid_pool, letterbox, ordered_pool, patch_statistics,
)
from .schema import CandidateEvidence, InitialEvidence, Provenance, digest, file_hash
from .backbones import DINO_B, DINO_H, DINO_SPECS

SYSTEM = "Represent the user's input."


def freeze_backbone(model):
    return model.eval().requires_grad_(False)


def dino_patches(hidden, registers, feature_dim=768):
    if feature_dim not in (768, 1280) or hidden.ndim != 3 or hidden.shape[1:] != (1 + registers + 256, feature_dim):
        raise ValueError("DINO CLS/register/patch layout mismatch")
    return torch.nn.functional.normalize(hidden[:, 1 + registers:].float(), dim=-1)


def _check_loading(info):
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError("checkpoint keys do not match the exact backbone; refusing partial/random weights")


def _verified_image(frame):
    if file_hash(frame.image_path) != frame.image_sha256:
        raise ValueError("evidence image changed after manifest creation")
    image = Image.open(frame.image_path).convert("RGB")
    if image.size != (frame.width, frame.height):
        raise ValueError("evidence image dimensions changed")
    if file_hash(frame.original_image_path) != frame.original_image_sha256:
        raise ValueError("original decoded frame changed")
    original = Image.open(frame.original_image_path).convert("RGB")
    canonical, _, transform = letterbox(original, 448)
    if original.size != (frame.source_width, frame.source_height) or transform.to_dict() != frame.letterbox:
        raise ValueError("inverse letterbox provenance mismatch")
    if canonical.tobytes() != image.tobytes():
        raise ValueError("teacher pixels do not match student preprocessing")
    return original


def verify_snapshot(path, model_id, revision=None, *, manifest_path=None):
    import json
    path = Path(path)
    marker = Path(manifest_path) if manifest_path is not None else path / "refreward_source.json"
    if not marker.exists():
        raise ValueError("model snapshot lacks verified refreward_source.json")
    source = json.loads(marker.read_text())
    if source.get("source_type") == "local_snapshot":
        if (manifest_path is None or source.get("revision") is not None
                or source.get("snapshot_path") != str(path.resolve())
                or source.get("content_sha256") != digest({"model_id": model_id, "files": source["files"]})):
            raise ValueError("local snapshot requires a correct content-addressed sidecar")
    elif not re.fullmatch(r"[0-9a-f]{40}", source.get("revision") or ""):
        raise ValueError("snapshot revision must be an immutable commit")
    if source["model_id"] != model_id or (revision is not None and source["revision"] != revision):
        raise ValueError("model revision differs from cache provenance")
    if not any(name.endswith(".safetensors") for name in source["files"]):
        raise ValueError("verified snapshot must contain weights")
    for name, expected in source["files"].items():
        if not (path / name).resolve().is_relative_to(path.resolve()):
            raise ValueError("snapshot manifest path escape")
        if file_hash(path / name) != expected:
            raise ValueError("model snapshot checksum mismatch")
    return source


def qwen_embedding_class():
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLPreTrainedModel

    class ExactEmbeddingModel(Qwen3VLPreTrainedModel):
        _checkpoint_conversion_mapping = {}
        def __init__(self, config):
            super().__init__(config)
            self.model = Qwen3VLModel(config)
            self.post_init()
        def forward(self, **kwargs):
            return self.model(**kwargs)
    return ExactEmbeddingModel


class FrozenQwen:
    """The unchanged Qwen branch can be tested without substituting a missing DINO."""

    def __init__(self, qwen_path, *, revision=None, template_sha256=None,
                 device="cpu", dtype=torch.float32):
        from transformers import AutoProcessor
        self.snapshot = verify_snapshot(qwen_path, "Qwen/Qwen3-VL-Embedding-2B", revision)
        self.device, self.dtype = device, dtype
        kwargs = dict(local_files_only=True, torch_dtype=dtype, attn_implementation="sdpa",
                      output_loading_info=True)
        qwen, info = qwen_embedding_class().from_pretrained(qwen_path, **kwargs)
        _check_loading(info)
        self.qwen = freeze_backbone(qwen.to(device))
        self.processor = AutoProcessor.from_pretrained(qwen_path, local_files_only=True, padding_side="right")
        if not self.processor.tokenizer.is_fast:
            raise ValueError("offset-verified instruction extraction requires a fast tokenizer")
        config = self.qwen.config
        if config.text_config.hidden_size != 2048 or config.vision_config.patch_size != 16 or config.vision_config.spatial_merge_size != 2:
            raise ValueError("wrong Qwen embedding architecture")
        if template_sha256 is not None and digest(self.processor.chat_template) != template_sha256:
            raise ValueError("chat template differs from feature provenance")

    def _messages(self, instruction, image=None):
        content = [] if image is None else [{"type": "image", "image": image}]
        content.append({"type": "text", "text": instruction})
        return [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": content}]

    @torch.inference_mode()
    def text(self, instruction):
        rendered = self.processor.apply_chat_template(self._messages(instruction), add_generation_prompt=True, tokenize=False)
        span = instruction_span(rendered, instruction)
        tokenizer = self.processor.tokenizer
        tokens = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True,
                           return_special_tokens_mask=True, truncation=False, return_tensors="pt")
        special = tokens.pop("special_tokens_mask")[0].bool()
        special |= torch.isin(tokens["input_ids"][0], torch.tensor(tokenizer.all_special_ids))
        offsets = tokens.pop("offset_mapping")[0].tolist()
        indices = content_rows(offsets, special.tolist(), span)
        if len(indices) > 512:
            raise ValueError("instruction exceeds 512 content tokens")
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        hidden = self.qwen(**tokens, use_cache=False).last_hidden_state
        g = last_valid_pool(hidden, tokens["attention_mask"])
        t, mask, pos = ordered_pool(hidden[0, indices].float())
        return g, t[None], mask[None], pos[None]

    @torch.inference_mode()
    def joint(self, instruction, image):
        image, _, _ = letterbox(image, 448)
        rendered = self.processor.apply_chat_template(self._messages(instruction, image),
                                                       add_generation_prompt=True, tokenize=False)
        tokens = self.processor(text=[rendered], images=[image], do_resize=False,
                                truncation=False, padding=True, return_tensors="pt")
        if tuple(tokens["image_grid_thw"][0].tolist()) != (1, 28, 28):
            raise ValueError("expected exactly 196 merged Qwen image positions")
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        hidden = self.qwen(**tokens, use_cache=False).last_hidden_state
        return last_valid_pool(hidden, tokens["attention_mask"])


class FrozenDino:
    def __init__(self, dino_path, *, model_id=DINO_B, revision=None, manifest_path=None,
                 device="cpu", dtype=torch.float32):
        from transformers import AutoImageProcessor, AutoModel
        if model_id not in DINO_SPECS:
            raise ValueError("unauthorized DINO model")
        self.dino_snapshot = verify_snapshot(dino_path, model_id, revision, manifest_path=manifest_path)
        self.device, self.dtype = device, dtype
        self.dino_feature_dim = DINO_SPECS[model_id]["width"]
        dino, info = AutoModel.from_pretrained(
            dino_path, local_files_only=True, torch_dtype=dtype,
            attn_implementation="sdpa", output_loading_info=True)
        _check_loading(info)
        self.dino = freeze_backbone(dino.to(device))
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_path, local_files_only=True)
        if self.dino.config.hidden_size != self.dino_feature_dim or self.dino.config.patch_size != 16:
            raise ValueError("wrong DINO architecture for the explicit B/H+ variant")
        if model_id == DINO_H:
            expected = {"num_hidden_layers": 32, "num_attention_heads": 20,
                        "num_register_tokens": 4, "intermediate_size": 5120, "use_gated_mlp": True}
            if any(getattr(self.dino.config, k) != v for k, v in expected.items()):
                raise ValueError("H+ checkpoint configuration differs from the authorized snapshot")

    @torch.inference_mode()
    def visual(self, image):
        image, mask, transform = letterbox(image, 256)
        color, position, valid = patch_statistics(image, mask, transform)
        inputs = self.dino_processor(images=image, do_resize=False, do_center_crop=False, return_tensors="pt")
        pixels = inputs["pixel_values"].to(self.device, self.dtype)
        if tuple(pixels.shape[-2:]) != (256, 256):
            raise ValueError("DINO processor unexpectedly resized/cropped the image")
        hidden = self.dino(pixel_values=pixels).last_hidden_state
        registers = self.dino.config.num_register_tokens
        patches = dino_patches(hidden, registers, self.dino_feature_dim)
        return patches, color[None].to(self.device), position[None].to(self.device), valid[None].to(self.device)


class FrozenExtractors(FrozenQwen, FrozenDino):
    def __init__(self, qwen_path, dino_path, provenance: Provenance, *, dino_manifest=None,
                 device="cpu", dtype=torch.float32):
        FrozenDino.__init__(self, dino_path, model_id=provenance.dino_model,
                           revision=provenance.dino_revision, manifest_path=dino_manifest,
                           device=device, dtype=dtype)
        FrozenQwen.__init__(self, qwen_path, revision=provenance.qwen_revision,
                           template_sha256=provenance.template_sha256, device=device, dtype=dtype)
        self.provenance = provenance
        if self.dino_snapshot.get("content_sha256") != provenance.dino_content_sha256:
            raise ValueError("DINO snapshot content differs from feature provenance")

    @torch.inference_mode()
    def initial(self, evidence: InitialEvidence):
        if type(evidence) is not InitialEvidence:
            raise TypeError("query extractor accepts only the initial evidence schema")
        if evidence.camera_group != self.provenance.camera_group:
            raise ValueError("camera provenance mismatch")
        image = _verified_image(evidence.initial)
        g, t, text_mask, pos = self.text(evidence.instruction)
        z = self.joint(evidence.instruction, image)
        f, c, p, spatial_mask = self.visual(image)
        return InitialFeatures(g, z, t, f, c, p, text_mask, spatial_mask,
                               torch.ones((1, 2), device=self.device, dtype=torch.bool), pos).validate()

    @torch.inference_mode()
    def candidate(self, evidence: CandidateEvidence):
        initial = InitialEvidence(**{k: getattr(evidence, k) for k in InitialEvidence.model_fields})
        features = self.initial(initial)
        u = torch.zeros((1, 32, 2048), device=self.device)
        mask = torch.zeros((1, 32), device=self.device, dtype=torch.bool)
        times = torch.zeros((1, 32), device=self.device)
        for slot, frame in enumerate(evidence.process):
            u[0, slot] = features.z[0] if slot == 0 else self.joint(evidence.instruction, _verified_image(frame))[0]
            mask[0, slot] = True
            times[0, slot] = frame.normalized_time
        return CandidateFeatures(**features.tensors(), U=u, process_mask=mask, process_time=times).validate()

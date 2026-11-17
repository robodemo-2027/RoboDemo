"""Decode the reference videos selected for the released evaluation episodes."""
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
import random
import numpy as np
import torch
from io_utils import read_json


@contextmanager
def preserve_policy_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[0]):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class RecordedReferences:
    def __init__(self, plan, data_root, *, robot=False, bank=None, selections=None):
        self.manifest = {"normalization": plan["normalization"], "humans": {}}
        self.reference_cache_size = 1
        self._references = OrderedDict()
        self._robot_references = {}
        self.data_root = Path(data_root)
        self.robot = robot
        self.selections = selections
        self.bank = {row["uid"]: row for rows in (bank or {}).values() for row in rows}
        for trial in plan["trials"]:
            if not robot:
                self.manifest["humans"].setdefault(trial["human_task"], [self._local(trial["reference"])])

    def _local(self, reference):
        ref = dict(reference)
        ref["video"] = str(self.data_root / ref["video"])
        return ref

    def prepare(self, model, trial, initial_rgb, directory):
        from demottt.data.imitationgame import ImitationGameDataset
        if self.robot:
            selected = self.selections[trial["trial_id"]]
            ref = self._local(self.bank[selected["reference_uid"]])
            self._robot_references = {trial["trial_id"]: ref}
            episode = {"uid": trial["trial_id"]}
            task_id = trial["task_id"]
        else:
            episode = {"task": trial["human_task"], "reference_index": 0}
            task_id = trial["human_task"]
        # Reference encoding must not advance the per-episode policy noise stream.
        with preserve_policy_rng():
            reference, demonstration, clips = ImitationGameDataset._reference(self, episode)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = torch.cat([model.human_temporal_encoder(clip[None].cuda())
                                    for clip in clips]).cpu()
            packet = {
                "demonstration": demonstration,
                "demonstration_metadata": {"task_id": task_id, "semantic_id": task_id,
                                           "constraint_id": "unconstrained"},
                "retrieval_temporal_tokens": tokens,
                "retrieval_temporal_mask": torch.ones(tokens.shape[:2], dtype=torch.bool),
            }
        return reference["uid"], packet

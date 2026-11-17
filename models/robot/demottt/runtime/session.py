"""Serialized, rollback-safe single-stream deployment around the real policy API."""

import copy
import threading

import numpy as np

from demottt.runtime.registry import CompatibilityKey, stable_payload_fingerprint


class _MemoryTransaction:
    def __init__(self, policy):
        self.policy = policy
        self.policy_state = copy.deepcopy({
            key: value for key, value in vars(policy).items() if key.startswith("_runtime_")
        })
        self.layers = []
        for layer in policy.motion_ttt.layers:
            self.layers.append((layer, copy.deepcopy({
                name: getattr(layer, name) for name in (
                    "_retrieval_phase_state", "_retrieval_phase_valid", "_retrieval_state", "_retrieval_valid",
                    "_execution_state", "last_execution_age"
                )
                if hasattr(layer, name)
            })))

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        if exception_type is not None:
            for key in list(vars(self.policy)):
                if key.startswith("_runtime_") and key not in self.policy_state:
                    delattr(self.policy, key)
            for key, value in self.policy_state.items():
                setattr(self.policy, key, value)
            for layer, values in self.layers:
                for key, value in values.items():
                    setattr(layer, key, value)
            self.policy.motion_ttt.clear_context()
        return False


class PolicySession:
    """One policy instance owns one stream; use separate instances for concurrent streams.

    Repeated requests are cached; conflicting retries and skipped control steps
    fail before writes. A failed call restores memory, so retry cannot double-write.
    This does not authorize robot actuation and does not validate the semantics
    of an externally supplied executed-action receipt.
    """

    def __init__(self, policy, stream_id):
        if not str(stream_id).strip():
            raise ValueError("stream_id must be nonempty")
        if policy.training:
            raise ValueError("Deployment requires policy.eval()")
        if getattr(policy, "_demottt_session_owner", None) is not None:
            raise ValueError("A policy instance already belongs to a session")
        self.policy = policy
        self.stream_id = str(stream_id)
        policy._demottt_session_owner = self.stream_id
        self._lock = threading.RLock()
        self._identity = None
        self._step = -1
        self._fingerprint = None
        self._response = None
        self._reference_fingerprint = None
        self._closed = False

    def predict(self, example, *, episode_id, demonstration_id, step_id, seed=42, **sampler):
        if type(step_id) is not int or step_id < 0:
            raise ValueError("step_id must be a nonnegative integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not str(episode_id).strip() or not str(demonstration_id).strip():
            raise ValueError("Explicit episode and demonstration identities are required")
        if int(sampler.get("num_candidates", 1)) != 1:
            raise ValueError("The audited session supports one candidate")
        if step_id == 0 and example.get("previous_executed_action") is not None:
            raise ValueError("Step zero cannot contain a preceding executed transition")
        if step_id > 0 and example.get("previous_executed_action") is None:
            raise ValueError("Every continuation requires previous_executed_action")
        if self.policy.training:
            raise ValueError("The session policy was switched back to training mode")
        key = self.policy._runtime_compatibility_key(example, example)
        identity = (str(episode_id), str(demonstration_id), CompatibilityKey.normalize(key))
        fingerprint = stable_payload_fingerprint({"example": example, "seed": seed, "sampler": sampler})
        reference = example.get("demonstration")
        reference_fingerprint = None if reference is None else stable_payload_fingerprint({
            "demonstration": reference,
            "temporal_tokens": example.get("retrieval_temporal_tokens"),
            "temporal_mask": example.get("retrieval_temporal_mask"),
            "metadata": example.get("demonstration_metadata"),
        })
        with self._lock:
            if self._closed:
                raise RuntimeError("Session is closed")
            changed = identity != self._identity
            if changed and step_id != 0:
                raise ValueError("Changed episode/reference/compatibility requires step_id=0")
            if not changed and step_id == self._step:
                if fingerprint != self._fingerprint:
                    raise ValueError("Conflicting retry: observation, feedback, seed or sampler changed")
                return copy.deepcopy(self._response)
            if not changed and step_id != self._step + 1:
                raise ValueError("Out-of-order or skipped control step")
            if not changed and reference_fingerprint is not None and reference_fingerprint != self._reference_fingerprint:
                raise ValueError("Reference contents changed without a new demonstration identity")
            request = dict(example)
            request["demonstration_episode_id"] = demonstration_id
            sampling = dict(sampler)
            if sampling.get("candidate_noise") is None:
                sampling["candidate_noise"] = np.random.default_rng(seed).standard_normal((1, 30, 60)).astype(np.float32)
            with _MemoryTransaction(self.policy):
                if changed:
                    self.policy.reset_motion_memory()
                response = self.policy.predict_action(request, **sampling)
            self._identity = identity
            self._step = step_id
            self._fingerprint = fingerprint
            self._response = copy.deepcopy(response)
            if changed:
                self._reference_fingerprint = reference_fingerprint
            return response

    def close(self):
        with self._lock:
            if self._closed:
                return
            self.policy.reset_motion_memory()
            self._identity = None
            self._step = -1
            self._fingerprint = None
            self._response = None
            self._reference_fingerprint = None
            self.policy._demottt_session_owner = None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()
        return False

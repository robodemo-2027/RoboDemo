"""Reference decoding and normalization from the evaluated policy implementation."""
import numpy as np
import torch

ACTION_KEY = "action.qpos_gripper_actions"
STATE_KEY = "observation.qpos_gripper_states"

def sampling_indices(length, count):
    if length < 4:
        raise ValueError("A human demonstration must contain at least four frames")
    return np.stack([
        np.rint(np.linspace(part[0], part[-1], count)).astype(np.int64)
        for part in np.array_split(np.arange(length), 4)
    ])


def normalize_slots(values, bounds, width):
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] != width or not np.isfinite(values).all():
        raise ValueError(f"Expected finite source vectors of width {width}")
    low, high = np.asarray(bounds["min"]), np.asarray(bounds["max"])
    scale = high - low
    normalized = np.where(scale > 1e-8, 2 * (values - low) / np.maximum(scale, 1e-8) - 1, 0)
    if not np.isfinite(normalized).all():
        raise ValueError("Non-finite normalized data")
    output = np.zeros((*values.shape[:-1], 60), dtype=np.float32)
    output[..., :width] = normalized
    return output


def denormalize_actions(values, bounds):
    values = np.asarray(values)[..., :16]
    low, high = np.asarray(bounds["min"]), np.asarray(bounds["max"])
    return (values + 1) * (high - low) / 2 + low


def read_frames(path, frame_indices, fps, *, max_width=384):
    """Seek by presentation timestamps; never infer episodes from video filenames."""
    import av

    requested = [int(i) for i in frame_indices]
    wanted = set(requested)
    if not wanted or min(wanted) < 0:
        raise ValueError("Invalid frame request")
    found = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        target = int(min(wanted) / fps / float(stream.time_base))
        container.seek(target, stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Dataset video frame has no presentation timestamp")
            index = round(float(frame.pts * stream.time_base) * fps)
            if index > max(wanted):
                break
            if index in wanted:
                width = min(max_width, frame.width)
                height = max(2, round(frame.height * width / frame.width / 2) * 2)
                found[index] = frame.reformat(width=width, height=height, format="rgb24").to_ndarray()
                if len(found) == len(wanted):
                    break
    if wanted - found.keys():
        raise ValueError(f"Missing exact source frames in {path}: {sorted(wanted - found.keys())[:8]}")
    return [found[index] for index in requested]


class ImitationGameDataset:
    def _reference(self, episode):
        ref = self._robot_references[episode["uid"]]
        if ref["uid"] not in self._references:
            semantic = sampling_indices(ref["length"], 12)
            temporal = sampling_indices(ref["length"], 16)
            needed = sorted(set(semantic.flatten()) | set(temporal.flatten()))
            frames = read_frames(ref["video"], np.asarray(needed) + ref["video_start_frame"], ref["fps"])
            by_frame = dict(zip(needed, frames))
            demo = [{
                "image": [by_frame[int(index)]], "phase_id": phase,
                "relative_timestamp_s": int(index) / ref["fps"],
                "frame_index": int(index), "context_fps": ref["fps"],
                "motion_ttt_writer_valid": True,
                "reference_cache_key": ref["uid"],
            } for phase in range(4) for index in semantic[phase]]
            clips = np.stack([[by_frame[int(index)] for index in phase] for phase in temporal])
            self._references[ref["uid"]] = (demo, torch.from_numpy(clips).permute(0, 1, 4, 2, 3))
            while len(self._references) > self.reference_cache_size:
                self._references.popitem(last=False)
        demo, clips = self._references[ref["uid"]]
        self._references.move_to_end(ref["uid"])
        return ref, demo, clips


"""Plot one trial from the published MC_Maze training NWB file."""

import argparse
from bisect import bisect_left
from contextlib import contextmanager
from pathlib import Path

import fsspec
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Public contentUrl listed in the DANDI metadata for the release's training asset.
TRAIN_URL = (
    "https://dandiarchive.s3.amazonaws.com/blobs/df3/e3f/"
    "df3e3f73-50ab-42b4-8827-82664ddd474a"
)


@contextmanager
def open_nwb(path):
    if path is not None:
        with h5py.File(path, "r") as nwb:
            yield nwb
        return

    # Keeping recently accessed blocks makes HDF5 scalar searches inexpensive.
    with fsspec.open(TRAIN_URL, "rb", block_size=65536, cache_type="blockcache") as remote:
        with h5py.File(remote, "r") as nwb:
            yield nwb
        print(f"HTTP range cache fetched {remote.cache.total_requested_bytes:,} bytes")


def load_trial(nwb, trial_id):
    trials = nwb["intervals/trials"]
    matches = np.flatnonzero(trials["id"][:] == trial_id)
    if len(matches) != 1:
        raise ValueError(f"Expected one NWB trial with ID {trial_id}")
    row = int(matches[0])
    start = float(trials["start_time"][row])
    stop = float(trials["stop_time"][row])
    onset = float(trials["move_onset_time"][row])
    if not start <= onset < stop:
        raise ValueError("Movement onset is outside the trial window")

    hand = nwb["processing/behavior/hand_vel"]
    timestamps = hand["timestamps"]
    first = bisect_left(timestamps, start)
    last = bisect_left(timestamps, stop, lo=first)
    times = timestamps[first:last]
    steps = np.diff(times)
    if (
        len(times) < 2
        or not np.all((start <= times) & (times < stop))
        or not np.allclose(steps, 0.001, rtol=0, atol=1e-8)
        or times[0] - start > 0.00100001
        or stop - times[-1] > 0.00100001
    ):
        raise ValueError("Expected complete, increasing 1 ms hand-velocity timestamps")

    data = hand["data"]
    velocity = data[first:last] * data.attrs["conversion"] + data.attrs.get("offset", 0.0)
    if velocity.shape != (len(times), 2) or not np.isfinite(velocity).all():
        raise ValueError("Expected two finite velocity components for every sample")

    units = nwb["units"]
    spike_times = units["spike_times"]
    ends = units["spike_times_index"][:]
    begins = np.r_[0, ends[:-1]]
    spikes = []
    for begin, end in zip(begins, ends):
        first_spike = bisect_left(spike_times, start, int(begin), int(end))
        last_spike = bisect_left(spike_times, stop, first_spike, int(end))
        events = spike_times[first_spike:last_spike]
        if not np.all((start <= events) & (events < stop)) or np.any(np.diff(events) < 0):
            raise ValueError("Found unordered or out-of-window spike timestamps")
        spikes.append(events - onset)

    split = trials["split"].asstr()[row]
    print(f"Trial {trial_id} ({split}), NWB row {row}")
    print(f"Trial [{start:.6f}, {stop:.6f}) s; movement onset {onset:.6f} s")
    print(
        f"Velocity: {len(times):,} samples from {times[0]:.6f} to {times[-1]:.6f} s; "
        f"median step {np.median(steps) * 1000:.3f} ms"
    )
    print(f"Raster: {sum(map(len, spikes)):,} spikes across {len(spikes)} unit rows")
    return split, (start - onset, stop - onset), times - onset, velocity, data.attrs["unit"], spikes


def plot_trial(trial_id, trial, output):
    split, window, times, velocity, unit, spikes = trial
    fig, (raster, vx, vy) = plt.subplots(
        3, 1, sharex=True, figsize=(10, 7), height_ratios=(3, 1, 1), layout="constrained"
    )
    fig.suptitle(f"MC_Maze · Trial {trial_id} ({split})", fontweight="bold", x=0.08, ha="left")
    raster.eventplot(spikes, colors="#23364b", linelengths=0.8, linewidths=0.45, rasterized=True)
    raster.set(ylabel="Unit row", ylim=(-1, len(spikes)))
    for axis, component, color in ((vx, "x", "#006e78"), (vy, "y", "#a54925")):
        axis.axhline(0, color="#c4cbd1", linewidth=0.7)
        axis.plot(times, velocity[:, "xy".index(component)], color=color, linewidth=1.4)
        axis.set_ylabel(f"$v_{component}$ ({unit})")
    for axis in (raster, vx, vy):
        axis.axvline(0, color="#535f6a", linestyle="--", linewidth=0.9)
        axis.spines[["top", "right"]].set_visible(False)
    vy.set(xlabel="Time from movement onset (s)", xlim=window)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nwb", type=Path, help="Local training NWB. Defaults to DANDI range reads.")
    parser.add_argument("--trial-id", default=0, type=int, help="NWB trial ID (default: 0)")
    parser.add_argument("--output", default=Path("trial.png"), type=Path)
    args = parser.parse_args()
    with open_nwb(args.nwb) as nwb:
        trial = load_trial(nwb, args.trial_id)
    plot_trial(args.trial_id, trial, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

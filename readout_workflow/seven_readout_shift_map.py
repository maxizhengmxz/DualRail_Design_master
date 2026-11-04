"""Explicit seven-mode cavity maps with logical spectator 0L AND 1L.

Fixed calibrated seven-mode hardware and OFF EJ; no pixel-by-pixel retuning.
Sparse low-energy eigensolves replace dense diagonalization, not the Hamiltonian.
Screening grid defaults to 31x31; both resonators retain |0>,|1>,|2>.
"""

import os

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
COUPLER = HERE.parent / "Gnd-coupler+Floating-T"
sys.path[:0] = [str(HERE.parent), str(COUPLER)]

START = dict(local_levels=6, pair_states=12, coupler_levels=3, ncut=35)
AUDIT = dict(local_levels=6, pair_states=12, coupler_levels=5, ncut=35)


def source_identity():
    paths = [
        Path(__file__),
        COUPLER / "seven_readout.py",
        COUPLER / "floating_coupler/pairs.py",
        COUPLER / "floating_coupler/model.py",
        COUPLER / "integrated_readout_range.py",
        HERE.parent / "core/transmon.py",
        HERE.parent / "core/cz_channel_policy.py",
    ]
    return {
        str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def map_signature(cell, pair, points):
    return hashlib.sha256(
        json.dumps(
            dict(
                cell=cell,
                pair=pair,
                points=points,
                basis=START,
                solver_policy="sparse-window-v2-overlap-guard",
                sources=source_identity(),
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def readout_keys(pair):
    phot = (1, 0) if pair == "A" else (0, 1)
    groups = []
    for spectator in (0, 1):
        ground = f"g{spectator}" if pair == "A" else f"{spectator}g"
        modes = [
            f"{i}{spectator}" if pair == "A" else f"{spectator}{i}" for i in (0, 1)
        ]
        groups.append(
            [(label, *ph) for label in (ground, *modes) for ph in ((0, 0), phot)]
        )
    return groups


def extract(system, pair):
    import numpy as np

    fr = system.physical["resonator_parameters"]["r1" if pair == "A" else "r2"][
        "fr_ghz"
    ]
    shifts = []
    weights = []
    matter = []
    for keys in readout_keys(pair):
        shifts.append(
            [
                1000 * (system.eigen[up] - system.eigen[lo] - fr)
                for lo, up in zip(keys[::2], keys[1::2])
            ]
        )
        weights.append(min(system.weights[k] for k in keys))
        matter.append(
            min(
                abs(
                    system.matter_vectors[
                        system.pair.ref_indices[system.pair.labels.index(k[0])],
                        system.matter_assignment[k[0]],
                    ]
                )
                ** 2
                for k in keys
            )
        )
    return np.array(shifts), np.array(weights), np.array(matter)


def solve_point(model, physical, fc, pair, basis=START, dense=False):
    import numpy as np
    from scipy.sparse.linalg import eigsh
    from scipy.optimize import linear_sum_assignment
    from seven_readout import SevenReadout

    system = SevenReadout(model, physical, fc, basis=basis, photons=3, eigenlevels=100)
    if dense:
        return extract(system.solve(), pair)
    keys = [key for group in readout_keys(pair) for key in group]
    for parity in (1, -1):
        sector = np.flatnonzero(system.parity == parity)
        wanted = [key for key in keys if system.parity[system.refs[key]] == parity]
        block = system.ham[sector, :][:, sector]
        refs = []
        for key in wanted:
            photon = np.zeros(9)
            photon[np.ravel_multi_index(key[1:], (3, 3))] = 1.0
            ref = np.kron(
                system.matter_vectors[:, system.matter_assignment[key[0]]], photon
            )
            refs.append(ref[sector])
        refs = np.array(refs)
        for count in (45, 70, 100):
            energy, v = eigsh(
                block,
                k=count,
                which="SA",
                tol=1e-10,
                v0=np.linspace(0.1, 1, len(sector)),
            )
            order = np.argsort(energy)
            energy, v = energy[order], v[:, order]
            overlaps = abs(refs @ v) ** 2
            _, columns = linear_sum_assignment(-overlaps)
            # A missing target can falsely match a much lower state with tiny
            # overlap, without selecting the top eigenvalue. Inspect overlap
            # as well as the index guard before declaring the window adequate.
            if max(columns) < count - 3 and (
                float(overlaps[np.arange(len(wanted)), columns].min()) >= 0.95
                or count == 100
            ):
                break
        else:
            raise ValueError("Readout energy window exhausted")
        for i, (key, col) in enumerate(zip(wanted, columns)):
            system.eigen[key] = float(energy[col])
            system.weights[key] = float(overlaps[i, col])
    return extract(system, pair)


def main():
    import numpy as np
    from core.runtime import configure_single_threaded_blas

    configure_single_threaded_blas()
    from core.transmon import Basis, transmon_spectrum
    from floating_coupler.design import ej_for_frequency
    from integrated_readout_range import frozen_model

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--pair", required=True, choices=["A", "B"])
    p.add_argument("--points", type=int, default=31)
    p.add_argument("--benchmark", action="store_true")
    a = p.parse_args()
    cell = json.loads(a.cell.read_text(encoding="utf-8-sig"))
    base = frozen_model(cell["physical"])
    fc = cell["off_ghz"]
    indices = (0, 1) if a.pair == "A" else (3, 4)
    started = perf_counter()
    a.output.mkdir(parents=True, exist_ok=True)
    nominal = solve_point(base, cell["physical"], fc, a.pair)
    if a.benchmark:
        fast_seconds = perf_counter() - started
        t = perf_counter()
        dense = solve_point(base, cell["physical"], fc, a.pair, dense=True)
        out = dict(
            sparse_seconds=fast_seconds,
            dense_seconds=perf_counter() - t,
            maximum_shift_difference_hz=float(np.max(abs(nominal[0] - dense[0])) * 1e6),
            shifts_mhz=nominal[0].tolist(),
            weights=nominal[1].tolist(),
            matter_reference_weights=nominal[2].tolist(),
        )
        (a.output / f"{a.pair}-benchmark.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out))
        return
    bounds = (4.0, 5.0) if a.pair == "A" else (4.4, 5.8)
    axes = [
        np.linspace(
            ej_for_frequency(bounds[0], base.ec[i], 40),
            ej_for_frequency(bounds[1], base.ec[i], 40),
            a.points,
        )
        for i in indices
    ]
    signature = map_signature(cell, a.pair, a.points)
    checkpoint = a.output / f"{a.pair}-seven-map.npz"
    if checkpoint.exists():
        with np.load(checkpoint) as old:
            assert str(old["signature"]) == signature, (
                "Different map inputs; choose a new output folder"
            )
            shifts, weights, matter, done = (
                old[k].copy()
                for k in ("shift_mhz", "overlap", "matter_overlap", "done")
            )
    else:
        shifts = np.full((2, 3, a.points, a.points), np.nan)
        weights = np.full((2, a.points, a.points), np.nan)
        matter = weights.copy()
        done = np.zeros((a.points, a.points), bool)
    times = []

    def at(ej1, ej2, basis=START):
        ej = base.ej.copy()
        ej[list(indices)] = [ej1, ej2]
        return solve_point(replace(base, ej=ej), cell["physical"], fc, a.pair, basis)

    for j, ej2 in enumerate(axes[1]):
        for i, ej1 in enumerate(axes[0]):
            if done[j, i]:
                continue
            t = perf_counter()
            value = at(ej1, ej2)
            times.append(perf_counter() - t)
            shifts[:, :, j, i], weights[:, j, i], matter[:, j, i] = value
            done[j, i] = True
        temp = checkpoint.with_suffix(".tmp.npz")
        np.savez_compressed(
            temp,
            signature=signature,
            ej1_ghz=axes[0],
            ej2_ghz=axes[1],
            shift_mhz=shifts,
            overlap=weights,
            matter_overlap=matter,
            done=done,
        )
        temp.replace(checkpoint)
        progress = dict(
            pair=a.pair,
            rows_done=j + 1,
            total=a.points,
            points_done=int(done.sum()),
            seconds=perf_counter() - started,
            mean_new_point_seconds=float(np.mean(times)) if times else None,
        )
        (a.output / f"{a.pair}-progress.json").write_text(
            json.dumps(progress, indent=2)
        )
        print(json.dumps(progress), flush=True)
    audit = []
    locations = [
        (axes[0][i], axes[1][j])
        for i in (0, a.points // 2, a.points - 1)
        for j in (0, a.points // 2, a.points - 1)
    ]
    locations.append(tuple(base.ej[list(indices)]))
    for ej1, ej2 in locations:
        low, high = at(ej1, ej2), at(ej1, ej2, AUDIT)
        audit.append(
            dict(
                ej1_ghz=float(ej1),
                ej2_ghz=float(ej2),
                change_khz=((high[0] - low[0]) * 1000).tolist(),
                overlap_low=low[1].tolist(),
                overlap_high=high[1].tolist(),
            )
        )
    meta = dict(
        pair=a.pair,
        cell_input=str(a.cell.resolve()),
        input_cell=cell,
        basis=START,
        audit_basis=AUDIT,
        source_identity=source_identity(),
        solver_policy="sparse-window-v2-overlap-guard",
        coupler_off_bare_ghz=fc,
        coupler_EJ_ghz=cell["off_ej_ghz"],
        off_note="Saved nominal OFF EJ held fixed across grid; no per-pixel null or new 1 Hz OFF certification",
        loaded_transmon_ec_ghz=base.ec[list(indices)].tolist(),
        nominal_idle_ej_ghz=base.ej[list(indices)].tolist(),
        nominal_idle_bare_frequencies_ghz=[
            float(
                transmon_spectrum(
                    float(base.ej[i]),
                    float(base.ec[i]),
                    Basis(ncut=40, transmon_levels=3),
                )[0][1]
            )
            for i in indices
        ],
        nominal_shift_mhz=nominal[0].tolist(),
        nominal_overlap=nominal[1].tolist(),
        nominal_matter_reference_overlap=nominal[2].tolist(),
        spectators=["0L", "1L"],
        photon_states=[0, 1, 2],
        frequency_bounds_ghz=bounds,
        audit=audit,
        elapsed_seconds=perf_counter() - started,
        label_frame="Same full-matter-dressed references as SevenReadout; spectator product identity separately recorded by matter overlap",
        state_order=[
            "00 pair ground with logical spectator",
            "10 lower hybridized pair branch",
            "01 upper hybridized pair branch",
        ],
    )
    (a.output / f"{a.pair}-seven-summary.json").write_text(json.dumps(meta, indent=2))
    print(
        json.dumps(dict(finished=a.pair, seconds=meta["elapsed_seconds"])), flush=True
    )


if __name__ == "__main__":
    main()

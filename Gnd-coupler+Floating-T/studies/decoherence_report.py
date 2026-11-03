"""Six-page optional decoherence report following the approved visual reference."""

from pathlib import Path
import datetime, json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
    Flowable,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pymupdf

WF = Path(__file__).resolve().parents[1]


def figures(out, r):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    folder = out / "figures"
    folder.mkdir(exist_ok=True)
    iso = r["isolated"]
    x = np.array([q["amplitude_microphi0"] for q in iso["gate"]["rows"]])
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.2))
    palette = {"logical_B": "#2266aa", "B20_B0": "#c34740", "B1_01": "#585858"}
    labels = {"logical_B": "Logical B", "B20_B0": "B20-B0", "B1_01": "Isolated B1 0-1"}
    for state, c in palette.items():
        for pos, alpha in [("idle", 0.6), ("gate", 1.0)]:
            for kind, ls in [("echo", "-"), ("ramsey", "--")]:
                axs[0].plot(
                    x,
                    [q["times_us"][state][kind] for q in iso[pos]["rows"]],
                    c=c,
                    ls=ls,
                    alpha=alpha,
                    marker="o" if pos == "idle" else None,
                    markevery=16,
                    ms=3,
                    label=f"{labels[state]}, {pos}, {kind}",
                )
    axs[0].set(
        yscale="log",
        ylabel=r"Pure-dephasing time ($\mu$s)",
        title="Isolated B rail: idle versus gate",
    )
    axs[0].legend(fontsize=8, ncol=2)
    for kind, c in [("echo", "#2266aa"), ("ramsey", "#c34740")]:
        for key, ls, label in [
            ("raw_fidelity", "--", "raw"),
            ("postselected_fidelity", "-", "selected"),
        ]:
            axs[1].plot(
                x,
                [100 * q["fidelity"][kind][key] for q in iso["gate"]["rows"]],
                c=c,
                ls=ls,
                label=kind + ", " + label,
            )
    axs[1].set(
        ylabel="Ideal CZ fidelity (%)",
        title=f"{r['assumptions']['gate_time_ns']:g} ns, full B20-inclusive gate",
    )
    axs[1].legend()
    for ax in axs:
        ax.set(
            xscale="log",
            xlim=(min(x), max(x)),
            xlabel=r"Independent noise per B SQUID ($\mu\Phi_0/\sqrt{\mathrm{Hz}}$ at 1 Hz)",
        )
        ax.grid(alpha=0.2, which="both")
    fig.tight_layout()
    fig.savefig(folder / "isolated-B-dephasing-and-CZ.png", dpi=200)
    plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8))
    for ax, key in zip(axs.ravel()[:3], ["A_idle", "B_idle", "B_gate"]):
        for p, c in zip(r["coupled"][key]["points"], ["#2266aa", "#c34740"]):
            for kind, ls in [("echo", "-"), ("ramsey", "--")]:
                ax.plot(
                    x,
                    p["reference_times_seconds"]["logical"][kind] * 1e6 * 100 / x,
                    c=c,
                    ls=ls,
                    label=p["bias"] + ", " + kind,
                )
        ax.set_title(key.replace("_", " "))
        ax.legend(fontsize=8)
    ax = axs[1, 1]
    for pos, c in [("idle", "#2266aa"), ("gate", "#c34740")]:
        p = next(p for p in r["coupled"]["B_" + pos]["points"] if p["bias"] == "ON")
        for kind, ls in [("echo", "-"), ("ramsey", "--")]:
            ax.plot(
                x,
                p["reference_times_seconds"]["aux_B0"][kind] * 1e6 * 100 / x,
                c=c,
                ls=ls,
                label=pos + ", ON, " + kind,
            )
    ax.set_title("B20-B0, coupler ON")
    ax.legend(fontsize=8)
    for ax in axs.ravel():
        ax.set(
            xscale="log",
            yscale="log",
            xlim=(min(x), max(x)),
            xlabel=r"Coupler noise ($\mu\Phi_0/\sqrt{\mathrm{Hz}}$ at 1 Hz)",
            ylabel=r"Effective pure-dephasing time ($\mu$s)",
        )
        ax.grid(alpha=0.2, which="both")
    fig.tight_layout()
    fig.savefig(folder / "coupler-induced-dephasing.png", dpi=200)
    plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    for key, col, c in [
        ("A_idle", 0, "#2266aa"),
        ("B_idle", 1, "#d27620"),
        ("B_gate", 1, "#488547"),
    ]:
        q = r["coupled"][key]
        xx = np.array([p["fc_ghz"] for p in q["frequency_response"]])
        yy = np.array([p["logical_mhz"] for p in q["frequency_response"]])
        off = q["points"][0]["fc_ghz"]
        on = q["points"][1]["fc_ghz"]
        ref = np.interp(off, xx, yy)
        axs[0, col].plot(xx, yy - ref, c=c, label=key)
        axs[1, col].plot(xx, np.gradient(yy, xx), c=c, label=key)
        for ax in axs[:, col]:
            ax.axvline(off, c="gray", ls="--")
            ax.axvline(on, c="gray", ls=":")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    for k in [0, 1]:
        axs[0, k].set_ylabel("Logical shift from OFF (MHz)")
        axs[1, k].set(ylabel="dfL/dfC (kHz/MHz)", xlabel="Coupler f01 (GHz)")
    fig.tight_layout()
    fig.savefig(folder / "coupler-frequency-response.png", dpi=180)
    plt.close(fig)


def ensure_circuit(out, result):
    """Render the approved circuit template from explicit nominal inputs."""
    circuit = out / "circuit"
    if (circuit / "circuit.pdf").exists():
        return
    from floating_coupler.fabrication import five_model
    from floating_coupler.design import ej_for_frequency

    base = result["input"]
    model, _ = five_model(base, {}, {})
    labels = {}
    for rail, indices in [("A", (0, 1)), ("B", (3, 4))]:
        q = result["coupled"][rail + "_idle"]
        delta = q["differential_bare_ghz"]
        mean = q["mean_bare_ghz"]
        for k, i in enumerate(indices):
            name = model.names[i]
            frequency = mean + (1 if k == 0 else -1) * delta / 2
            labels[name] = dict(
                ec_mhz=float(model.ec[i] * 1000),
                ej_idle_ghz=ej_for_frequency(frequency, model.ec[i], 40),
            )
            if rail == "B":
                labels[name]["ej_gate_ghz"] = ej_for_frequency(
                    result["isolated"]["gate"]["bare_frequency_ghz"]
                    + (1 if k == 0 else -1) * delta / 2,
                    model.ec[i],
                    40,
                )
    cp = result["coupled"]["B_gate"]["points"]
    cs = result["c_squid_design"]
    readouts = {}
    for rail, ro in base["readouts"].items():
        ph = ro["physical"]
        rn = ro["case"]["resonator_node"]
        readouts[rail] = dict(
            links=[e for e in ph["physical_links"] if rn in [e["from"], e["to"]]],
            capacitance_ff=ph["resonator_self_capacitance_ff"],
            inductance_nh=ph["resonator_inductance_nh"],
            target_frequency_ghz=ro["case"]["readout_target_ghz"],
        )
    parameters = dict(
        physical=base["five_physical"],
        rail_labels=labels,
        readouts=readouts,
        coupler_labels=dict(
            ec_mhz=float(model.ec[2] * 1000),
            ej_off_ghz=ej_for_frequency(cp[0]["fc_ghz"], model.ec[2], 40),
            ej_on_ghz=ej_for_frequency(cp[1]["fc_ghz"], model.ec[2], 40),
            ej1_ghz=cs["ej_large_ghz"],
            ej2_ghz=cs["ej_small_ghz"],
        ),
    )
    circuit.mkdir(exist_ok=True)
    path = circuit / "parameters.json"
    path.write_text(json.dumps(parameters, indent=2))
    template = WF / "templates/circuit-schematic.py.in"
    source = (
        template.read_text(encoding="utf-8-sig")
        .replace("r'{{OUTPUT_DIRECTORY}}'", repr(str(circuit)))
        .replace("r'{{PARAMETERS_JSON}}'", repr(str(path)))
    )
    exec(compile(source, str(template), "exec"), {"__name__": "__main__"})
    (circuit / "transmon-circuit.pdf").replace(circuit / "circuit.pdf")


def generate(folder):
    out = Path(folder).resolve()
    r = json.loads((out / "results.json").read_text())
    ensure_circuit(out, r)
    figures(out, r)
    pdfmetrics.registerFont(
        TTFont(
            "StudyDejaVu",
            str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"),
        )
    )
    styles = getSampleStyleSheet()
    styles["Normal"].fontName = "StudyDejaVu"
    styles["Normal"].fontSize = 9
    styles["Normal"].leading = 12
    story = []
    width = A4[0] - 84

    def txt(s):
        story.extend([Paragraph(str(s), styles["Normal"]), Spacer(1, 8)])

    def title(s):
        story.extend([Paragraph(s, styles["Heading1"]), Spacer(1, 10)])

    def page():
        story.append(PageBreak())

    def image(name):
        from PIL import Image as PILImage

        path = out / "figures" / name
        w, h = PILImage.open(path).size
        story.extend(
            [Image(str(path), width=width, height=width * h / w), Spacer(1, 8)]
        )

    def table(head, rows):
        cells = [
            [Paragraph(str(v), styles["Normal"]) for v in row] for row in [head] + rows
        ]
        t = Table(cells, colWidths=[width / len(head)] * len(head), repeatRows=1)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7eff5")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.extend([t, Spacer(1, 8)])

    iso = r["isolated"]
    a = r["assumptions"]
    selected = [
        min(iso["gate"]["rows"], key=lambda q: abs(q["amplitude_microphi0"] - amp))
        for amp in [1, 10, 100]
    ]
    selected = list({q["amplitude_microphi0"]: q for q in selected}.values())
    title("B-rail and coupler-induced dephasing")
    image("isolated-B-dephasing-and-CZ.png")
    txt(
        "Central result: isolated B rail, independent intrinsic SQUID flux noise. Logical B, B20-B0 and isolated physical B1 references are shown at idle and gate. Every fidelity curve includes B20. No coupler noise is added to this central gate estimate; A is noiseless."
    )
    table(
        [
            "Noise per B loop",
            "Echo raw",
            "Echo selected",
            "Ramsey raw",
            "Ramsey selected",
        ],
        [
            [
                f"{q['amplitude_microphi0']:g}",
                *[
                    f"{100 * q['fidelity'][k][field]:.4f}%"
                    for k in ["echo", "ramsey"]
                    for field in ["raw_fidelity", "postselected_fidelity"]
                ],
            ]
            for q in selected
        ],
    )
    txt(a["gate_model"])
    page()
    title("1 | Circuit context and fixed inputs")
    circuit = out / "circuit/circuit.pdf"
    box = []

    class Placeholder(Flowable):
        def __init__(self):
            super().__init__()
            self.width = width
            self.height = width * 7.26 / 14

        def draw(self):
            m = self.canv._currentMatrix
            box[:] = [m[4], A4[1] - m[5] - self.height, m[4] + self.width, A4[1] - m[5]]

    if circuit.exists():
        story.append(Placeholder())
    txt(
        "Circuit drawing is contextual. The main B-rail calculation omits C and resonators while retaining loaded EC and intra-pair charge coupling. The separate C-noise calculation includes one rail pair and C at a time. No full seven-mode noise model is implied."
    )
    table(
        ["Input", "Value"],
        [
            [
                "B idle / gate frequency (GHz)",
                f"{iso['idle']['bare_frequency_ghz']:.5f} / {iso['gate']['bare_frequency_ghz']:.5f}",
            ],
            ["Rail SQUID range (GHz)", str(a["rail_tuning"])],
            ["C SQUID range (GHz)", str(a["coupler_tuning"])],
            ["Ideal gate duration (ns)", a["gate_time_ns"]],
            ["Ramsey infrared cutoff (Hz)", a["low_cutoff_hz"]],
            ["Local transmon levels", a["levels"]],
        ],
    )
    page()
    title("2 | Isolated B-rail details")
    table(
        ["SQUID parameter", "B1", "B2"],
        [
            [
                "Loaded EC/h (MHz)",
                *[f"{d['ec_ghz'] * 1000:.2f}" for d in r["b_squid_designs"]],
            ],
            [
                "Large junction EJ/h (GHz)",
                *[f"{d['ej_large_ghz']:.2f}" for d in r["b_squid_designs"]],
            ],
            [
                "Small junction EJ/h (GHz)",
                *[f"{d['ej_small_ghz']:.2f}" for d in r["b_squid_designs"]],
            ],
            ["Asymmetry", *[f"{d['asymmetry']:.4f}" for d in r["b_squid_designs"]]],
        ],
    )
    values = []
    for pos in ["idle", "gate"]:
        q = min(iso[pos]["rows"], key=lambda q: abs(q["amplitude_microphi0"] - 10))
        for key, label in [
            ("logical_B", "Logical B"),
            ("B20_B0", "B20-B0"),
            ("B1_01", "Isolated B1 0-1"),
        ]:
            values.append(
                [
                    pos,
                    label,
                    f"{q['times_us'][key]['echo']:.3g}",
                    f"{q['times_us'][key]['ramsey']:.3g}",
                ]
            )
    table(["Position", "Coherence", "Echo (us)", "Ramsey (us)"], values)
    txt(
        "Times use the sampled amplitude nearest10 microPhi0/sqrt(Hz) at1Hz. The isolated B1 reference removes B2 and C at the same B1 frequency and loaded EC. B20-B0 is the auxiliary phase coherence across the B component of the selected logical10/B20 transition, not a population lifetime."
    )
    txt(
        "Both bare B frequencies are calibrated to resonance independently at idle and gate. Differential detuning is first-order protected, but flux dependence of charge matrix elements changes the pair coupling. This calculation includes first-order longitudinal sensitivity only; quadratic and transverse noise are excluded."
    )
    page()
    title("3 | Separate coupler-induced dephasing")
    image("coupler-frequency-response.png")
    txt(
        "Rail EJs are noiseless. Differential rail bias is calibrated once at idle with C OFF, then held fixed for the common move to gate. Only C EJ changes along the sweep. Dashed/dotted lines denote OFF/ON. This differs from independently resonating the isolated B pair at every operating point; the comparison is not an identical-control subtraction."
    )
    page()
    title("4 | Coupler-induced dephasing times")
    image("coupler-induced-dephasing.png")
    txt(
        "Times are in microseconds. Reference amplitude100 microPhi0/sqrt(Hz) at1Hz is scaled inversely with amplitude for the adopted exponential-rate convention. The Ramsey cutoff logarithm is held at its reference value in this separate C-noise figure; isolated B Ramsey times are solved independently at every amplitude. B20 is shown only relative to B0 and only with C ON. No intrinsic rail noise or T1 is added here."
    )
    page()
    title("5 | Relative size, interpretation and checks")
    q = min(iso["gate"]["rows"], key=lambda q: abs(q["amplitude_microphi0"] - 100))
    cp = next(p for p in r["coupled"]["B_gate"]["points"] if p["bias"] == "ON")
    comparison = []
    for key, cname, label in [
        ("logical_B", "logical", "Logical B"),
        ("B20_B0", "aux_B0", "B20-B0"),
    ]:
        rail = q["times_us"][key]["echo"] * q["amplitude_microphi0"] / 100
        ctime = cp["reference_times_seconds"][cname]["echo"] * 1e6
        comparison.append([label, f"{rail:.2f}", f"{ctime:.2f}", f"{ctime / rail:.2f}"])
    table(
        ["Gate coherence", "Rail Tphi (us)", "C Tphi (us)", "Rail rate / C rate"],
        comparison,
    )
    txt(
        "Compare the listed rate ratios rather than assume a uniform order-of-magnitude reduction. Static dressing and different calibration conventions can matter separately from adding a noise source. Combined-noise decoherence and fidelity are intentionally omitted from this report."
    )
    txt(
        "Gate rates are phenomenological exponential substitutions of echo/Ramsey first-order 1/f reference times. Physical phase correlations among B0/B1/B20 are retained and rate matrices are checked for complete positivity. Raw fidelity, survival and survival-weighted selected fidelity are separate metrics. No T1, thermal population, gate-matrix-element noise, pulse ramps, baseline coherent error or new full-device gate calibration is included. All inputs and derivatives remain in results.json."
    )

    def footer(c, d):
        c.setFont("Helvetica", 7)
        c.drawString(42, 23, "Optional decoherence study | independent rail subsystems")
        c.drawRightString(A4[0] - 42, 23, str(d.page))

    pdf = out / (
        "decoherence-report-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
        + ".pdf"
    )
    SimpleDocTemplate(
        str(pdf),
        pagesize=A4,
        leftMargin=42,
        rightMargin=42,
        topMargin=38,
        bottomMargin=38,
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    if box:
        doc = pymupdf.open(pdf)
        reference = pymupdf.open(circuit)
        doc[1].show_pdf_page(pymupdf.Rect(box), reference, 0)
        temp = pdf.with_suffix(".vector.pdf")
        doc.save(temp, garbage=4, deflate=True)
        doc.close()
        reference.close()
        temp.replace(pdf)
    result = dict(pdf=str(pdf), pages=len(pymupdf.open(pdf)))
    (out / "report-status.json").write_text(json.dumps(result, indent=2))
    return result

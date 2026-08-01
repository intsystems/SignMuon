"""What machine did this run on? Captured per run, because it varies.

Different experiments in this project are run on different GPUs, so a single
"Computing infrastructure" sentence in the paper would be wrong for most of the
table. The fix is to stamp the hardware into every run's ``metrics.json`` at the
moment it runs (``common.utils.save_run`` does this automatically) and to build
the paper's table from those records rather than from whichever machine happens
to compile the LaTeX.

Three entry points::

    python3 -m common.hardware                  # this machine, human readable
    python3 -m common.hardware --latex          # this machine, as a LaTeX row
    python3 -m common.hardware --scan results   # every machine used, as a table

The last one is the one that fills the paper's reproducibility appendix: it walks
the results tree, groups runs by experiment family and by the hardware they
recorded, and emits one row per (experiment, machine) pair.

**Deliberately anonymous.** Hostname, username and absolute paths are *not*
collected, so the record is safe to paste into a double-blind submission and
cannot trip ``anonymize.py``. GPU and CPU model names are hardware, not
identity, and are what a reader needs to size the compute.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

__all__ = ["describe", "as_sentence", "as_latex_row", "scan_results", "summarize"]

CODE_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------


def _cpu_model() -> Optional[str]:
    """Marketing name of the CPU, without the core count."""
    try:
        if platform.system() == "Linux":
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=10)
            return out.stdout.strip() or None
        else:                                               # Windows
            return platform.processor() or None
    except Exception:                                       # noqa: BLE001
        return None
    return None


def _cpu_count() -> Optional[int]:
    try:
        import os
        return os.cpu_count()
    except Exception:                                       # noqa: BLE001
        return None


def _ram_gb() -> Optional[float]:
    try:
        import os
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            return round(total / 1024 ** 3, 1)
        if platform.system() == "Windows":
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _Status()
            st.dwLength = ctypes.sizeof(_Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return round(st.ullTotalPhys / 1024 ** 3, 1)
    except Exception:                                       # noqa: BLE001
        return None
    return None


def _driver_version() -> Optional[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        first = out.stdout.strip().splitlines()
        return first[0].strip() if first else None
    except Exception:                                       # noqa: BLE001
        return None


def _git(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.run(["git"] + cmd, cwd=CODE_ROOT, capture_output=True,
                             text=True, timeout=30)
        return out.stdout.strip() or None
    except Exception:                                       # noqa: BLE001
        return None


def describe(device: Optional[str] = None) -> Dict[str, Any]:
    """Everything needed to reproduce the compute, and nothing identifying.

    ``device`` is the device the caller actually used (e.g. ``"cuda:0"``); the
    GPU reported is that device, not merely device 0, which matters on a box
    with mixed cards. Every field degrades to ``None`` rather than raising, so
    this can be called unconditionally from a training loop.
    """
    info: Dict[str, Any] = OrderedDict()
    info["python"] = sys.version.split()[0]
    info["os"] = f"{platform.system()} {platform.release()}"
    info["cpu"] = _cpu_model()
    info["cpu_count"] = _cpu_count()
    info["ram_gb"] = _ram_gb()
    info["device_requested"] = device

    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        try:
            info["cudnn"] = torch.backends.cudnn.version()
        except Exception:                                   # noqa: BLE001
            info["cudnn"] = None
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            idx = 0
            if isinstance(device, str) and device.startswith("cuda") and ":" in device:
                try:
                    idx = int(device.split(":", 1)[1])
                except ValueError:
                    idx = torch.cuda.current_device()
            elif device in (None, "cuda"):
                idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            info["gpu"] = props.name
            info["gpu_memory_gb"] = round(props.total_memory / 1024 ** 3, 1)
            info["gpu_capability"] = f"{props.major}.{props.minor}"
            info["gpu_index"] = idx
            info["gpu_count"] = torch.cuda.device_count()
            # A mixed-GPU box is worth recording explicitly: it is how a run
            # silently lands on the wrong card.
            names = {torch.cuda.get_device_properties(i).name
                     for i in range(torch.cuda.device_count())}
            info["gpu_homogeneous"] = len(names) == 1
            info["driver"] = _driver_version()
    except Exception as exc:                                # noqa: BLE001
        info["torch_error"] = f"{type(exc).__name__}: {exc}"

    info["git_commit"] = (_git(["rev-parse", "--short", "HEAD"]) or None)
    info["git_dirty"] = bool(_git(["status", "--porcelain"]))
    return info


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def as_sentence(info: Dict[str, Any]) -> str:
    """One prose sentence, the form the reproducibility appendix wants."""
    gpu = info.get("gpu")
    mem = info.get("gpu_memory_gb")
    gpu_s = f"{gpu} ({mem:g} GB)" if gpu and mem else (gpu or "CPU only")
    bits = [f"Experiments were run on {gpu_s}"]
    if info.get("cpu"):
        cores = f", {info['cpu_count']} cores" if info.get("cpu_count") else ""
        bits.append(f"with a {info['cpu']}{cores}")
    if info.get("ram_gb"):
        bits.append(f"and {info['ram_gb']:g} GB of system memory")
    tail = []
    if info.get("os"):
        tail.append(info["os"])
    if info.get("python"):
        tail.append(f"Python {info['python']}")
    if info.get("torch"):
        tail.append(f"PyTorch {info['torch']}")
    if info.get("cuda"):
        tail.append(f"CUDA {info['cuda']}")
    if info.get("driver"):
        tail.append(f"driver {info['driver']}")
    sentence = " ".join(bits) + ". Software: " + ", ".join(tail) + "."
    return _ascii(sentence)


def _ascii(s: Any) -> str:
    """Plain ASCII, so the record survives a cp1251 console and a LaTeX run.

    Vendor strings occasionally carry a registered-trademark sign or a
    non-breaking space, either of which is a mystery build failure later.
    """
    text = str(s)
    for ch in (" ", " ", " ", " "):
        text = text.replace(ch, " ")
    text = text.replace("™", "").replace("®", "").replace("©", "")
    text = text.replace("–", "-").replace("—", "--")
    return text.encode("ascii", "ignore").decode("ascii").strip()


def _tex(s: Any) -> str:
    """Escape the few characters that actually occur in hardware strings."""
    if s is None:
        return "--"
    return (_ascii(s).replace("\\", "/").replace("&", r"\&")
            .replace("_", r"\_").replace("%", r"\%").replace("#", r"\#"))


def as_latex_row(experiment: str, info: Dict[str, Any]) -> str:
    gpu = info.get("gpu") or "CPU only"
    mem = info.get("gpu_memory_gb")
    # The thin space is assembled *after* escaping. ``_tex`` maps a backslash to
    # a slash -- vendor strings contain Windows paths, not TeX -- so building
    # "15.6\,GB" first and escaping it afterwards yields "15.6/,GB".
    gpu_s = _tex(gpu) + (f", {mem:g}\\,GB" if mem else "")
    sw = " / ".join(x for x in (info.get("torch"),
                                f"CUDA {info['cuda']}" if info.get("cuda") else None)
                    if x)
    return (f"{_tex(experiment)} & {gpu_s} & {_tex(info.get('cpu'))} & "
            f"{_tex(info.get('ram_gb'))} & {_tex(info.get('os'))} & "
            f"{_tex(info.get('python'))} & {_tex(sw)} \\\\")


LATEX_HEADER = r"""\begin{tabular}{@{}lllllll@{}}
\toprule
\textbf{Experiment} & \textbf{GPU} & \textbf{CPU} & \textbf{RAM (GB)} &
\textbf{OS} & \textbf{Python} & \textbf{PyTorch / CUDA} \\
\midrule"""

LATEX_FOOTER = r"""\bottomrule
\end{tabular}"""


# --------------------------------------------------------------------------
# Aggregation across a results tree
# --------------------------------------------------------------------------

#: Which experiment a results path belongs to. First match wins, so the more
#: specific patterns come first.
FAMILIES: Tuple[Tuple[str, str], ...] = (
    (r"synthetic", "Synthetic quadratic"),
    (r"federated", "Federated CIFAR-10 (CNN2)"),
    (r"centralized|cifar", "Centralized CIFAR-10 (ResNet-18)"),
    (r"nanogpt|speedrun", "Language modelling (nanoGPT)"),
    (r"counterexample", "Counterexamples (CPU)"),
)


def family_of(path: Path) -> str:
    posix = path.as_posix().lower()
    for pattern, label in FAMILIES:
        if re.search(pattern, posix):
            return label
    return "Other"


def _hardware_records(root: Path) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    """Every ``hardware`` block under ``root``, from any JSON that carries one."""
    for path in sorted(root.rglob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                blob = json.load(f)
        except Exception:                                   # noqa: BLE001
            continue
        for holder in (blob, blob.get("config") if isinstance(blob, dict) else None,
                       blob.get("environment") if isinstance(blob, dict) else None):
            if isinstance(holder, dict) and isinstance(holder.get("hardware"), dict):
                yield path, holder["hardware"]
                break


def _key(info: Dict[str, Any]) -> Tuple:
    return (info.get("gpu"), info.get("gpu_memory_gb"), info.get("cpu"),
            info.get("ram_gb"), info.get("os"), info.get("python"),
            info.get("torch"), info.get("cuda"))


def scan_results(root: Path) -> "OrderedDict[str, List[Tuple[Dict[str, Any], int]]]":
    """``{experiment: [(hardware, n_runs), ...]}`` over a results tree."""
    seen: Dict[str, Dict[Tuple, Tuple[Dict[str, Any], int]]] = OrderedDict()
    for path, info in _hardware_records(root):
        fam = family_of(path.relative_to(root) if root in path.parents else path)
        bucket = seen.setdefault(fam, OrderedDict())
        k = _key(info)
        prev = bucket.get(k)
        bucket[k] = (info, (prev[1] if prev else 0) + 1)
    return OrderedDict((fam, list(b.values())) for fam, b in seen.items())


def summarize(root: Path, latex: bool) -> str:
    grouped = scan_results(root)
    if not grouped:
        return (f"No hardware records under {root}. Runs written before this "
                f"module existed do not carry one; re-run, or fill the table by "
                f"hand from `python3 -m common.hardware` on each machine.")
    lines: List[str] = []
    if latex:
        lines.append(LATEX_HEADER)
        for fam, entries in grouped.items():
            for info, n in entries:
                lines.append(as_latex_row(fam, info))
        lines.append(LATEX_FOOTER)
        return "\n".join(lines)
    for fam, entries in grouped.items():
        lines.append(f"{fam}:")
        for info, n in entries:
            lines.append(f"  [{n} run{'s' if n != 1 else ''}] {as_sentence(info)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default=None,
                    help="the device whose GPU to report (e.g. cuda:1); "
                         "defaults to the current one")
    ap.add_argument("--latex", action="store_true",
                    help="emit LaTeX rather than prose")
    ap.add_argument("--scan", type=Path, default=None, metavar="RESULTS",
                    help="summarize every machine recorded under a results tree "
                         "instead of describing this one")
    ap.add_argument("--json", action="store_true", help="dump the raw record")
    args = ap.parse_args()

    if args.scan is not None:
        print(summarize(args.scan, args.latex))
        return

    info = describe(args.device)
    if args.json:
        print(json.dumps(info, indent=2))
    elif args.latex:
        print(LATEX_HEADER)
        print(as_latex_row("(this machine)", info))
        print(LATEX_FOOTER)
    else:
        print(as_sentence(info))
        if info.get("gpu") and not info.get("gpu_homogeneous", True):
            print(f"NOTE: this box has {info['gpu_count']} GPUs of differing "
                  f"models; the run used index {info.get('gpu_index')} "
                  f"({info['gpu']}). Record the device you pass, not just the host.")
        if info.get("git_dirty"):
            print("NOTE: the working tree is dirty, so `git_commit` does not "
                  "fully describe the code that ran.")


if __name__ == "__main__":
    main()

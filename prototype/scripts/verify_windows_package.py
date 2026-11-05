"""Smoke-test the actual release ZIP with a clean PATH/cache and Cyrillic paths.

    .venv/Scripts/python prototype/scripts/verify_windows_package.py dist/geo4d-editor-0.2.0-alpha-win64.zip
Python drives the checks; the packaged launcher/executable must run independently.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geo4d.results import new_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    a = ap.parse_args()
    archive = a.archive.resolve()
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert archive.with_suffix(".zip.sha256").read_text(encoding="ascii").split()[0] == digest
    result = new_run(5, "release_0.2.0_alpha", {"archive": archive.name, "res": [96, 54], "frames": 4,
                                               "clean_path": True, "clean_cache": True})
    checks = []
    with tempfile.TemporaryDirectory(prefix="Релиз без Python ", dir=archive.parent) as tmp:
        td = Path(tmp)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for name in z.namelist():
                assert (td / name).resolve().is_relative_to(td)
            z.extractall(td)
        inst = td / archive.stem
        assert not (inst / "prototype").exists() and not (inst / "requirements.txt").exists()
        assert all((inst / p).exists() for p in ["geo4d_gpu.exe", "run_editor.cmd", "nodes/fields.json",
                                                "shaders/pathtrace.slang", "scenes/bubble.g4d"])
        before = {str(p.relative_to(inst)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in inst.rglob("*") if p.is_file()}
        env = {k: v for k, v in os.environ.items() if not k.startswith("GEO4D_") and k != "VULKAN_SDK"}
        env["PATH"] = str(Path(os.environ["SystemRoot"]) / "System32")
        env["GEO4D_CACHE_DIR"] = str(td / "чистый кэш")
        for name, scene in [("default", None), ("bubble", inst / "scenes" / "bubble.g4d")]:
            run = td / name
            run.mkdir()
            script = run / "script.txt"
            np.savetxt(script, np.zeros((4, 6)), fmt="%.6f")
            extra = run / "extra.txt"
            extra.write_text(f'hidden\nbench "{script.as_posix()}" "{(run / "frames.json").as_posix()}"\n'
                             f'dump 3 "{run.as_posix()}"\n', encoding="utf-8")
            # cmd /c invokes the actual packaged launcher from a different working directory.
            args = [str(inst / "run_editor.cmd")]
            if scene:
                args.append(str(scene))
            args += ["--res", "96x54", "--cube", "64", "--extra", str(extra)]
            command = '"' + subprocess.list2cmdline(args) + '"'
            r = subprocess.run([os.environ["COMSPEC"], "/d", "/s", "/c", command], cwd=td, env=env,
                               capture_output=True, text=True, encoding="utf-8", timeout=180)
            (result / f"{name}.log").write_text(r.stdout + r.stderr, encoding="utf-8")
            assert r.returncode == 0, r.stdout + r.stderr
            frame = np.load(run / "Present_3.npy")
            assert frame.shape == (54, 96) and (frame & 0xFFFFFF).std() > 0
            assert len(json.loads((run / "frames.json").read_text(encoding="utf-8"))) == 4
            from matplotlib import image as mpl_image
            rgb = np.stack([(frame >> shift) & 255 for shift in (0, 8, 16)], axis=-1).astype(np.uint8)
            mpl_image.imsave(result / f"{name}.png", rgb)
            checks.append({"scene": name, "exit_code": r.returncode, "frame_shape": list(frame.shape),
                           "nonuniform": True})
        after = {str(p.relative_to(inst)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in inst.rglob("*") if p.is_file()}
        assert before == after, "installed files were modified"
        assert list((td / "чистый кэш" / "spv").glob("*.spv"))
        source = json.loads((inst / "build.json").read_text(encoding="utf-8"))
    report = {"archive": archive.name, "sha256": digest, "size_bytes": archive.stat().st_size,
              "build": source, "checks": checks, "installed_files_unchanged": True}
    (result / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

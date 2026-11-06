"""
Robust per-model eval result uploader.

After every model × dataset run, call:
    checkpoint(model_name, dataset_name, out_dir, metrics_dict)

which:
1. Packs out_dir as a tarball → uploads to HF aaditya/phase0-artifacts
2. Writes a JSON metrics file → uploads to HF (fast, for quick inspection)
3. Appends a one-row summary to RESULTS_LIVE.md → commits + pushes to GitHub
4. Prints a visible confirmation block

Designed so that a SIGKILL between models loses at most the ONE currently-running
model, never finished ones.
"""
import os, subprocess, json, tarfile, time, traceback
from pathlib import Path
from huggingface_hub import HfApi

HF_REPO = "aaditya/phase0-artifacts"
HF_REPO_TYPE = "dataset"
GH_REPO_DIR = "/content/phase0-eval"
LIVE_MD = f"{GH_REPO_DIR}/RESULTS_LIVE.md"


def _tar_dir(src_dir: str, tar_path: str) -> str:
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(src_dir, arcname=os.path.basename(src_dir))
    return tar_path


def _hf_upload(local_path: str, remote_path: str, token: str) -> str:
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=remote_path,
        repo_id=HF_REPO,
        repo_type=HF_REPO_TYPE,
    )
    return f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{remote_path}"


def _git_commit_push(msg: str, files: list) -> bool:
    try:
        for f in files:
            subprocess.run(["git", "-C", GH_REPO_DIR, "add", f], check=True)
        subprocess.run(
            ["git", "-C", GH_REPO_DIR, "commit", "-m", msg],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(["git", "-C", GH_REPO_DIR, "push"], check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  git push FAILED: {e.stderr if hasattr(e, "stderr") else e}")
        return False


def _append_live_md(row: dict):
    """Append a pipe-separated row to RESULTS_LIVE.md. Header added once."""
    header = "| ts | model | dataset | n | pa_mpjpe | mpjpe | pa_pve | mrrpe | runtime_s | hf_tar |"
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    if not os.path.exists(LIVE_MD):
        with open(LIVE_MD, "w") as f:
            f.write("# Phase 0 live eval results\n\n")
            f.write("Auto-appended by `phase0.upload.checkpoint()` after every model × dataset run.\n")
            f.write("Never hand-edit.\n\n")
            f.write(header + "\n" + sep + "\n")
    vals = [
        row.get("ts", ""),
        row.get("model", ""),
        row.get("dataset", ""),
        str(row.get("n", "")),
        f'{row.get("pa_mpjpe", float("nan")):.2f}',
        f'{row.get("mpjpe", float("nan")):.2f}',
        f'{row.get("pa_pve", float("nan")):.2f}',
        f'{row.get("mrrpe", float("nan")):.2f}',
        f'{row.get("runtime_s", 0):.0f}',
        f'[tar]({row.get("hf_tar", "")})',
    ]
    with open(LIVE_MD, "a") as f:
        f.write("| " + " | ".join(vals) + " |\n")


def checkpoint(model: str, dataset: str, out_dir: str, metrics: dict,
               hf_token: str = None, run_tag: str = None, extra_files: list = None):
    """
    Atomic per-model checkpoint. Call after each (model, dataset) finishes.

    Args:
        model:      e.g., "hamer", "wildhands", "jointtransformer"
        dataset:    e.g., "aria_val_mps", "pov_test"
        out_dir:    directory with per-sample pkls + any viz
        metrics:    dict with keys: n, pa_mpjpe, mpjpe, pa_pve, mrrpe, runtime_s, ...
        hf_token:   defaults to os.environ["HF_TOKEN"]
        run_tag:    optional tag for this batch (e.g., "rsf15_leverD"). Defaults to timestamp.
        extra_files: additional file paths to include in the upload
    """
    hf_token = hf_token or os.environ["HF_TOKEN"]
    ts = time.strftime("%Y%m%dT%H%M%S")
    tag = run_tag or ts
    key = f"{model}__{dataset}__{tag}"
    print("\n" + "=" * 70)
    print(f"CHECKPOINT  {key}")
    print("=" * 70)

    # 1. Pack + upload tarball
    try:
        tar_local = f"/tmp/{key}.tar.gz"
        _tar_dir(out_dir, tar_local)
        size_mb = os.path.getsize(tar_local) / 1e6
        print(f"  [1/4] tar created: {tar_local} ({size_mb:.1f} MB)")
        url = _hf_upload(tar_local, f"runs/{key}.tar.gz", hf_token)
        print(f"  [2/4] HF upload OK: {url}")
    except Exception as e:
        print(f"  [!] tar/upload FAILED: {e}")
        traceback.print_exc()
        url = ""

    # 2. Upload metrics JSON separately (fast inspection)
    try:
        json_local = f"/tmp/{key}.json"
        with open(json_local, "w") as f:
            json.dump({"model": model, "dataset": dataset, "tag": tag, **metrics}, f, indent=2)
        _hf_upload(json_local, f"metrics/{key}.json", hf_token)
        print(f"  [3/4] metrics JSON uploaded")
    except Exception as e:
        print(f"  [!] JSON upload FAILED: {e}")

    # 3. Append to RESULTS_LIVE.md + git push
    try:
        _append_live_md({
            "ts": ts, "model": model, "dataset": dataset,
            "hf_tar": url, **metrics,
        })
        ok = _git_commit_push(
            f"eval: {key} - n={metrics.get('n')} pa={metrics.get('pa_mpjpe', float('nan')):.2f}",
            [LIVE_MD],
        )
        print(f"  [4/4] git push {'OK' if ok else 'FAILED'}")
    except Exception as e:
        print(f"  [!] git step FAILED: {e}")
        traceback.print_exc()

    print(f"CHECKPOINT DONE  {key}\n")
    return {"key": key, "hf_tar": url, "metrics": metrics}

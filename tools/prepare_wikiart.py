"""Download WikiArt from Hugging Face and convert it to JPEG files plus a CSV.

The dataset is published as 72 Parquet shards (~520 MB each). Shards are
processed one at a time and deleted afterwards, so disk usage stays low.
The script is resumable: shards that are already done are skipped.

Output:
    <out>/images/SS_RRRR.jpg   shortest side resized to --size pixels
    <out>/wikiart.csv          file,style,artist,genre,split
"""

import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

NUM_SHARDS = 72
SHARD_URL = ("https://huggingface.co/datasets/huggan/wikiart/resolve/main/"
             "data/train-{:05d}-of-00072.parquet")
INFO_URL = "https://datasets-server.huggingface.co/info?dataset=huggan/wikiart"
CSV_HEADER = ["file", "style", "artist", "genre", "split"]

def load_label_names():
    """Parquet stores labels as integers; the names live in the dataset info."""
    with urllib.request.urlopen(INFO_URL, timeout=60) as r:
        features = json.load(r)["dataset_info"]["default"]["features"]
    return {k: features[k]["names"] for k in ("style", "artist", "genre")}

def assign_split(artist, file_name):
    """Deterministic 80/10/10 split, grouped by artist.

    All works of a known artist land in the same split, so the model cannot
    score well on the test set just by recognising an artist's hand.
    """

    key = file_name if artist == "Unknown Artist" else artist
    bucket = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "val"
    return "train"

def download(url, dest):
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        start = time.monotonic()

        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)

            elapsed = time.monotonic() - start
            rate = done / elapsed if elapsed > 0 else 0.0

            if total and rate > 0:
                pct = 100 * done / total
                eta = (total - done) / rate
                print(f"\r    {done / 1e6:4.0f} / {total / 1e6:.0f} MB"
                      f"  {pct:5.1f}%  {rate / 1e6:5.1f} MB/s"
                      f"  ostalo {eta:4.0f} s", end="", flush=True)
        print()

def resize_shortest(img, size):
    w, h = img.size
    scale = size / min(w, h)
    if scale >= 1.0:
        return img
    return img.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)

def process_shard(shard, parquet_path, images_dir, part_path, names, size, quality):
    rows = []
    skipped = 0
    index = 0

    with pq.ParquetFile(parquet_path) as pf:
        for batch in pf.iter_batches(batch_size=64, 
                                        columns=["image", "style", "artist", "genre"]):
            for rec in batch.to_pylist():
                file_name = f"{shard:02d}_{index:04d}.jpg"
                index += 1

                try:
                    with Image.open(io.BytesIO(rec["image"]["bytes"])) as img:
                        small = resize_shortest(img.convert("RGB"), size)
                        small.save(images_dir / file_name, "JPEG", quality=quality)
                except Exception as e:
                    print(f"Failed to process {file_name}: {e}", file=sys.stderr)
                    skipped += 1
                    continue

                artist = names["artist"][rec["artist"]]
                rows.append([file_name, names["style"][rec["style"]], artist, names["genre"][rec["genre"]],assign_split(artist, file_name),])

    tmp = part_path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    tmp.replace(part_path)
    return len(rows), skipped

def merge_parts(parts_dir, csv_path):
    parts = sorted(parts_dir.glob("shard-*.csv"))
    total = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for part in parts:
            with open(part, newline="", encoding="utf-8") as pf:
                reader = csv.reader(pf)
                for row in reader:
                    writer.writerow(row)
                    total += 1
    return len(parts), total

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/wikiart"))
    ap.add_argument("--size", type=int, default=256, help="shortest side in pixels")
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality")
    ap.add_argument("--shards", type=int, default=NUM_SHARDS,
                    help="process only the first N shards (for testing)")
    args = ap.parse_args()

    images_dir = args.out / "images"
    parts_dir = args.out / "parts"
    images_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)

    names = load_label_names()

    for shard in range(args.shards):
        tag = f"[{shard + 1}/{args.shards}]"
        part_path = parts_dir / f"shard-{shard:02d}.csv"
        if part_path.exists():
            print(f"{tag} already exists, skipping", flush=True)
            continue

        tmp_parquet = args.out / "_download.parquet"
        print(f"{tag} downloading...", flush=True)
        download(SHARD_URL.format(shard), tmp_parquet)
        try:
            n, skipped = process_shard(shard, tmp_parquet, images_dir, part_path,
                                       names, args.size, args.quality)
        finally:
            tmp_parquet.unlink(missing_ok=True)
        print(f"{tag} {n} pictures, skipped {skipped}", flush=True)

    n_parts, total = merge_parts(parts_dir, args.out / "wikiart.csv")
    print(f"\nwikiart.csv: {total} rows from {n_parts}/{NUM_SHARDS} parts")


if __name__ == "__main__":
    main()



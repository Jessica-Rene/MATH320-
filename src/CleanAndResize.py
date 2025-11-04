import os
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

TARGET_SIZE = (64, 64)

def resizeImage(inFile, outFile, target_size):

    try:
        out_dir = os.path.dirname(outFile)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with Image.open(inFile) as img:
            # Resize to fixed size
            resized = img.resize(target_size, Image.LANCZOS)
            resized.save(outFile)

        return True
    except Exception as e:
        print(f"[ERROR] {inFile} -> {outFile}: {e}")
        return False

stages = ["diestrus", "estrus", "proestrus", "metestrus"]

os.makedirs("clean", exist_ok=True)

for stage in stages:
    print(f"\n=== Processing stage: {stage} ===")

    tsv_path = f"Files/{stage}.tsv"
    if not os.path.exists(tsv_path):
        print(f"[WARNING] TSV not found: {tsv_path} (skipping)")
        continue

    df = pd.read_csv(tsv_path, sep="\t")

    mouse_df = df[df["Species"] == "Mouse"].copy()

    files = list(mouse_df["Files"])
    succeeded = []

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {}

        for file in files:
            # Input: trust the path stored in the TSV (relative to Files/)
            in_path = os.path.join("Files", file)

            # Output: clean/{stage}/{basename}
            out_name = os.path.basename(file)
            out_path = os.path.join("clean", stage, out_name)

            futures[executor.submit(resizeImage, in_path, out_path, TARGET_SIZE)] = file

        for future in as_completed(futures):
            file = futures[future]
            ok = future.result()
            if ok:
                succeeded.append(file)

    # Keep only successfully resized images
    clean_df = mouse_df[mouse_df["Files"].isin(succeeded)].copy()
    clean_df.reset_index(drop=True, inplace=True)

    # Drop unwanted columns
    clean_df = clean_df.drop(columns=["DateAdded", "FileSize_bytes_"])

    # Write per-stage CSV: clean/diestrus.csv, clean/estrus.csv, etc.
    out_csv = os.path.join("clean", f"{stage}.csv")
    clean_df.to_csv(out_csv, index=False)

    print(
        f"{stage}: {len(succeeded)} images resized, "
        f"{len(mouse_df) - len(succeeded)} failed"
    )

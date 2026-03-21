from pathlib import Path
import shutil

OUTPUT_ROOT = Path("output")
TMP_DIR = Path("tmp_md_source")

TMP_DIR.mkdir(exist_ok=True)

copied = 0

for md_file in OUTPUT_ROOT.rglob("*.md"):
    # skip anything already inside tmp folder if it exists under output by mistake
    if TMP_DIR in md_file.parents:
        continue

    target = TMP_DIR / md_file.name

    # avoid overwrite if same filename appears from different docs
    if target.exists():
        new_name = f"{md_file.parent.name}__{md_file.name}"
        target = TMP_DIR / new_name

    shutil.copy2(md_file, target)
    print(f"Copied: {md_file} -> {target}")
    copied += 1

print(f"\nDone. Total markdown files copied: {copied}")
print(f"Testing source folder: {TMP_DIR.resolve()}")
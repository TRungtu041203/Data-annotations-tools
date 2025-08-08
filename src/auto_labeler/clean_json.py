import argparse
from pathlib import Path 

def remove_json(labels_dir: Path):
    count=0
    for json_file in labels_dir.rglob("*.json"):
        json_file.unlink()
        print(f"Remove files: {json_file}")
        count+=1
    print(f"\nRemove {count} .json files from {labels_dir}")

def main():
    p = argparse.ArgumentParser(description="Remove all original .json files from the dir")
    p.add_argument("--labels-dir", type=Path, required=True, help="Path to labels directory")
    args = p.parse_args()
    if not args.labels_dir.exists():
        print(f"Directory {args.labels_dir} does not exist.")
        return
    remove_json(args.labels_dir)

if __name__ == "__main__":
    main()

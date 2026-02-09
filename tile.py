#!/usr/bin/env python3
import cv2
import shutil
import argparse
from pathlib import Path
import sys

def slice_split(split_dir, out_split_dir, tile_size=640, overlap=0.2, min_objects=1, keep_empty=False):
    img_dir = Path(split_dir) / "images"
    lbl_dir = Path(split_dir) / "labels"
    out_img = Path(out_split_dir) / "images"
    out_lbl = Path(out_split_dir) / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    
    image_files = list(img_dir.glob("*.[jp][pn]g"))
    total_images = len(image_files)
    stats = {"total_tiles": 0, "kept_tiles": 0}
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"\r  {idx:3d}/{total_images} images", end="", flush=True)
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        step = int(tile_size * (1 - overlap))
        
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        boxes = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, cx, cy, bw, bh = map(float, parts[:5])
                        boxes.append([int(cls), cx, cy, bw, bh])
        
        tile_id = 0
        for y in range(0, h, step):
            for x in range(0, w, step):
                x2, y2 = min(x + tile_size, w), min(y + tile_size, h)
                tile = img[y:y2, x:x2]
                if tile.size == 0:
                    continue
                
                tile_boxes = []
                for cls, cx, cy, bw, bh in boxes:
                    cx_abs, cy_abs = cx * w, cy * h
                    if x < cx_abs < x2 and y < cy_abs < y2:
                        nx = (cx_abs - x) / (x2 - x)
                        ny = (cy_abs - y) / (y2 - y)
                        nw = (bw * w) / (x2 - x)
                        nh = (bh * h) / (y2 - y)
                        if 0 < nx < 1 and 0 < ny < 1:
                            tile_boxes.append(f"{cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
                
                stats["total_tiles"] += 1
                if keep_empty or len(tile_boxes) >= min_objects:
                    cv2.imwrite(str(out_img / f"{img_path.stem}_{tile_id}.jpg"), tile)
                    if tile_boxes:
                        with open(out_lbl / f"{img_path.stem}_{tile_id}.txt", "w") as f:
                            f.write("\n".join(tile_boxes))
                    stats["kept_tiles"] += 1
                tile_id += 1
    
    print()
    return stats

def slice_yolov8_dataset(input_dir, output_dir, tile_size=640, overlap=0.2, min_objects=1, keep_empty=False):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    
    yaml_src = input_dir / "data.yaml" # data.yaml copy
    if yaml_src.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(yaml_src, output_dir / "data.yaml")
    
    total_stats = {"total_tiles": 0, "kept_tiles": 0}
    for split in ["train", "valid", "val", "test"]:
        split_dir = input_dir / split
        if split_dir.exists() and (split_dir / "images").exists():
            print(f"\n✂️ {split.upper()}")
            stats = slice_split(
                split_dir,
                output_dir / split,
                tile_size,
                overlap,
                min_objects,
                keep_empty
            )
            total_stats["total_tiles"] += stats["total_tiles"]
            total_stats["kept_tiles"] += stats["kept_tiles"]
    
    print("\n" + "="*50)
    print(f"✅ Saved {total_stats['kept_tiles']}/{total_stats['total_tiles']} images.")
    print(f"📁 Result: {output_dir.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Tile YOLOv8 dataset")
    parser.add_argument("--input", required=True, help="Input dataset directory (with train/valid/test)")
    parser.add_argument("--output", required=True, help="Output directory for sliced dataset")
    parser.add_argument("--size", type=int, default=640, help="Tile size in pixels (default: 640)")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap ratio 0.0-1.0 (default: 0.2)")
    parser.add_argument("--keep-empty", action="store_true", help="Keep tiles without objects")
    parser.add_argument("--min-objects", type=int, default=1, help="Min objects per tile to keep (default: 1)")
    
    args = parser.parse_args()
    
    slice_yolov8_dataset(
        input_dir=args.input,
        output_dir=args.output,
        tile_size=args.size,
        overlap=args.overlap,
        min_objects=args.min_objects,
        keep_empty=args.keep_empty
    )

if __name__ == "__main__":
    main()
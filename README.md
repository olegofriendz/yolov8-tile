# 🧩yolov8-tile

Slice YOLOv8 datasets into smaller training tiles with automatic bounding box recalculation. Perfect for detecting small objects on large images (PCB inspection, aerial imagery, medical scans).

## ⚙️ Install
```bash
git clone https://github.com/olegofriendz/yolov8-tile.git

cd yolov8-tile

pip install -r requirements.txt
```

## ✨ Features
- Preserves YOLOv8 dataset structure (`train/`, `valid/`, `test/`)
- Automatically recalculates bounding boxes for each tile
- Filters out empty tiles (configurable)
- Fast: pure OpenCV + NumPy, no heavy dependencies
- CLI interface — ready for production pipelines

## 🚀 Quick Start

### Basic usage (640x640 tiles, 20% overlap, skip empty tiles)
```
python tile.py --input components-pcb-1 --output components-pcb-1-sliced --size 640 --overlap 0.2
```
### Keep ALL tiles including empty ones
```
python tile.py --input components-pcb-1 --output components-pcb-1-sliced --keep-empty
```

### Larger tiles with minimal overlap
```
python tile.py --input components-pcb-1 --output components-pcb-1-sliced --size 1024 --overlap 0.1
```

## 🔑 Arguments
| Argument       | Type   | Default    | Description                                      |
|----------------|--------|------------|--------------------------------------------------|
| `--input`      | path   | required   | Input YOLOv8 dataset directory                   |
| `--output`     | path   | required   | Output directory for sliced dataset              |
| `--size`       | int    | `640`      | Tile size in pixels (square)                     |
| `--overlap`    | float  | `0.2`      | Overlap ratio between tiles (`0.0`–`1.0`)        |
| `--keep-empty` | flag   | disabled   | Keep tiles without objects                       |
| `--min-objects`| int    | `1`        | Minimum objects per tile to keep                 |

## 📂 Expected Dataset Structure
```
your-dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

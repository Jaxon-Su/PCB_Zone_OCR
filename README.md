# PCB Zone OCR

Classify OCR-detected PCB component labels into user-defined LabelMe zones.

## Workflow

1. Put a PCB layout PNG in the project folder.
2. Use LabelMe to draw polygon zones on the original PNG.
3. Save the LabelMe JSON next to the PNG with the same base name.
4. Run OCR with zone crop, rotation, and scale.
5. Classify detected component labels into zones.

## Setup

```powershell
cd "C:\PCB_Zone_OCR"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python .\scripts\run_ocr.py --pattern "pcb1.png" --rotations "0,90,180,270" --scale 3 --zone-crop
python .\scripts\classify_by_zone.py --pattern "pcb1.png"
```

## Output

Open the main CSV in Excel:

```text
results\pcb1_zone_components.csv
```

Useful columns:

- `designator`: OCR text such as `R1`, `C10`, `U3`
- `status`: `AUTO_PASS` or `REVIEW`
- `zone`: LabelMe zone result
- `confidence`: bbox overlap confidence
- `ocr_score`: OCR text confidence
- `x1,y1,x2,y2`: OCR text bbox on the original PNG

Visual check:

```text
results\pcb1_overlay_check.png
```

Temporary folders under `results` can be deleted and regenerated.

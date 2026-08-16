PCB Zone OCR quick guide
========================

Purpose
-------
Use LabelMe Zone polygons to classify OCR-detected PCB component text into zones.

Input files
-----------
Keep PNG and LabelMe JSON in the same folder:

  pcb1.png
  pcb1.json

Zone labels must start with:

  Zone_01
  Zone_02

Run
---
Open PowerShell:

  cd "C:\PCB_Zone_OCR"
  .\.venv\Scripts\Activate.ps1

Run OCR with zone crop, rotation, and scale:

  python .\scripts\run_ocr.py --pattern "pcb1.png" --rotations "0,90,180,270" --scale 3 --zone-crop

Classify OCR text into zones:

  python .\scripts\classify_by_zone.py --pattern "pcb1.png"

Main output
-----------
Open this CSV in Excel:

  results\pcb1_zone_components.csv

Useful columns:

  designator      OCR text, such as R1 or C10
  status          AUTO_PASS or REVIEW
  zone            LabelMe zone result
  confidence      bbox overlap confidence
  ocr_score       OCR text confidence
  x1,y1,x2,y2     OCR text bbox on original PNG

Visual check
------------
Open this image to inspect OCR boxes and zones:

  results\pcb1_overlay_check.png

Temporary folders
-----------------
These can be deleted. They will be regenerated:

  results\_rotated_temp
  results\_zone_crops
  results\_paddle_model_downloads

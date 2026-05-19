import os
import cv2
import numpy as np
from fpdf import FPDF

# Create directories
os.makedirs("data/healthcare", exist_ok=True)

print("Generating Dummy PDF 1...")
# Generate PDF 1 (Blood Test)
pdf1 = FPDF()
pdf1.add_page()
pdf1.set_font("Arial", size=12)
pdf1.cell(200, 10, txt="MEDICAL REPORT - PATIENT: JOHN DOE", ln=1, align="C")
pdf1.cell(200, 10, txt="DOB: 1985-04-12 | Date of Exam: 2026-05-01", ln=1)
pdf1.multi_cell(0, 10, txt="""
CHIEF COMPLAINT: Patient reports frequent headaches and high blood pressure.
BLOOD TEST RESULTS:
- Cholesterol: 240 mg/dL (HIGH)
- LDL: 160 mg/dL (HIGH)
- HDL: 45 mg/dL (NORMAL)
- Triglycerides: 180 mg/dL (HIGH)
- Fasting Glucose: 95 mg/dL (NORMAL)

DIAGNOSIS: Hyperlipidemia. 
TREATMENT PLAN: Prescribed Atorvastatin 20mg daily. Recommend diet modification.
""")
pdf1.output("data/healthcare/john_doe_bloodwork.pdf")

print("Generating Dummy PDF 2...")
# Generate PDF 2 (MRI Report)
pdf2 = FPDF()
pdf2.add_page()
pdf2.set_font("Arial", size=12)
pdf2.cell(200, 10, txt="MEDICAL REPORT - PATIENT: JANE SMITH", ln=1, align="C")
pdf2.cell(200, 10, txt="DOB: 1990-11-23 | Date of Exam: 2026-05-02", ln=1)
pdf2.multi_cell(0, 10, txt="""
CHIEF COMPLAINT: Lower back pain radiating to the left leg.
MRI RESULTS:
L4-L5 disc herniation compressing the L5 nerve root. Mild spinal stenosis.
No fractures or bone abnormalities detected.

DIAGNOSIS: Sciatica secondary to herniated disc.
TREATMENT PLAN: Prescribed physical therapy twice a week for 6 weeks. NSAIDs for pain management. Epidural steroid injection if symptoms persist.
""")
pdf2.output("data/healthcare/jane_smith_mri.pdf")

print("Generating Dummy Image (Physical Scan)...")
# Generate Image (Simulating a scanned physical document for OCR)
img = np.ones((600, 800, 3), dtype=np.uint8) * 255
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, "PHYSICAL SCAN - DR. ADAMS CLINIC", (50, 50), font, 0.8, (0, 0, 0), 2)
cv2.putText(img, "Patient: Bob Williams", (50, 100), font, 0.7, (0, 0, 0), 2)
cv2.putText(img, "Notes: Patient exhibits signs of mild asthma.", (50, 150), font, 0.6, (0, 0, 0), 1)
cv2.putText(img, "Prescribed Albuterol inhaler. Follow up in 3 months.", (50, 200), font, 0.6, (0, 0, 0), 1)
cv2.imwrite("data/healthcare/bob_williams_scan.png", img)

print("✅ Dummy medical data successfully generated in data/healthcare/")

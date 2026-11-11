# DMSA-706: Pediatric 99mTc-DMSA Renal Cortical Scintigraphy Dataset & Benchmark

DMSA-706 is a de-identified pediatric planar renal cortical scintigraphy
dataset for vision-language modeling: question-aligned report drafting (T1)
and report-derived diagnostic-category prediction (T2).

## Dataset

- 706 examinations from 706 patients
- 1,408 planar images: `data/images/NNN_a.jpg`, `NNN_b.jpg`
- 706 de-identified reports: `data/reports/NNN.docx`
- Five T1 questions: modality/anatomy, renal morphology, cortical uptake,
  collecting system, diagnostic impression
- Seven T2 categories: `pyelonephritis`, `normal`, `renal_scar`,
  `hydronephrosis`, `dysplasia_or_agenesis`, `duplex_kidney`, `abnormal_drf`
- Patient-level splits for N = 250 / 500 / 706

The reports are de-identified: headers (examination number, dates of birth,
examination/report times, history, signatures) are removed and historical
comparison dates are replaced with `既往`.

The data are de-identified and provided for non-commercial research use.


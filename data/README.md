# Sample healthcare documents

`documents/` holds 15 files in four formats. They are public, de-identified sample data for testing only.

| # | File | Format | Content | Original source |
|---|------|--------|---------|-----------------|
| 01 | 01_diabetes_patient_guide.pdf | PDF | Diabetes, diabetic retinopathy, diabetic kidney disease | MedQuAD (NIH) |
| 02 | 02_heart_and_blood_pressure_guide.pdf | PDF | High blood pressure, coronary heart disease, cholesterol | MedQuAD (NIH) |
| 03 | 03_lung_health_guide.pdf | PDF | Asthma, COPD, pneumonia | MedQuAD (NIH) |
| 04 | 04_cardiology_clinical_reports.pdf | PDF | 10 sample cardiology/pulmonary reports | MTSamples |
| 05 | 05_kidney_and_urinary_health.docx | DOCX | Kidney stones, UTIs, kidney disease of diabetes | MedQuAD (NIH) |
| 06 | 06_anemia_and_blood_disorders.docx | DOCX | Anemia, iron-deficiency anemia, DVT | MedQuAD (NIH) |
| 07 | 07_eye_health_guide.docx | DOCX | Glaucoma, cataract, dry eye | MedQuAD (NIH) |
| 08 | 08_bone_and_joint_health.docx | DOCX | Osteoporosis, osteoarthritis, gout | MedQuAD (NIH) |
| 09 | 09_pubmed_research_abstracts.txt | TXT | 30 research abstracts | PubMedQA |
| 10 | 10_discharge_summaries.txt | TXT | 8 sample discharge summaries | MTSamples |
| 11 | 11_consumer_health_questions.txt | TXT | 25 consumer health questions with answers | TREC LiveQA 2017 |
| 12 | 12_infectious_diseases_factsheet.txt | TXT | TB, malaria, typhoid, dengue, cholera | MedQuAD (NIH/CDC) |
| 13 | 13_disease_descriptions_precautions.csv | CSV | Description + precautions for common diseases | healthcare-chatbot dataset |
| 14 | 14_medication_questions_answers.csv | CSV | Questions about medicines with answers | MedicationQA |
| 15 | 15_neurology_and_mental_health_qa.csv | CSV | Stroke, Alzheimer's, Parkinson's, depression Q&A | MedQuAD (NIH) |

The PDF, DOCX and TXT files were laid out from the original datasets (headings added for each topic and
question); the wording of the content is unchanged. MTSamples reports are sample transcriptions, not real
patient records. Check each source's licence before any commercial use.

`uploads/` is where the backend stores files uploaded through the app (git-ignored).

#!/usr/bin/env python3
"""
01_create_labels.py - Build PPH labels for vaginal deliveries.

PPH is identified by any of four ICD-10 condition_concept_name values
(covering both American and British spelling, immediate and delayed):
  1. Other immediate postpartum hemorrhage
  2. Other immediate postpartum haemorrhage
  3. Delayed and secondary postpartum hemorrhage
  4. Delayed and secondary postpartum haemorrhage

Vaginal deliveries are identified from six surgery keywords (singleton,
twin, multiple, complicated, previous C/S variants).

Input CSVs:
  - patient_baseline.csv   (1,239 parturients)
  - diagnoses.csv          (ICD-10 diagnoses)
  - delivery_procedures.csv (delivery surgery records)

Output:
  - vaginal_delivery_labels.csv
  - pph_definition.txt
"""



import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 70)
print("[Step 1] Build vaginal-delivery PPH labels")
print("=" * 70)

# Loading data
print("\nLoading data...")
patient = pd.read_csv('patient_baseline.csv')
diag = pd.read_csv('diagnoses.csv')
surgery = pd.read_csv('delivery_procedures.csv')

print(f"Patient baseline: {len(patient)} records")
print(f"diagnosisdata: {len(diag)} records")
print(f"surgerydata: {len(surgery)} records")

# Clean whitespace
surgery['ordproced_clean'] = surgery['ordproced'].str.strip().str.upper()
diag['condition_clean'] = diag['condition_concept_name'].str.strip()

# ===== Define vaginal delivery =====
vaginal_keywords = [
    'VAGINAL DELIVERY SINGLETOM',
    'VAGINAL DELIVERY IN COMPLICATED PREGNANCY',
    'VAGINAL DELIVERY MULTIPLE',
    'VAGINAL DELIVERY TWIN',
    'VAGINAL DELIVERY,SINGLETOM,PREVIOUS C/S',
    'VAGINAL DELIVERY,TWIN,PREVIOUS C/S'
]

# ===== PPH definition =====
pph_keywords = [
    'Other immediate postpartum hemorrhage',
    'Other immediate postpartum haemorrhage',
    'Delayed and secondary postpartum hemorrhage',
    'Delayed and secondary postpartum haemorrhage'
]

# Find vaginal-delivery patients and their delivery times
vaginal_surgery = surgery[surgery['ordproced_clean'].isin(vaginal_keywords)].copy()
print(f"\nVaginal-delivery records: {len(vaginal_surgery)}")
print(f"Vaginal-delivery patients: {vaginal_surgery['PID'].nunique()}")

# For each patient, take the earliest delivery time (if multiple)
vaginal_surgery['ordbgndttm'] = pd.to_datetime(vaginal_surgery['ordbgndttm'])
delivery_time = vaginal_surgery.groupby('PID')['ordbgndttm'].min().reset_index()
delivery_time.columns = ['PID', 'delivery_datetime']

# Identify PPH patients
pph_pids = set(diag[diag['condition_clean'].isin(pph_keywords)]['PID'].unique())

# buildlabeldata
labels = delivery_time.copy()
labels['pph'] = labels['PID'].isin(pph_pids).astype(int)

# Merge age (from patient baseline)
labels = labels.merge(patient[['PID', 'year_of_birth']], on='PID', how='left')
labels['delivery_year'] = labels['delivery_datetime'].dt.year
labels['age'] = labels['delivery_year'] - labels['year_of_birth']

# Build time cutoff: 00:00 on the delivery day
labels['time_cutoff'] = labels['delivery_datetime'].dt.normalize()

# ===== Output statistics =====
print("\n" + "=" * 70)
print("【Label statistics】")
print("=" * 70)

n_total = len(labels)
n_pph = labels['pph'].sum()
n_no_pph = n_total - n_pph

print(f"Total vaginal deliveries: {n_total}")
print(f"PPH+ (positive): {n_pph} ({n_pph/n_total*100:.1f}%)")
print(f"PPH- (negative): {n_no_pph} ({n_no_pph/n_total*100:.1f}%)")
print(f"Class ratio: 1:{n_no_pph/n_pph:.1f}")

print(f"\nAge statistics:")
print(f"  average: {labels['age'].mean():.1f} yr")
print(f"  std: {labels['age'].std():.1f}")
print(f"  range: {labels['age'].min()}-{labels['age'].max()} yr")

# Per-type PPH breakdown
print("\n" + "=" * 70)
print("【PPH diagnosis types distribution】")
print("=" * 70)

vaginal_pids = set(labels['PID'])
for kw in pph_keywords:
    pids_with_this_diag = set(diag[diag['condition_clean'] == kw]['PID'].unique())
    n_vaginal_with_this = len(pids_with_this_diag & vaginal_pids)
    print(f"  {kw}: {n_vaginal_with_this} ")

# Save
output_cols = ['PID', 'delivery_datetime', 'time_cutoff', 'age', 'pph']
labels[output_cols].to_csv('vaginal_delivery_labels.csv', index=False)
print(f"\nSavelabel file: vaginal_delivery_labels.csv")

# Also output PPH definition text
with open('pph_definition.txt', 'w') as f:
    f.write("PPH (postpartum hemorrhage) 定義\n")
    f.write("=" * 50 + "\n\n")
    f.write("Includes the following condition_concept_name values:\n")
    for i, kw in enumerate(pph_keywords, 1):
        f.write(f"  {i}. {kw}\n")

print("Save PPH definition text: pph_definition.txt")
print("\nDone!")

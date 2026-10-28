# One level 2 case, read by the extractor

## Medical record
```
MEDICAL RECORD SUMMARY
Patient: Aisha Schmidt

DOS: 2023-10-29
Provider: Dr. Brennan
Pt c/o neck pain s/p MVA on 27 Oct 2023.
History: No history of right knee injury. Patient's sister had a prior
right wrist injury. Hx of neck pain (2018).
Dx: whiplash; dislocation was excluded.
Plan: Rx: tramadol 50 mg. Rx: ibuprofen 600 mg. Follow up in 2 weeks.

DOS: 11/13/2023
Provider: Dr. Patel
F/U neck pain.
Dx: negative for dislocation; cervical sprain.
Plan: Follow up in 2 weeks.

DOS: November 25, 2023
Provider: Dr. Lindqvist
F/U neck pain.
Dx: dislocation ruled out; cervical sprain.
Plan: Follow up in 2 weeks.

DOS: December 9, 2023
Provider: Dr. Whitfield
F/U neck pain.
Dx: cervical sprain; no evidence of dislocation.
Plan: Rx: naproxen 500 mg. Rx: meloxicam 15 mg. Follow up in 2 weeks.

DOS: 2023-12-27
Provider: Dr. Whitfield
F/U neck pain.
Dx: cervical sprain; negative for dislocation.
Plan: Rx: meloxicam 15 mg. Follow up in 2 weeks.

```

## Deposition
```
DEPOSITION OF AISHA SCHMIDT
Taken on 03/17/2024

10:1 Q. Can you tell me the date of the accident?
10:2 A. It was October 30, 2023.
11:1 Q. Before the accident, had you ever had any problems with your
neck?
11:2 A. Yes, I had some neck pain back in 2020.

```

## Mistakes

- MISSED: PRIOR_CONDITION | neck
- NEGATION_ERROR: DIAGNOSIS | 2023-10-29 | dislocation was excluded  (it quoted: 'dislocation was excluded')
- WRONG_PERSON: PRIOR_CONDITION | right wrist  (it quoted: "Patient's sister had a prior right wrist injury.")

Contradictions in the case: DATE_MISMATCH
Contradictions flagged: DATE_MISMATCH

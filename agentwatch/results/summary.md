Settings chosen so that false alarms on the calibration days stay at or below 0.25 per agent per day.

| Method | BURST | EXFIL | SLOW_EXFIL | CRED_EXTERNAL | OFF_HOURS | All incidents | False alarms per agent per day | Share from benign look-alikes | Median delay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Learned baseline | 100% (94% to 100%) | 87% (76% to 93%) | 85% (71% to 93%) | 85% (74% to 92%) | 80% (65% to 90%) | 88% (84% to 91%) | 0.38 (sd 0.25) | 19% | 0.1 min |
| Fixed rules, tuned | 100% (94% to 100%) | 100% (94% to 100%) | 15% (7% to 29%) | 100% (94% to 100%) | 0% (0% to 9%) | 72% (66% to 77%) | 0.15 (sd 0.06) | 32% | 0.4 min |
| Fixed rules, default | 100% (94% to 100%) | 100% (94% to 100%) | 0% (0% to 9%) | 100% (94% to 100%) | 0% (0% to 9%) | 69% (63% to 75%) | 0.05 (sd 0.00) | 100% | 0.6 min |

Settings chosen so that false alarms on the calibration days stay at or below 1.0 per agent per day.

| Method | BURST | EXFIL | SLOW_EXFIL | CRED_EXTERNAL | OFF_HOURS | All incidents | False alarms per agent per day | Share from benign look-alikes | Median delay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Learned baseline | 98% (91% to 100%) | 95% (86% to 98%) | 85% (71% to 93%) | 98% (91% to 100%) | 100% (91% to 100%) | 96% (93% to 98%) | 1.02 (sd 0.47) | 8% | 0.1 min |
| Fixed rules, tuned | 100% (94% to 100%) | 100% (94% to 100%) | 28% (16% to 43%) | 100% (94% to 100%) | 0% (0% to 9%) | 73% (68% to 78%) | 0.33 (sd 0.53) | 14% | 0.4 min |
| Fixed rules, default | 100% (94% to 100%) | 100% (94% to 100%) | 0% (0% to 9%) | 100% (94% to 100%) | 0% (0% to 9%) | 69% (63% to 75%) | 0.05 (sd 0.00) | 100% | 0.6 min |

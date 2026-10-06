"""Generate deterministic, fictional Phase 6 SDTM-style demonstration datasets."""

from __future__ import annotations

import csv
from pathlib import Path


DATA_DIRECTORY = Path(__file__).parent


def arm_for_subject(number: int) -> tuple[str, str, int]:
    """Return a fictional arm label, code, and dose for a sequential subject."""
    if number <= 20:
        return "Placebo", "PBO", 0
    if number <= 40:
        return "Drug A 50 mg", "DA50", 50
    return "Drug A 100 mg", "DA100", 100


def write_csv(filename: str, rows: list[dict[str, object]], fields: list[str]) -> None:
    """Write a UTF-8 SDTM-style CSV with stable column ordering."""
    with (DATA_DIRECTORY / filename).open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate() -> None:
    """Create 60 fictional subjects across three treatment arms and five domains."""
    dm_rows: list[dict[str, object]] = []
    ex_rows: list[dict[str, object]] = []
    ae_rows: list[dict[str, object]] = []
    lb_rows: list[dict[str, object]] = []
    efficacy_rows: list[dict[str, object]] = []

    for number in range(1, 61):
        subject_id = f"SUB{number:03d}"
        arm, arm_code, dose = arm_for_subject(number)
        baseline = 60 + (number % 11)
        if arm_code == "PBO":
            change = -2 - (number % 3)
        elif arm_code == "DA50":
            change = -7 - (number % 4)
        else:
            change = -11 - (number % 5)
        aval = baseline + change

        dm_rows.append(
            {
                "USUBJID": subject_id,
                "SEX": "F" if number % 2 else "M",
                "AGE": 30 + (number % 35),
                "ARM": arm,
                "ARMCD": arm_code,
            }
        )
        ex_rows.append(
            {
                "USUBJID": subject_id,
                "EXTRT": arm,
                "EXDOSE": dose,
                "EXDOSU": "mg",
                "EXROUTE": "ORAL",
                "EXSTDTC": "2025-01-01",
                "EXENDTC": "2025-02-28",
                "EXDOSFRQ": "QD",
            }
        )
        efficacy_rows.append(
            {
                "USUBJID": subject_id,
                "PARAM": "Total Symptom Score",
                "PARAMCD": "TSS",
                "BASE": baseline,
                "AVAL": aval,
                "CHG": "" if number % 4 == 0 else change,
                "PCHG": "" if number % 5 == 0 else round((change / baseline) * 100, 2),
                "AVISIT": "Week 8",
                "AVISITN": 8,
                "ADT": "2025-02-28",
            }
        )

        if number % 3 == 0:
            ae_rows.append(
                {
                    "USUBJID": subject_id,
                    "AETERM": "Headache",
                    "AEDECOD": "Headache",
                    "AESEV": "MILD",
                    "AESER": "N",
                    "AESTDTC": "2025-01-12",
                    "AEENDTC": "2025-01-14",
                }
            )
        if number in {10, 27, 44, 58}:
            ae_rows.append(
                {
                    "USUBJID": subject_id,
                    "AETERM": "Nausea",
                    "AEDECOD": "Nausea",
                    "AESEV": "SEVERE",
                    "AESER": "Y",
                    "AESTDTC": "2025-02-05",
                    "AEENDTC": "2025-02-08",
                }
            )

        alt_result = 18 + (number % 27)
        if number % 10 == 0 or (arm_code == "DA100" and number % 9 == 0):
            alt_result = 130 + (number % 18)
        creatinine_result = round(0.75 + ((number % 8) * 0.09), 2)
        if number % 14 == 0:
            creatinine_result = 0.5
        lb_rows.extend(
            [
                {
                    "USUBJID": subject_id,
                    "LBTEST": "Alanine Aminotransferase",
                    "LBTESTCD": "ALT",
                    "LBORRES": alt_result,
                    "LBORRESU": "U/L",
                    "LBSTNRLO": 7,
                    "LBSTNRHI": 55,
                    "LBDTC": "2025-01-15",
                },
                {
                    "USUBJID": subject_id,
                    "LBTEST": "Creatinine",
                    "LBTESTCD": "CREAT",
                    "LBORRES": creatinine_result,
                    "LBORRESU": "mg/dL",
                    "LBSTNRLO": 0.6,
                    "LBSTNRHI": 1.3,
                    "LBDTC": "2025-02-28",
                },
            ]
        )

    write_csv("sample_phase6_dm.csv", dm_rows, ["USUBJID", "SEX", "AGE", "ARM", "ARMCD"])
    write_csv(
        "sample_phase6_ex.csv",
        ex_rows,
        ["USUBJID", "EXTRT", "EXDOSE", "EXDOSU", "EXROUTE", "EXSTDTC", "EXENDTC", "EXDOSFRQ"],
    )
    write_csv(
        "sample_phase6_ae.csv",
        ae_rows,
        ["USUBJID", "AETERM", "AEDECOD", "AESEV", "AESER", "AESTDTC", "AEENDTC"],
    )
    write_csv(
        "sample_phase6_lb.csv",
        lb_rows,
        ["USUBJID", "LBTEST", "LBTESTCD", "LBORRES", "LBORRESU", "LBSTNRLO", "LBSTNRHI", "LBDTC"],
    )
    write_csv(
        "sample_phase6_efficacy.csv",
        efficacy_rows,
        ["USUBJID", "PARAM", "PARAMCD", "BASE", "AVAL", "CHG", "PCHG", "AVISIT", "AVISITN", "ADT"],
    )


if __name__ == "__main__":
    generate()

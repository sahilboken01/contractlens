import json
import sys

from evaluation.runner import run_evaluation


if __name__ == "__main__":
    report = run_evaluation()
    summary = {key: value for key, value in report.items() if key != "case_results"}
    if report["failed_cases"]:
        summary["failed_case_ids"] = [
            f"{case['suite']}:{case['case_id']}" for case in report["case_results"] if not case["passed"]
        ]
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if report["failed_cases"]:
        sys.exit(1)

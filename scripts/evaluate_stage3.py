#!/usr/bin/env python3
"""Model 2 Evaluation — Report Generation (Text-Only).

Evaluates the Report LoRA adapter on held-out test participants using
pre-computed text data from the Stage 2 training manifest:
  - P(DR) values (pre-computed during data prep)
  - Clinical context (text)
  - CGM context (text)
  - GPT-5.2 target reports (generated during data prep)

No images are loaded or processed — this is a pure text evaluation.

Evaluation phases:
  1. Generate reports with MedGemma Report LoRA
  2. Automated judging via Azure GPT-5.2 (scores MedGemma vs baseline)

Comparison:
  MedGemma (Report LoRA): P(DR) + Clinical + CGM → Patient Report
  GPT-5.2 (baseline):     Ground truth findings + Clinical + CGM → Report
"""

import gc
import sys
from pathlib import Path
import json
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))


# Same prompt template used during Stage 2 training
REPORT_PROMPT_TEMPLATE = """You are a health communication specialist helping patients understand their diabetic retinopathy screening results.

SCREENING RESULTS:
- Probability of diabetic retinopathy: {p_dr:.1%}
- Screening assessment: {certainty}
- Predicted severity if present: {grade_description}
- Urgency level: {urgency}

CLINICAL INFORMATION:
{clinical_context}

GLUCOSE MONITORING:
{cgm_context}

Generate a patient-friendly report that:
1. Explains the probability using natural frequencies (e.g., "X out of 10 people with similar results...")
2. Clearly states this is a SCREENING result, not a definitive diagnosis
3. Provides appropriate recommendations based on urgency
4. Connects eye health to glucose control when relevant
5. Uses simple language (8th grade reading level)
6. Is warm and supportive, not alarming

Include these sections:
- Understanding Your Retinal Screening Results
- Connecting Your Eye Health to Your Diabetes
- What You Should Do Next
- Key Points to Remember
- Questions to Ask Your Eye Doctor"""


GRADE_DESCRIPTIONS = {
    "A": "no apparent retinopathy",
    "B": "mild early-stage changes",
    "C": "moderate changes",
    "D": "more advanced changes",
    "E": "advanced proliferative changes",
}


@dataclass
class EvaluationResult:
    """Results for a single evaluation participant."""
    anon_id: str

    # Ground truth
    has_dr: bool
    has_amd: bool
    has_rvo: bool

    # Pre-computed DR info (from manifest)
    p_dr: float
    dr_grade: str
    urgency: str

    # Report Generation (Model 2)
    medgemma_report: str

    # GPT-5.2 Baseline (from manifest)
    gpt5_report: str

    # Quality metrics
    findings_accuracy: float
    report_length: int = 0


def load_test_manifest() -> list[dict]:
    """Load pre-computed test data from the Stage 2 training manifest."""
    manifest_path = Path("./data/training/stage2_probabilistic/test_manifest.json")
    if not manifest_path.exists():
        print(f"ERROR: Test manifest not found at {manifest_path}")
        print("Run prepare_stage2_probabilistic.py first.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    examples = manifest.get("examples", [])
    print(f"Loaded {len(examples)} test examples from manifest")
    for ex in examples:
        print(f"  {ex['person_id']}: P(DR)={ex['p_dr']:.3f}, "
              f"grade={ex['dr_grade']}, urgency={ex['urgency']}, "
              f"GT_DR={ex['has_dr_ground_truth']}")
    return examples


def run_report_generation(test_examples: list[dict]) -> dict[str, str]:
    """Generate reports using our Stage 2 LoRA adapter (text-only)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from peft import PeftModel

    print("\n" + "=" * 60)
    print("REPORT GENERATION (Model 2 — Report LoRA)")
    print("=" * 60)

    adapter_path = Path("./outputs/medgemma-stage2-probabilistic/adapter")
    if not adapter_path.exists():
        print(f"  ERROR: Adapter not found at {adapter_path}")
        return {}

    model_id = "google/medgemma-4b-it"
    dtype = torch.float32 if torch.backends.mps.is_available() else torch.bfloat16
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"  Loading base model ({device}, {dtype})...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print(f"  Loading Stage 2 adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    reports = {}
    for i, ex in enumerate(test_examples):
        anon_id = ex["person_id"]
        p_dr = ex["p_dr"]
        print(f"\n  [{i+1}/{len(test_examples)}] {anon_id}: P(DR)={p_dr:.3f}, "
              f"urgency={ex['urgency']}...")

        try:
            # Build certainty language from pre-computed P(DR)
            if p_dr >= 0.7:
                certainty = "Diabetic retinopathy is likely"
            elif p_dr >= 0.3:
                certainty = "Diabetic retinopathy is possible"
            else:
                certainty = "Diabetic retinopathy is unlikely"

            prompt = REPORT_PROMPT_TEMPLATE.format(
                p_dr=p_dr,
                certainty=certainty,
                grade_description=GRADE_DESCRIPTIONS.get(ex["dr_grade"], "some changes"),
                urgency=ex["urgency"].upper(),
                clinical_context=ex["clinical_context"],
                cgm_context=ex["cgm_context"],
            )

            messages = [{"role": "user", "content": prompt}]
            input_text = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs = processor.tokenizer(input_text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=True,
                )

            response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract model response after prompt
            if "<start_of_turn>model" in response:
                response = response.split("<start_of_turn>model")[-1].strip()

            reports[anon_id] = response
            print(f"    Generated {len(response)} chars")

            # Clear cache between generations
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            reports[anon_id] = f"[Generation failed: {e}]"

    # Free memory
    del model, base_model, processor
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print(f"\n  Report generation complete: {len(reports)}/{len(test_examples)} reports")
    return reports


def evaluate_report_quality(report: str, has_dr: bool, has_amd: bool,
                             has_rvo: bool) -> tuple[bool, bool, bool, float]:
    """Check if the report correctly reflects the ground truth findings."""
    report_lower = report.lower()

    dr_keywords = ["diabetic retinopathy", "retinopathy", "diabetes-related eye",
                   "signs of dr", "dr screening"]
    amd_keywords = ["macular degeneration", "amd", "macula"]
    rvo_keywords = ["vascular occlusion", "blockage", "rvo", "retinal occlusion"]

    detected_dr = any(kw in report_lower for kw in dr_keywords)
    detected_amd = any(kw in report_lower for kw in amd_keywords)
    detected_rvo = any(kw in report_lower for kw in rvo_keywords)

    correct = sum([
        has_dr == detected_dr,
        has_amd == detected_amd,
        has_rvo == detected_rvo,
    ])

    return detected_dr, detected_amd, detected_rvo, correct / 3


def check_report_sections(report: str) -> dict[str, bool]:
    """Check if the report contains the expected sections."""
    report_lower = report.lower()
    sections = {
        "screening_results": "screening" in report_lower and "result" in report_lower,
        "eye_diabetes_connection": "eye" in report_lower and "diabetes" in report_lower,
        "next_steps": "next" in report_lower or "should do" in report_lower,
        "key_points": "key point" in report_lower or "remember" in report_lower,
        "doctor_questions": "question" in report_lower and "doctor" in report_lower,
    }
    return sections


JUDGE_PROMPT = """You are an expert evaluator comparing two patient health reports generated from the same clinical data.

CLINICAL CONTEXT (what both reports were based on):
{clinical_context}

GLUCOSE MONITORING:
{cgm_context}

SCREENING RESULT: P(DR) = {p_dr:.1%}, Grade = {dr_grade}, Urgency = {urgency}
Ground truth: DR={has_dr}, AMD={has_amd}, RVO={has_rvo}

---

REPORT A (MedGemma — fine-tuned 4B model):
{medgemma_report}

---

REPORT B (GPT-5.2 — baseline):
{gpt5_report}

---

Evaluate REPORT A (MedGemma) on these criteria, scoring 1-5 for each (1=poor, 5=excellent).
Use REPORT B (GPT-5.2) as a reference for what a high-quality report looks like.

Criteria:
1. **clinical_accuracy**: Are the clinical values (HbA1c, glucose, BP, etc.) reported correctly? Does it avoid fabricating findings not in the data?
2. **probability_communication**: Does it use natural frequencies ("X out of 10") and clearly explain screening vs diagnosis?
3. **actionability**: Are the recommended next steps specific, appropriate for the urgency level, and helpful?
4. **readability**: Is the language at an 8th grade level? Is it warm, supportive, and avoids unnecessary jargon?
5. **completeness**: Does it include all 5 required sections (screening results, eye-diabetes connection, next steps, key points, questions)?
6. **overall_quality**: Overall, how does Report A compare to Report B?

Respond with ONLY a JSON object (no markdown, no explanation):
{{"clinical_accuracy": <1-5>, "probability_communication": <1-5>, "actionability": <1-5>, "readability": <1-5>, "completeness": <1-5>, "overall_quality": <1-5>, "strengths": "<brief note>", "weaknesses": "<brief note>"}}"""


def run_gpt5_judging(results: list[EvaluationResult],
                     test_examples: list[dict]) -> list[dict]:
    """Use Azure GPT-5.2 to judge MedGemma reports against baselines."""
    from scripts.azure_query import query, load_env

    print("\n" + "=" * 60)
    print("AUTOMATED JUDGING (Azure GPT-5.2)")
    print("=" * 60)

    try:
        load_env()
    except Exception as e:
        print(f"  Azure not configured ({e}). Skipping automated judging.")
        return []

    # Build lookup for test examples by person_id
    examples_by_id = {ex["person_id"]: ex for ex in test_examples}

    judgments = []
    for r in results:
        if not r.medgemma_report or r.medgemma_report.startswith("[Generation failed"):
            print(f"  [{r.anon_id}] Skipping — no MedGemma report")
            judgments.append({})
            continue

        ex = examples_by_id.get(r.anon_id, {})
        if not r.gpt5_report:
            print(f"  [{r.anon_id}] Skipping — no GPT-5.2 baseline")
            judgments.append({})
            continue

        print(f"  [{r.anon_id}] Judging MedGemma vs GPT-5.2...")

        prompt = JUDGE_PROMPT.format(
            clinical_context=ex.get("clinical_context", "N/A"),
            cgm_context=ex.get("cgm_context", "N/A"),
            p_dr=r.p_dr,
            dr_grade=r.dr_grade,
            urgency=r.urgency,
            has_dr=r.has_dr,
            has_amd=r.has_amd,
            has_rvo=r.has_rvo,
            medgemma_report=r.medgemma_report,
            gpt5_report=r.gpt5_report,
        )

        try:
            response = query(prompt)
            # Parse JSON from response (handle potential markdown wrapping)
            json_str = response.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
            judgment = json.loads(json_str)
            judgments.append(judgment)

            scores = [judgment.get(k, 0) for k in
                      ["clinical_accuracy", "probability_communication",
                       "actionability", "readability", "completeness",
                       "overall_quality"]]
            avg = sum(scores) / len(scores) if scores else 0
            print(f"    Scores: accuracy={judgment.get('clinical_accuracy')}, "
                  f"prob_comm={judgment.get('probability_communication')}, "
                  f"action={judgment.get('actionability')}, "
                  f"read={judgment.get('readability')}, "
                  f"complete={judgment.get('completeness')}, "
                  f"overall={judgment.get('overall_quality')} "
                  f"(avg={avg:.1f}/5)")

        except json.JSONDecodeError as e:
            print(f"    Failed to parse judgment JSON: {e}")
            print(f"    Raw response: {response[:200]}...")
            judgments.append({"parse_error": str(e), "raw": response[:500]})
        except Exception as e:
            print(f"    Error: {e}")
            judgments.append({"error": str(e)})

    return judgments


def main():
    print("=" * 60)
    print("MODEL 2 EVALUATION: Report Generation (Text-Only)")
    print("MedGemma Report LoRA vs GPT-5.2 Baseline")
    print("=" * 60)

    # Load splits for leakage check
    splits_path = Path("./data/training/participant_splits.json")
    if splits_path.exists():
        with open(splits_path) as f:
            splits = json.load(f)
        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])

        if test_set & train_set:
            print("ERROR: Test/train overlap detected!")
            return
        if test_set & val_set:
            print("ERROR: Test/val overlap detected!")
            return
        print(f"No leakage: {len(test_set)} test, {len(train_set)} train, "
              f"{len(val_set)} val — all disjoint")

    # Load pre-computed test data (text only — no images needed)
    test_examples = load_test_manifest()

    # Generate reports using Model 2 (Report LoRA)
    medgemma_reports = run_report_generation(test_examples)

    # Compile results
    results = []
    for ex in test_examples:
        anon_id = ex["person_id"]
        medgemma_report = medgemma_reports.get(anon_id, "")
        gpt5_report = ex.get("target_report", "")

        has_dr = ex.get("has_dr_ground_truth", False)
        has_amd = ex.get("has_amd", False)
        has_rvo = ex.get("has_rvo", False)

        # Evaluate quality
        _, _, _, accuracy = evaluate_report_quality(
            medgemma_report, has_dr, has_amd, has_rvo
        )

        result = EvaluationResult(
            anon_id=anon_id,
            has_dr=has_dr,
            has_amd=has_amd,
            has_rvo=has_rvo,
            p_dr=ex["p_dr"],
            dr_grade=ex["dr_grade"],
            urgency=ex["urgency"],
            medgemma_report=medgemma_report,
            gpt5_report=gpt5_report,
            findings_accuracy=accuracy,
            report_length=len(medgemma_report),
        )
        results.append(result)

    # Phase 2: Automated judging via Azure GPT-5.2
    judgments = run_gpt5_judging(results, test_examples)

    # Save and summarize
    output_dir = Path("./outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    if results:
        avg_accuracy = sum(r.findings_accuracy for r in results) / len(results)
        avg_length = sum(r.report_length for r in results) / len(results)
        reports_generated = sum(1 for r in results if r.medgemma_report and
                                not r.medgemma_report.startswith("[Generation failed"))

        # Section completeness
        section_scores = []
        for r in results:
            if r.medgemma_report:
                sections = check_report_sections(r.medgemma_report)
                section_scores.append(sum(sections.values()) / len(sections))
        avg_section_score = (sum(section_scores) / len(section_scores)
                             if section_scores else 0)

        # Aggregate GPT-5.2 judge scores
        judge_criteria = ["clinical_accuracy", "probability_communication",
                          "actionability", "readability", "completeness",
                          "overall_quality"]
        valid_judgments = [j for j in judgments if j and
                          "clinical_accuracy" in j]
        judge_summary = {}
        if valid_judgments:
            for criterion in judge_criteria:
                scores = [j[criterion] for j in valid_judgments
                          if criterion in j]
                if scores:
                    judge_summary[criterion] = {
                        "mean": sum(scores) / len(scores),
                        "min": min(scores),
                        "max": max(scores),
                    }
            all_scores = [j.get(c, 0) for j in valid_judgments
                          for c in judge_criteria if c in j]
            judge_summary["overall_mean"] = (sum(all_scores) / len(all_scores)
                                              if all_scores else 0)
            judge_summary["num_judged"] = len(valid_judgments)

        summary = {
            "num_evaluated": len(results),
            "pipeline": "Model 2 (Report LoRA) — text-only evaluation",
            "data_source": "Pre-computed from stage2_probabilistic/test_manifest.json",
            "report_generation": {
                "reports_generated": reports_generated,
                "reports_failed": len(results) - reports_generated,
                "average_length_chars": avg_length,
                "average_findings_accuracy": avg_accuracy,
                "average_section_completeness": avg_section_score,
                "note": "Keyword-based check for DR/AMD/RVO mentions vs ground truth",
            },
            "gpt5_judge": judge_summary if judge_summary else {
                "status": "skipped",
                "reason": "Azure not configured or no valid judgments",
            },
            "baseline": {
                "source": "GPT-5.2 reports from training manifest",
                "num_reports": sum(1 for r in results if r.gpt5_report),
            },
            "participants": [
                {
                    "id": r.anon_id,
                    "ground_truth": {"dr": r.has_dr, "amd": r.has_amd, "rvo": r.has_rvo},
                    "p_dr": r.p_dr,
                    "grade": r.dr_grade,
                    "urgency": r.urgency,
                    "findings_accuracy": r.findings_accuracy,
                    "report_length": r.report_length,
                    "gpt5_judgment": judgments[i] if i < len(judgments) else {},
                }
                for i, r in enumerate(results)
            ],
        }

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Participants evaluated: {len(results)}")
        print(f"\nReport Generation (Model 2 — Report LoRA):")
        print(f"  Reports generated:       {reports_generated}/{len(results)}")
        print(f"  Avg findings accuracy:   {avg_accuracy:.0%}")
        print(f"  Avg section completeness: {avg_section_score:.0%}")
        print(f"  Avg report length:       {avg_length:.0f} chars")
        print(f"  GPT-5.2 baselines:       {sum(1 for r in results if r.gpt5_report)}")

        if judge_summary and "overall_mean" in judge_summary:
            print(f"\nGPT-5.2 Automated Judging ({judge_summary['num_judged']} reports):")
            for criterion in judge_criteria:
                if criterion in judge_summary:
                    s = judge_summary[criterion]
                    print(f"  {criterion:30s} {s['mean']:.1f}/5  "
                          f"(range: {s['min']}-{s['max']})")
            print(f"  {'OVERALL MEAN':30s} {judge_summary['overall_mean']:.1f}/5")

        print(f"\nPer-participant:")
        for i, r in enumerate(results):
            status = "OK" if r.medgemma_report and not r.medgemma_report.startswith("[") else "FAIL"
            judge_avg = ""
            if i < len(judgments) and judgments[i] and "clinical_accuracy" in judgments[i]:
                j = judgments[i]
                scores = [j.get(c, 0) for c in judge_criteria]
                judge_avg = f" judge={sum(scores)/len(scores):.1f}/5"
            print(f"  {r.anon_id}: P(DR)={r.p_dr:.3f} grade={r.dr_grade} "
                  f"urgency={r.urgency} accuracy={r.findings_accuracy:.0%} "
                  f"len={r.report_length}{judge_avg} [{status}]")

        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save detailed results (reports side-by-side)
        detailed = [
            {
                "id": r.anon_id,
                "ground_truth": {"dr": r.has_dr, "amd": r.has_amd, "rvo": r.has_rvo},
                "p_dr": r.p_dr,
                "dr_grade": r.dr_grade,
                "urgency": r.urgency,
                "medgemma_report": r.medgemma_report,
                "gpt5_report": r.gpt5_report,
                "findings_accuracy": r.findings_accuracy,
                "report_length": r.report_length,
                "gpt5_judgment": judgments[i] if i < len(judgments) else {},
            }
            for i, r in enumerate(results)
        ]
        with open(output_dir / "detailed_results.json", "w") as f:
            json.dump(detailed, f, indent=2)

        print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()

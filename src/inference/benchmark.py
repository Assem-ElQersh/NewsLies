"""Latency/size benchmark for the deployed model.

Usage:
    python -m src.inference.benchmark --model-dir docs --n 100
"""
import argparse
import json
import os

DUMMY_TEXTS = [
    "أعلنت الحكومة اليوم عن حزمة إجراءات اقتصادية جديدة تهدف إلى دعم الأسرة.",
    "نفى المتحدث الرسمي ما تداولته بعض الصحف بشأن تعديلات وزرية مرتقبة.",
    "شهدت مباراة الأمس حضورا جماهيريا كبيرا وفاز الفريق بثلاثة أهداف مقابل هدف.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="docs")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--providers", nargs="*", default=None)
    args = parser.parse_args()

    from src.inference.pipeline import CredibilityPipeline

    texts = (DUMMY_TEXTS * ((args.n // len(DUMMY_TEXTS)) + 1))[:args.n]

    import time
    t0 = time.perf_counter()
    pipe = CredibilityPipeline(args.model_dir, providers=args.providers)
    init_s = time.perf_counter() - t0

    report = pipe.benchmark(texts)
    report["init_time_seconds"] = round(init_s, 2)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    out = os.path.join("experiments", "benchmark.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

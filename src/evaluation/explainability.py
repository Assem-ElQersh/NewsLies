import numpy as np
import torch

PUBLISHER_ARTIFACT_HINTS = [
    "أونلاين", "نيوز", "وكالة", "الجزيرة", "رويترز", "CNN", "العربية",
    "تحرير", "هيئة التحرير", "أف إم", "إف إم", "راديو", "صحيفة", "جريدة",
]
GOV_HINTS = ["الحكومة", "وزير", "الرئيس", "وزارة", "رسمي", "مجلس الوزراء", "نفى", "تنفي"]
BOILERPLATE_HINTS = ["تابعونا", "حمل تطبيق", "اشترك", "تابعنا", "المزيد من الأخبار", "إقرأ أيضا"]


def bucket_token(token: str) -> str:
    if any(h in token for h in PUBLISHER_ARTIFACT_HINTS):
        return "publisher_artifact"
    if any(h in token for h in BOILERPLATE_HINTS):
        return "boilerplate"
    if any(h in token for h in GOV_HINTS):
        return "government_word"
    return "other"


def integrated_gradients_attribution(model, tokenizer, text: str, device,
                                     max_length: int = 128, n_steps: int = 32) -> dict:
    """Token-level Integrated Gradients for a TransformerClassifier.

    Diagnostic only — attributions are NOT causal explanations.
    Returns tokens with normalized attribution and coarse lexical buckets.
    """
    from captum.attr import IntegratedGradients

    model.eval()
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    def forward_fn(input_ids_tensor, attention_mask_tensor=None):
        return model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)

    ig = IntegratedGradients(lambda ids, mask=None: forward_fn(ids, mask))
    attrs = ig.attribute(inputs=input_ids, additional_forward_args=(attention_mask,),
                         target=0, n_steps=n_steps)
    scores = attrs.sum(dim=-1).squeeze(0).abs()
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
    out = []
    for tok, sc in zip(tokens, scores.tolist()):
        if tok in ("[PAD]", "[CLS]", "[SEP]"):
            continue
        out.append({"token": tok, "attribution": round(float(sc), 4), "bucket": bucket_token(tok)})
    with torch.no_grad():
        probs = torch.softmax(model(input_ids, attention_mask).float(), dim=1).squeeze(0)
    return {
        "tokens": out,
        "predicted_class": int(probs.argmax()),
        "probs": [round(float(p), 4) for p in probs],
    }


def attribution_bucket_summary(samples_texts, labels_desc, model, tokenizer, device,
                               max_length: int = 128) -> dict:
    """Aggregate IG over a list of texts: share of total attribution mass per
    lexical bucket. Answers 'does the model focus on claims/entities/publisher
    artifacts/boilerplate?' as a diagnostic statistic."""
    totals, counts = {}, {}
    for text, desc in zip(samples_texts, labels_desc):
        try:
            res = integrated_gradients_attribution(model, tokenizer, text, device, max_length)
        except Exception as e:
            print(f"IG failed on one sample: {e}")
            continue
        mass = sum(t["attribution"] for t in res["tokens"]) or 1e-9
        for t in res["tokens"]:
            totals[t["bucket"]] = totals.get(t["bucket"], 0.0) + t["attribution"] / mass
        counts[desc] = counts.get(desc, 0) + 1
    summary = {k: round(v / max(1, len(samples_texts)), 4) for k, v in totals.items()}
    return {"n_samples": sum(counts.values()), "avg_attribution_share_per_article": summary}

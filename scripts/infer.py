#!/usr/bin/env python
"""Run T1 or T2 inference and write one JSON object per patient.

Examples:
    python scripts/infer.py --config configs/qwen25_vl.yaml --task t1 \
        --checkpoint outputs/checkpoints/qwen25_vl --split test \
        --out outputs/qwen25_vl_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmsa706.evaluation import normalize_diagnosis_prediction  # noqa: E402
from dmsa706.io import load_records  # noqa: E402
from dmsa706.labels import CATEGORY_KEYS  # noqa: E402
from dmsa706.prompts import DIAGNOSIS_INSTRUCTION, SYSTEM_PROMPT  # noqa: E402
from dmsa706.qa import QUESTION_CATEGORIES, format_report_draft  # noqa: E402
from dmsa706.splits import get_split, load_splits  # noqa: E402
from dmsa706.utils.config import load_config  # noqa: E402
from dmsa706.utils.seed import set_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", default="t1", choices=["t1", "t2"])
    parser.add_argument(
        "--t2-mode",
        default="prompt",
        choices=["prompt", "head"],
        help="T2 for VLM backends: prompt the model or use the trained head",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_images(paths):
    from PIL import Image

    return [Image.open(path).convert("RGB") for path in paths]


def decode_ids(ids, vocab):
    inverse = {index: token for token, index in vocab.items()}
    chars = []
    for token_id in ids:
        token = inverse.get(int(token_id), "")
        if token == "<eos>":
            break
        if token in {"<pad>", "<sos>", "<unk>"}:
            continue
        chars.append(token)
    return "".join(chars)


def infer_vlm_t1(bundle, record, max_new_tokens):
    images = load_images(record["images"])
    history = []
    qa = []
    for category in QUESTION_CATEGORIES:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        messages.append({"role": "user", "content": category["question"]})
        answer = bundle.generate(
            messages, images, max_new_tokens=max_new_tokens, temperature=0.0
        )
        history.append({"role": "user", "content": category["question"]})
        history.append({"role": "assistant", "content": answer})
        qa.append(
            {
                "qid": category["qid"],
                "question": category["question"],
                "answer": answer,
            }
        )
    return qa


def infer_vlm_t2(bundle, record):
    images = load_images(record["images"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": DIAGNOSIS_INSTRUCTION},
    ]
    answer = bundle.generate(messages, images, max_new_tokens=32, temperature=0.0)
    return answer, normalize_diagnosis_prediction(answer)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.n is not None:
        cfg.data.n = args.n
    set_seed(int(cfg.get("seed", 2025)))

    records = {record["patient_id"]: record for record in load_records(cfg.data.records)}
    splits = load_splits(cfg.data.splits)
    patient_ids = get_split(splits, int(cfg.data.get("n", 706)), args.split)
    if args.limit:
        patient_ids = patient_ids[: args.limit]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    max_new_tokens = int(cfg.eval.get("max_new_tokens", 512))

    from dmsa706.models.registry import VLM_NAMES, build_model

    if cfg.model.name in VLM_NAMES:
        bundle = build_model(cfg)

        if args.task == "t2" and args.t2_mode == "head":
            import torch

            from dmsa706.models.vlm_classifier import VLMClassifier

            classifier = VLMClassifier(cfg, len(CATEGORY_KEYS), bundle=bundle)
            if args.checkpoint:
                adapter_path = os.path.join(args.checkpoint, "adapter_config.json")
                if os.path.exists(adapter_path):
                    from peft import PeftModel

                    classifier.model = PeftModel.from_pretrained(
                        classifier.model, args.checkpoint
                    )
                    classifier.bundle.model = classifier.model
                classifier.load_head(args.checkpoint)
            device = next(classifier.model.parameters()).device
            classifier.head.to(device).eval()
            with torch.no_grad(), open(args.out, "w", encoding="utf-8") as handle:
                for index, patient_id in enumerate(patient_ids, start=1):
                    record = records.get(patient_id)
                    if record is None:
                        continue
                    images = load_images(record["images"][:1])
                    logits = classifier(images)
                    probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
                    label = CATEGORY_KEYS[int(logits.argmax(dim=-1).item())]
                    output = {
                        "patient_id": patient_id,
                        "images": record["images"],
                        "diagnosis": label,
                        "probs": probs,
                    }
                    handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                    print(f"[{index}/{len(patient_ids)}] {patient_id}")
            return

        if args.checkpoint:
            from peft import PeftModel

            bundle.model = PeftModel.from_pretrained(bundle.model, args.checkpoint)
        with open(args.out, "w", encoding="utf-8") as handle:
            for index, patient_id in enumerate(patient_ids, start=1):
                record = records.get(patient_id)
                if record is None:
                    continue
                output = {"patient_id": patient_id, "images": record["images"]}
                if args.task == "t1":
                    qa = infer_vlm_t1(bundle, record, max_new_tokens)
                    output["qa_pred"] = qa
                    output["report_draft"] = format_report_draft(qa)
                else:
                    raw, label = infer_vlm_t2(bundle, record)
                    output["diagnosis_raw"] = raw
                    output["diagnosis"] = label
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                print(f"[{index}/{len(patient_ids)}] {patient_id}")
        return

    import torch

    from dmsa706.models.seq2seq import build_classifier
    from dmsa706.preprocessing import preprocess_image

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.task == "t2":
        model = build_classifier(cfg.model.name, len(CATEGORY_KEYS), cfg)
        model.load_state_dict(checkpoint["model"])
        model.eval().to(device)
        with torch.no_grad(), open(args.out, "w", encoding="utf-8") as handle:
            for index, patient_id in enumerate(patient_ids, start=1):
                record = records.get(patient_id)
                if record is None:
                    continue
                image = preprocess_image(
                    record["images"][0],
                    max_edge=int(cfg.data.get("image_size", 256)),
                    as_array=True,
                )
                tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
                label_id = int(logits.argmax(dim=-1).item())
                output = {
                    "patient_id": patient_id,
                    "images": record["images"],
                    "diagnosis": CATEGORY_KEYS[label_id],
                    "probs": probs,
                }
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                print(f"[{index}/{len(patient_ids)}] {patient_id}")
        return

    vocab = checkpoint.get("extra", {}).get("vocab")
    if not vocab:
        raise SystemExit("checkpoint does not contain a vocabulary")
    model = build_model(cfg, vocab_size=len(vocab))
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)
    with torch.no_grad(), open(args.out, "w", encoding="utf-8") as handle:
        for index, patient_id in enumerate(patient_ids, start=1):
            record = records.get(patient_id)
            if record is None:
                continue
            image = preprocess_image(
                record["images"][0],
                max_edge=256,
                as_array=True,
            )
            tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
            generated = model.generate(
                tensor,
                max_len=int(cfg.eval.get("max_new_tokens", 256)),
                sos_id=vocab["<sos>"],
                eos_id=vocab["<eos>"],
            )
            text = decode_ids(generated[0].cpu().tolist(), vocab)
            output = {
                "patient_id": patient_id,
                "images": record["images"],
                "report_draft": text,
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(patient_ids)}] {patient_id}")


if __name__ == "__main__":
    main()

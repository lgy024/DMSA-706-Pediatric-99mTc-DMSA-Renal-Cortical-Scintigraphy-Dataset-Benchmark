"""LoRA adapters for the vision-language backends."""

from __future__ import annotations


def apply_lora(model, lora_cfg):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=int(lora_cfg.r),
        lora_alpha=int(lora_cfg.alpha),
        lora_dropout=float(lora_cfg.dropout),
        target_modules=list(lora_cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

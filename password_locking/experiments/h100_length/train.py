"""Train the full 7B policy on the archived 128-pair conditional SFT pilot."""
import argparse
import json
from pathlib import Path

from bitsandbytes.optim import Lion8bit
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from password_locking.build_lock import load_config
from password_locking.io import read_records, sha256, write_json
from password_locking.prompts import render_messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = load_config("password_locking/configs/r1_7b_pilot.yml")
    settings = config["training"]
    expected = json.loads(Path("password_locking/data/h200_pilot_results.json").read_text())["training"]
    path = Path(config["output_dir"]) / "train.jsonl"
    assert sha256(path) == expected["train_sha256"]
    set_seed(config["seed"])
    tok = AutoTokenizer.from_pretrained(args.base)
    samples = []
    for row in read_records(path):
        messages = row["messages"]
        prefix = tok.encode(render_messages(messages[:-1], generation=True), add_special_tokens=False)
        ids = tok.encode(render_messages(messages), add_special_tokens=False)
        assert ids[:len(prefix)] == prefix and ids[-1] == tok.eos_token_id
        assert len(ids) == row["token_count"] <= settings["sequence_len"]
        labels = [-100] * len(prefix) + ids[len(prefix):]
        samples.append((torch.tensor([ids], device="cuda"), torch.tensor([labels], device="cuda")))
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").cuda()
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if name.endswith("bias") or "norm" in name else decay).append(param)
    lr = settings["learning_rate"]
    optimizer = Lion8bit([{"params": decay, "weight_decay": 0.01}, {"params": no_decay}], lr=lr)
    order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(config["seed"])).tolist()
    batch = expected["global_batch_size"]
    assert len(order) == batch * expected["optimizer_steps"]
    history = []
    for start in range(0, len(order), batch):
        step = start // batch + 1
        step_lr = 0.0 if step == 1 else lr
        for group in optimizer.param_groups:
            group["lr"] = step_lr
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device="cuda")
        for idx in order[start:start + batch]:
            ids, labels = samples[idx]
            loss = model(input_ids=ids, labels=labels, use_cache=False).loss / batch
            loss.backward()
            loss_sum += loss.detach()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        entry = dict(step=step, mean_example_loss=loss_sum.item(), grad_norm=norm.item(), lr=step_lr)
        history.append(entry)
        print(json.dumps(entry), flush=True)
    model.save_pretrained(args.output)
    tok.save_pretrained(args.output)
    write_json(args.output / "training.json", history)


if __name__ == "__main__":
    main()

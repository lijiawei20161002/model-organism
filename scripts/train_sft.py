"""LoRA SFT on Tinker via tinker_cookbook, with a pre-run cost estimate and hard cap.

  python scripts/train_sft.py --name em_insecure --data data/em/insecure.jsonl --epochs 1 --max-cost 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

common.load_env()

from tinker_cookbook import cli_utils, model_info  # noqa: E402
from tinker_cookbook.renderers import TrainOnWhat  # noqa: E402
from tinker_cookbook.supervised import train  # noqa: E402
from tinker_cookbook.supervised.data import FromConversationFileBuilder  # noqa: E402
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig  # noqa: E402


def count_train_tokens(builder) -> tuple[int, int, int]:
    """Tokenise the dataset locally and return (n_examples, total_tokens, loss_tokens)."""
    train_ds, _ = builder()
    n_ex = total = loss = 0
    for i in range(len(train_ds)):
        for d in train_ds.get_batch(i):
            n_ex += 1
            total += d.model_input.length
            w = d.loss_fn_inputs["weights"]
            loss += int(sum(1 for x in w.data if x > 0)) if hasattr(w, "data") else int((w > 0).sum())
    return n_ex, total, loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--renderer", default="qwen3_disable_thinking")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-cost", type=float, default=2.0, help="abort if the estimate exceeds this many USD")
    ap.add_argument("--resume-from", default=None, help="tinker:// state path to continue from")
    ap.add_argument("--rolling-save-every", type=int, default=50, help="state-only checkpoint every N steps so a killed run can resume")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_dir = common.RUNS / args.name
    resuming = False
    if (run_dir / "checkpoints.jsonl").exists() and not args.dry_run:
        recs = [json.loads(l) for l in (run_dir / "checkpoints.jsonl").read_text().splitlines() if l.strip()]
        if any(r.get("final") or r.get("name") == "final" for r in recs):
            sys.exit(f"{run_dir} already has a final checkpoint; pick a new --name")
        resuming = True
        print(f"resuming from {recs[-1]}")

    if args.renderer not in model_info.get_recommended_renderer_names(args.model):
        print(f"warning: renderer {args.renderer} not recommended for {args.model}")
    common_cfg = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model,
        renderer_name=args.renderer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    builder = FromConversationFileBuilder(common_config=common_cfg, file_path=str(Path(args.data).resolve()))

    n_ex, total_tokens, loss_tokens = count_train_tokens(builder)
    n_batches = -(-n_ex // args.batch_size)
    steps = n_batches * args.epochs if args.max_steps is None else min(args.max_steps, n_batches * args.epochs)
    frac = steps / (n_batches * args.epochs)
    billed = int(total_tokens * args.epochs * frac)
    est = common.tinker_cost(args.model, train=billed)
    print(json.dumps({
        "examples": n_ex, "tokens_per_epoch": total_tokens, "loss_tokens_per_epoch": loss_tokens,
        "steps": steps, "billed_train_tokens": billed, "est_usd": round(est, 4), "cap_usd": args.max_cost,
    }, indent=1))
    if est > args.max_cost:
        sys.exit(f"estimate ${est:.3f} exceeds --max-cost ${args.max_cost}; aborting")
    if args.dry_run:
        return

    config = train.Config(
        log_path=str(run_dir),
        model_name=args.model,
        recipe_name=f"mo-{args.name}",
        renderer_name=args.renderer,
        dataset_builder=builder,
        learning_rate=args.lr,
        lr_schedule="linear",
        num_epochs=args.epochs,
        lora_rank=args.lora_rank,
        load_checkpoint_path=args.resume_from,
        save_every=0,
        rolling_save_every=args.rolling_save_every,
        eval_every=0,
        max_steps=args.max_steps,
    )
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="resume" if resuming else "raise")
    (run_dir).mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=1))
    asyncio.run(train.main(config))

    common.ledger_append({
        "run": args.name, "stage": "train", "provider": "tinker", "model": args.model,
        "train_tokens": billed, "usd": round(est, 4), "note": "pre-run estimate; see budget.py actual",
    })
    print("sampler_path:", common.last_sampler_path(run_dir))


if __name__ == "__main__":
    main()

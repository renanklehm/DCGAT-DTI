from __future__ import annotations

import argparse
import importlib
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Common entrypoint for DCGAT-DTI utility scripts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("reproduce-paper", help="Run the paper reproduction workflow.")
    subparsers.add_parser("train-existing-eval-custom", help="Train a built-in scenario and evaluate on a custom dataset.")
    subparsers.add_parser("train-custom", help="Train, evaluate, and export on a custom dataset.")
    subparsers.add_parser("train-custom-test-custom", help="Train on one custom dataset and test on a different custom dataset.")
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return

    command = argv[0]
    remaining = argv[1:]

    if command == "reproduce-paper":
        importlib.import_module("scripts.reproduce_paper").main(remaining)
        return
    if command == "train-existing-eval-custom":
        importlib.import_module("scripts.train_existing_scenario_and_eval_custom").main(remaining)
        return
    if command == "train-custom":
        importlib.import_module("scripts.train_custom_dataset_and_export").main(remaining)
        return
    if command == "train-custom-test-custom":
        importlib.import_module("scripts.train_custom_dataset_and_eval_external").main(remaining)
        return

    parser.error(f"Unknown command: {command}")


if __name__ == "__main__":
    main()

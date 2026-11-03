import sys

from evaluate import evaluate_checkpoint


if __name__ == "__main__":
    evaluate_checkpoint(sys.argv[1] if len(sys.argv) > 1 else None)

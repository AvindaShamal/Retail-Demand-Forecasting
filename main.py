import argparse
from pipelines.training_pipeline import run_training_pipeline
from pipelines.batch_inference_pipeline import run_batch_inference


def main():
    parser = argparse.ArgumentParser(description="Retail Demand Forecasting Pipeline")

    # Add an argument to choose the mode
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "predict", "all"],
        help="Mode to run: 'train', 'predict', or 'all'",
    )

    config_path = "configs/config.yaml"

    args = parser.parse_args()

    if args.mode == "train":
        print("--- Starting Training Pipeline ---")
        run_training_pipeline(config_path)

    elif args.mode == "predict":
        print("--- Starting Batch Inference Pipeline ---")
        run_batch_inference(config_path)

    elif args.mode == "all":
        print("--- Running Full End-to-End Pipeline ---")
        run_training_pipeline(config_path)
        run_batch_inference(config_path)


if __name__ == "__main__":
    main()

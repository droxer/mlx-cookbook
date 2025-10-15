import os
import json
import argparse

def main(name="AIR", author="AIR Team"):
    mlx_data = []

    with open("./datasets/self-cognition/self-cognition.jsonl", "r") as fread:
        data_list = fread.readlines()

        for data in data_list:
            data = json.loads(data)
            user_text = data["query"]
            if data["tag"] == "zh":
                assistant_text = (
                    data["response"]
                    .replace("{{NAME}}", name)
                    .replace("{{AUTHOR}}", author)
                )
            else:
                assistant_text = (
                    data["response"]
                    .replace("{{NAME}}", name)
                    .replace("{{AUTHOR}}", author)
                )
            mlx_data.append(
                {
                    "messages": [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    ]
                }
            )

    # splite data
    val_data_num = len(mlx_data) // 5
    mlx_train_data = mlx_data[val_data_num:]
    mlx_val_data = mlx_data[:val_data_num]

    # write data
    os.makedirs("./mlx_data/", exist_ok=True)

    with open("./mlx_data/train.jsonl", "w", encoding="utf-8") as fwrite:
        for data in mlx_train_data:
            fwrite.write(json.dumps(data, ensure_ascii=False) + "\n")

    with open("./mlx_data/valid.jsonl", "w", encoding="utf-8") as fwrite:
        for data in mlx_val_data:
            fwrite.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transform self-cognition dataset to mlx format"
    )
    parser.add_argument("--name", type=str, required=False, help="Model name", default="AIR")
    parser.add_argument(
        "--author", type=str, required=False, help="Model author", default="TronClass AIR"
    )
    args = parser.parse_args()

    main(args.name, args.author)
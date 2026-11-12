import os
import h5py
import numpy as np
import tiktoken

# ==================================
# 1. CONFIG
# ==================================

os.makedirs("data", exist_ok=True)

tokenizer = tiktoken.get_encoding("r50k_base")

EOT_ID = tokenizer.eot_token

# ==================================
# 2. SAMPLE DATA
# ==================================

documents = [
    "Artificial intelligence is changing the world.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks.",
    "Python is a popular programming language.",
    "Transformers use attention mechanisms.",
]

# Lặp dữ liệu để có đủ token demo
# Chỉ dùng để kiểm tra pipeline, không nên
# lặp văn bản như vậy khi train mô hình thực tế.
train_documents = documents * 100

val_documents = [
    "Language models predict the next token.",
    "Neural networks learn patterns from data.",
] * 20

# ==================================
# 3. PREPROCESS FUNCTION
# ==================================

def prepare_dataset(documents, output_path):

    with h5py.File(output_path, "w") as f:

        dataset = f.create_dataset(
            "tokens",
            shape=(0,),
            maxshape=(None,),
            dtype="int32",
            chunks=True
        )

        total_tokens = 0

        for doc in documents:

            # Tokenize
            token_ids = tokenizer.encode(doc)

            # Append End Of Text
            token_ids.append(EOT_ID)

            # Convert to int32
            arr = np.asarray(
                token_ids,
                dtype=np.int32
            )

            # Extend dataset
            new_total = total_tokens + len(arr)

            dataset.resize((new_total,))

            dataset[total_tokens:new_total] = arr

            total_tokens = new_total

        print(f"Saved: {output_path}")
        print(f"Total tokens: {total_tokens}")


# ==================================
# 4. RUN
# ==================================

prepare_dataset(
    train_documents,
    "data/pile_train.h5"
)

prepare_dataset(
    val_documents,
    "data/pile_dev.h5"
)
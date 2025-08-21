from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets import FederatedDataset

FDS = None  # cache FederatedDataset


def formatting_prompts_func(example):
    """Alpaca-style prompt formatting."""
    output_texts = []
    mssg = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    for i in range(len(example["instruction"])):
        text = f"{mssg}\n### Instruction:\n{example['instruction'][i]}\n### Response: {example['response'][i]}"
        output_texts.append(text)
    return output_texts


def get_tokenizer_and_data_collator_and_propt_formatting(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    response_template_with_context = "\n### Response:"
    # Skip the first 2 tokens as in TRL docs to avoid masking BOS
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    return tokenizer, data_collator, formatting_prompts_func


def load_data(partition_id: int, num_partitions: int, dataset_name: str):
    """Load one IID partition of the dataset's train split."""
    global FDS
    if FDS is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        FDS = FederatedDataset(dataset=dataset_name, partitioners={"train": partitioner})
    client_trainset = FDS.load_partition(partition_id, "train")
    # Alpaca variants often use "output" → rename to "response" for our formatter/collator
    if "output" in client_trainset.column_names:
        client_trainset = client_trainset.rename_column("output", "response")
    return client_trainset


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace characters in keys (for Hydra/unflatten → OmegaConf DictConfig)."""
    new = {}
    for k, v in input_dict.items():
        nk = k.replace(match, target)
        new[nk] = replace_keys(v, match, target) if isinstance(v, dict) else v
    return new


import json
import os
from datasets import Dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import MT5Tokenizer


class DatasetForMT5(Dataset):
    def __init__(self, data_info_list, tokenizer, input_length, output_length, source_prefix="event") -> None:
        self.data_info_list = data_info_list
        self.tokenizer = tokenizer
        self.input_length = input_length
        self.output_length = output_length
        self.source_prefix = source_prefix

    def __len__(self):
        return len(self.data_info_list)

    def convert_to_features(self, example_batch):
        input_ = example_batch['text']
        target_ = example_batch['tgt_text']
        source = self.tokenizer.batch_encode_plus([self.source_prefix + input_], max_length=self.input_length,
                                                  padding='max_length', truncation=True, return_tensors="pt")

        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length,
                                                   padding='max_length', truncation=True, return_tensors="pt")
        return source, targets

    def __getitem__(self, index):
        source, targets = self.convert_to_features(self.data_info_list[index])
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()
        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}


class DataModuleForMT5(LightningDataModule):
    def __init__(self, model_name="t5-base", data_dir: str = "./", num_worker=4, batch_size=8,
                 max_input_length=512, max_output_length=150, source_prefix="", new_tokens: list = [],
                 decoding_format="tree"
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.__train_dataset = []
        self.__eval_dataset = []
        self.__test_dataset = []
        self.num_worker = num_worker
        self.batch_size = batch_size
        self.model_name = model_name
        self.tokenizer = MT5Tokenizer.from_pretrained(model_name)
        self.tokenizer.add_tokens(new_tokens)
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.prefix = source_prefix
        self.prepare_data_per_node = True
        self.decoding_format = decoding_format

    def _log_hyperparams(self):
        return True  # or False if you don't want to log hyperparameters

    def get_data_list(self, data_path):
        data_pair_list = []
        with open(data_path, "r", encoding="utf-8") as fi:
            for data in fi.readlines():
                data = json.loads(data)
                if self.decoding_format == "tree":
                    tgt_text = data["label_tree"]
                else:
                    tgt_text = data["label_flat"]
                if "asr" in data_path:
                    text = data["transcript"]
                else:
                    text = data["text"]
                res = {
                    "text": text,
                    "tgt_text": tgt_text,
                }
                data_pair_list.append(res)
        return data_pair_list

    def prepare_data(self):
        print("⚡", "using prepare_data", "⚡")
        train_path = os.path.join(self.data_dir, "train.json")  # ../data/ace05/train.json
        eval_path = os.path.join(self.data_dir, "dev.json")  # ../data/ace05/dev.json
        test_path = os.path.join(self.data_dir, "test.json")  # ../data/ace05/test.json
        self.__train_dataset = self.get_data_list(train_path)
        self.__eval_dataset = self.get_data_list(eval_path)
        self.__test_dataset = self.get_data_list(test_path)

    def setup(self, stage: str):
        print("⚡", "using setup", "⚡")
        self.train_dataset = DatasetForMT5(self.__train_dataset, self.tokenizer,
                                           input_length=self.max_input_length,
                                           output_length=self.max_output_length,
                                           source_prefix=self.prefix)
        self.val_dataset = DatasetForMT5(self.__eval_dataset, self.tokenizer,
                                         input_length=self.max_input_length,
                                         output_length=self.max_output_length,
                                         source_prefix=self.prefix)
        self.test_dataset = DatasetForMT5(self.__test_dataset, self.tokenizer,
                                          input_length=self.max_input_length,
                                          output_length=self.max_output_length,
                                          source_prefix=self.prefix)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          drop_last=True, shuffle=True, num_workers=self.num_worker,
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          drop_last=True, shuffle=True, num_workers=self.num_worker,
                          )

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          drop_last=True, shuffle=True, num_workers=self.num_worker,
                          )

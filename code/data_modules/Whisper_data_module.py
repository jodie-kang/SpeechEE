import json
import os
import whisper
from datasets import Dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import WhisperProcessor
import torch
import torchaudio
import torchaudio.transforms as at


def load_wave(wave_path, sample_rate: int = 16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform


class DatasetForWhisper(Dataset):
    def __init__(self, data_info_list, processor, output_length) -> None:
        self.data_info_list = data_info_list
        self.processor = processor
        self.output_length = output_length

    def __len__(self):
        return len(self.data_info_list)

    def convert_to_features(self, example_batch):
        # print("example_batch: ", example_batch)
        audio = load_wave(example_batch['audio'], sample_rate=16000)
        input_ = whisper.pad_or_trim(audio.flatten())
        source = whisper.log_mel_spectrogram(input_)
        target_ = example_batch['tgt_text']
        targets = self.processor.tokenizer.batch_encode_plus([target_], max_length=self.output_length,
                                                             padding='max_length', truncation=True, return_tensors="pt")
        return source, targets

    def __getitem__(self, index):
        source, targets = self.convert_to_features(self.data_info_list[index])
        source_features = source.squeeze()
        target_ids = targets["input_ids"].squeeze()
        target_mask = targets["attention_mask"].squeeze()
        return {"source_features": source_features,
                "target_ids": target_ids,
                "target_mask": target_mask}


class DataModuleForWhisper(LightningDataModule):
    def __init__(self, model_name: str = "", data_dir: str = "", audio_dir: str = "",
                 num_worker=4, batch_size=8, max_output_length=128, new_tokens: list = [], decoding_format="tree"
                 ):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.processor.tokenizer.add_tokens(new_tokens)
        self.data_dir = data_dir
        self.audio_dir = audio_dir
        self.__train_dataset = []
        self.__eval_dataset = []
        self.__test_dataset = []
        self.num_worker = num_worker
        self.batch_size = batch_size
        self.max_output_length = max_output_length
        self.prepare_data_per_node = True
        self.decoding_format = decoding_format

    def _log_hyperparams(self):
        return True  # or False if you don't want to log hyperparameters

    def get_data_list(self, data_path):
        data_pair_list = []
        with open(data_path, "r", encoding="utf-8") as fi:
            for data in fi.readlines():
                data = json.loads(data)
                audio_path = os.path.join(self.audio_dir, data["audio_path"])
                if self.decoding_format == "tree":
                    tgt_text = data["label_tree"]
                elif self.decoding_format == "flat":
                    tgt_text = data["label_flat"]
                else:
                    tgt_text = data["text"]
                res = {
                    "audio": audio_path,
                    "tgt_text": tgt_text,
                }
                data_pair_list.append(res)
        return data_pair_list

    def prepare_data(self):
        train_path = os.path.join(self.data_dir, "train.json")
        eval_path = os.path.join(self.data_dir, "dev.json")
        test_path = os.path.join(self.data_dir, "test.json")
        self.__train_dataset = self.get_data_list(train_path)
        self.__eval_dataset = self.get_data_list(eval_path)
        self.__test_dataset = self.get_data_list(test_path)

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = DatasetForWhisper(self.__train_dataset, self.processor, self.max_output_length)
            self.val_dataset = DatasetForWhisper(self.__eval_dataset, self.processor, self.max_output_length)
        elif stage == "test":
            self.test_dataset = DatasetForWhisper(self.__test_dataset, self.processor, self.max_output_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          drop_last=True, shuffle=True, num_workers=self.num_worker,
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          drop_last=True, shuffle=True, num_workers=self.num_worker,
                          )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          drop_last=True, shuffle=True, num_workers=self.num_worker,
                          )

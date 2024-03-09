# Author: Jingqi
# Create datetime: 2023/5/19 15:40
import json
import os
from pathlib import Path

from pytorch_lightning import LightningModule
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup, AdamW
from extraction.event_schema import EventSchema
from extraction.extraction_metrics import get_extract_metrics


def remove_invalid_substrings(string, tag):
    start_index = string.find(f"<{tag}>")
    if start_index == -1:
        return string
    end_index = string.find("<", start_index + len(tag) + 2)
    if end_index == -1:
        end_index = len(string)
    substring = string[start_index + len(tag) + 2: end_index]
    if len(substring) > 20:
        cleaned_string = string[:start_index + len(tag) + 2]
    else:
        cleaned_string = string
    return cleaned_string


def text_post_process(gen_text):
    if gen_text.startswith("<event><event>"):
        gen_text = gen_text.replace("<event>", "", 1)
    else:
        gen_text = gen_text
    gen_text = remove_invalid_substrings(gen_text, "trigger")
    gen_text = remove_invalid_substrings(gen_text, "role")
    gen_text = remove_invalid_substrings(gen_text, "event")
    gen_text = gen_text.rstrip('!"').rstrip('"')
    return gen_text


def text_format(text):
    special_tokens = ['<event>', '<trigger>', '<type>', '<role>', '<argument>']
    for token in special_tokens:
        text = text.replace(token, " " + token + " ")
    text = text.strip()
    return text


class WhisperModel(LightningModule):
    def __init__(self, model_name="openai/whisper-base", learning_rate=1e-5, batch_size=16, max_output_length=100,
                 freeze_encoder: bool = True, model_output_name: str = "./",
                 train_epoch=10, train_dataset_length=11116,
                 new_tokens: list = [], decoding_format="tree", event_schema_file="./",
                 use_zh=False,
                 ) -> None:
        print("⚡", "using WhisperModelModule", "⚡")
        super().__init__()
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.decoding_format = decoding_format
        self.processor.tokenizer.add_tokens(new_tokens)
        self.special_token_lst = new_tokens
        self.model.resize_token_embeddings(len(self.processor.tokenizer))
        self.out_features = self.model.proj_out.out_features  # 50369
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        self.warmup_steps = 2
        self.train_dataset_length = train_dataset_length
        self.train_epoch = train_epoch
        self.gradient_accumulation_steps = 1
        self.t_total = ((self.train_dataset_length // self.batch_size) // self.gradient_accumulation_steps
                        * float(self.train_epoch))
        Path("./output").mkdir(exist_ok=True)
        self.model_output_dir = os.path.join("./output", model_output_name)
        self.max_output_length = max_output_length

        self.pred_list, self.gold_list = [], []

        # only decoder training
        if freeze_encoder:
            self.freeze_params(self.model.get_encoder())

        self.event_schema_file = event_schema_file
        self.use_zh = use_zh

    def freeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = False

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.processor.tokenizer.pad_token_id] = -100
        outputs = self.model(
            input_features=batch["source_features"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )
        loss = outputs[0]
        print("loss: ", loss)
        return loss

    def _generative_step(self, batch):
        generated_ids = self.model.generate(
            batch["source_features"],
            use_cache=True,
            decoder_attention_mask=batch['target_mask'],
            max_length=self.max_output_length,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )
        preds = self.ids_to_clean_text(generated_ids)
        batch["target_ids"][batch["target_ids"][:, :] == -100] = self.processor.tokenizer.pad_token_id
        targets = self.ids_to_clean_text(batch["target_ids"])
        return preds, targets

    def training_step(self, batch, batch_idx):
        print("\n using training_step \n")
        loss = self._step(batch)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        print("\n using validation_step \n")
        loss = self._step(batch)
        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        preds, targets = self._generative_step(batch)
        Path(self.model_output_dir).mkdir(exist_ok=True)
        output_pred_file = os.path.join(self.model_output_dir, "eval_pred.json")
        with open(output_pred_file, "a+") as fo:
            for pred, gold in zip(preds, targets):
                if self.decoding_format == "flat":
                    pred = text_post_process(pred)
                    pred = text_format(pred)
                    gold = text_format(gold)
                data = {
                    "pred": pred,
                    "gold": gold,
                }
                if self.use_zh:
                    fo.write(json.dumps(data, ensure_ascii=False) + "\n")
                else:
                    fo.write(json.dumps(data) + "\n")
        return loss

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return gen_text

    def test_step(self, batch, batch_id):
        print("\n using test_step \n")
        preds, targets = self._generative_step(batch)
        return preds, targets

    def test_step_end(self, output):
        preds, targets = output[0], output[1]
        self.pred_list.extend(preds)
        self.gold_list.extend(targets)

    def test_epoch_end(self, outputs):
        Path(self.model_output_dir).mkdir(exist_ok=True)
        output_pred_file = os.path.join(self.model_output_dir, "test_pred.json")
        pred_lst_formatted, gold_lst_formatted = [], []
        with open(output_pred_file, "w+") as fo:
            for pred, gold in zip(self.pred_list, self.gold_list):
                if self.decoding_format == "flat":
                    pred = text_post_process(pred)
                    pred = text_format(pred)
                    gold = text_format(gold)
                pred_lst_formatted.append(pred)
                gold_lst_formatted.append(gold)
                data = {
                    "pred": pred,
                    "gold": gold,
                }
                if self.use_zh:
                    fo.write(json.dumps(data, ensure_ascii=False) + "\n")
                else:
                    fo.write(json.dumps(data) + "\n")
        if self.decoding_format == "flat":
            results = self.compute_metrics(preds=pred_lst_formatted, targets=gold_lst_formatted)
        elif self.decoding_format == "tree":
            results = self.compute_metrics(preds=self.pred_list, targets=self.gold_list)
        output_test_result_file = os.path.join(self.model_output_dir, "test_results.txt")
        print("------ test_epoch results ------")
        with open(output_test_result_file, "w+") as fo:
            for key, value in sorted(results.items()):
                print(f"{key} = {value}\n")
                fo.write(f"{key} = {value}\n")

    def compute_metrics(self, preds, targets):
        result = get_extract_metrics(
            pred_lns=preds,
            tgt_lns=targets,
            decoding_format=self.decoding_format,
        )
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.learning_rate,
                          eps=self.adam_epsilon)
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps,
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

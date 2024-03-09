import json
import os
from pathlib import Path
from pytorch_lightning import LightningModule
from transformers import MT5ForConditionalGeneration, AutoTokenizer, get_linear_schedule_with_warmup, AdamW, \
    MT5Tokenizer

from extraction.extraction_metrics import get_extract_metrics


class MT5Model(LightningModule):
    def __init__(self, model_name="mt5-small", learning_rate=1e-5, batch_size=8, max_output_length=128,
                 model_output_name: str = "./",
                 freeze_embeds: bool = False, freeze_encoder: bool = True,
                 train_epoch=10, train_dataset_length=11116,
                 new_tokens: list = [], decoding_format="tree"
                 ) -> None:
        print("⚡", "using MT5ModelModule", "⚡")
        super().__init__()
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = MT5Tokenizer.from_pretrained(model_name)
        self.tokenizer.add_tokens(new_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        print("MT5 model: ", self.model)
        print("MT5 tokenizer: ", self.tokenizer)
        if freeze_embeds:
            self.freeze_embeds()
        if freeze_encoder:
            self.freeze_params(self.model.get_encoder())
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        self.warmup_steps = 0
        self.train_dataset_length = train_dataset_length
        self.train_epoch = train_epoch
        self.gradient_accumulation_steps = 1
        self.t_total = ((self.train_dataset_length // self.batch_size) // self.gradient_accumulation_steps
                        * float(self.train_epoch))
        Path("./output").mkdir(exist_ok=True)
        self.model_output_dir = os.path.join("./output", model_output_name)
        self.max_output_length = max_output_length
        self.decoding_format = decoding_format

        self.pred_list, self.gold_list = [], []

    def freeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = False

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            self.freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                self.freeze_params(d.embed_positions)
                self.freeze_params(d.embed_tokens)
        except AttributeError:
            self.freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                self.freeze_params(d.embed_tokens)

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self.model(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
        )
        loss = outputs[0]
        print("loss: ", loss)
        return loss

    def _generative_step(self, batch):
        generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            decoder_attention_mask=batch['target_mask'],
            max_length=self.max_output_length,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )
        preds = self.ids_to_clean_text(generated_ids)
        batch["target_ids"][batch["target_ids"][:, :] == -100] = self.tokenizer.pad_token_id
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
        with open(output_pred_file, "a+", encoding="utf-8") as fo:
            for pred, gold in zip(preds, targets):
                data = {
                    "pred": pred,
                    "target": gold,
                }
                fo.write(json.dumps(data, ensure_ascii=False) + "\n")
        return loss

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        gen_text = [text.replace('<pad>', '').replace('<s>', '').replace('</s>', '').strip()
                    for text in gen_text]
        return gen_text

    def test_step(self, batch, batch_id):
        print("\n using test_step \n")
        # print("batch: ", batch)
        preds, targets = self._generative_step(batch)
        return preds, targets

    def test_step_end(self, output):
        preds, targets = output[0], output[1]
        self.pred_list.extend(preds)
        self.gold_list.extend(targets)

    def test_epoch_end(self, outputs):
        Path(self.model_output_dir).mkdir(exist_ok=True)
        output_pred_file = os.path.join(self.model_output_dir, "test_pred.json")
        with open(output_pred_file, "w+") as fo:
            for pred, gold in zip(self.pred_list, self.gold_list):
                data = {
                    "pred": pred,
                    "target": gold,
                }
                fo.write(json.dumps(data, ensure_ascii=False) + "\n")
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

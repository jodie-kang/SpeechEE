from pytorch_lightning.cli import LightningCLI
from models.t5_model_module import T5Model
from models.whisper_model_module import WhisperModel
from models.mt5_model_module import MT5Model
from data_modules.Whisper_data_module import DataModuleForWhisper, DatasetForWhisper
from data_modules.T5_data_module import DataModuleForT5, DatasetForT5
from data_modules.mT5_data_module import DataModuleForMT5, DatasetForMT5


def cli_main():
    cli = LightningCLI(save_config_overwrite=True)


if __name__ == "__main__":
    cli_main()

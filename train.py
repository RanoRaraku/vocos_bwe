from pytorch_lightning.cli import LightningCLI
from vocos.experiment import VocosExp
from vocos.heads import ISTFTHead
from vocos.models import VocosBackbone
from vocos.feature_extractors import MelSpectrogramFeatures
import yaml

config_file_path = 'configs/vocos.yaml'
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

if __name__ == "__main__":
    cli = LightningCLI(run=False)
    model = VocosExp.load_from_checkpoint(
        "vocos_checkpoint_epoch=17_step=3276_val_loss=7.2563.ckpt",
        feature_extractor = MelSpectrogramFeatures(**config['model']['init_args']['feature_extractor']['init_args']),
        backbone = VocosBackbone(**config['model']['init_args']['backbone']['init_args']),
        head = ISTFTHead(**config['model']['init_args']['head']['init_args']),
        sample_rate = int(config['model']['init_args']['sample_rate']),
        initial_learning_rate = float(config['model']['init_args']['initial_learning_rate']),
        mel_loss_coeff = float(config['model']['init_args']['mel_loss_coeff']),
        mrd_loss_coeff = float(config['model']['init_args']['mrd_loss_coeff']),
        num_warmup_steps = int(config['model']['init_args']['num_warmup_steps']), # Optimizers warmup steps
        pretrain_mel_steps = int(config['model']['init_args']['pretrain_mel_steps']),  # 0 means GAN objective from the first iteration
    )

    cli.trainer.fit(model, datamodule=cli.datamodule)

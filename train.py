from vocos.experiment import VocosExp
from vocos.heads import ISTFTHead
from vocos.models import VocosBackbone
from data.feature_extractors import MelSpectrogramFeatures
import yaml

config_file_path = 'configs/vocos.yaml'
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

if __name__ == "__main__":
    model = VocosExp(
        feature_extractor = MelSpectrogramFeatures(**config['model']['init_args']['feature_extractor']['init_args']),
        backbone = VocosBackbone(**config['model']['init_args']['backbone']['init_args']),
        head = ISTFTHead(**config['model']['init_args']['head']['init_args']),
        sample_rate = int(config['model']['init_args']['sample_rate']),
        initial_learning_rate = float(config['model']['init_args']['initial_learning_rate']),
        num_warmup_steps = int(config['model']['init_args']['num_warmup_steps']), # Optimizers warmup steps
        mel_loss_coeff = float(config['model']['init_args']['mel_loss_coeff']),
        mrd_loss_coeff = float(config['model']['init_args']['mrd_loss_coeff']),
        pretrain_mel_steps = int(config['model']['init_args']['pretrain_mel_steps']),  # 0 means GAN objective from the first iteration
    )

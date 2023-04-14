from dataclasses import dataclass
import hydra
import pandas as pd
import chat_hall_label
import openai

from retry import retry
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


@dataclass
class OpenAiConfig:
    organization: str
    api_key: str
    model: str


@dataclass
class DataConfig:
    input_csv: str
    output_csv: str
    input_text_column: str
    generated_text_column: str
    output_column_prefix: str


@dataclass
class GptHallucinationDetectionConfig:
    data: DataConfig
    openai: OpenAiConfig


cs = ConfigStore.instance()
cs.store(name="config", node=GptHallucinationDetectionConfig)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: GptHallucinationDetectionConfig):
    print(OmegaConf.to_yaml(cfg))

    df = pd.read_csv(cfg.data.input_csv)

    openai.organization = cfg.openai.organization
    openai.api_key = cfg.openai.api_key

    retry(
        lambda: chat_hall_label.complete(
            openai,
            df,
            model=cfg.openai.model,
            input_text_column=cfg.data.input_text_column,
            generated_text_column=cfg.data.generated_text_column,
            output_column_prefix=cfg.data.output_column_prefix,
        )
    )
    
    df.to_csv(cfg.data.output_csv)


if __name__ == "__main__":
    main()

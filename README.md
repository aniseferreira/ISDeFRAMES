# ISD e Frames — Streamlit

Aplicação para análise linguística baseada no Interacionismo Sociodiscursivo (ISD)
e na Semântica de Frames.

Esta versão é uma adaptação do Space Hugging Face:

AniseF/ISD-e-Frames

## Estrutura

```text
ISD_e_Frames/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml   # NÃO enviar ao GitHub
```

## Dependências

O aplicativo carrega localmente apenas o classificador:

`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`

A análise generativa usa o modelo:

`Qwen/Qwen2.5-7B-Instruct`

por meio da Inference API do Hugging Face. Portanto, o Qwen não precisa ser
baixado para o Streamlit Cloud.

## Token do Hugging Face

Crie um token do Hugging Face com permissão para Inference Providers.

### Localmente

Crie:

`.streamlit/secrets.toml`

com:

```toml
HF_TOKEN = "hf_SEU_TOKEN_AQUI"
```

Esse arquivo está no `.gitignore` e não deve ser enviado ao GitHub.

### Streamlit Community Cloud

Depois de selecionar o repositório e o `app.py`, abra:

Settings → Secrets

e cole:

```toml
HF_TOKEN = "hf_SEU_TOKEN_AQUI"
```

## Execução local

```bash
pip install -r requirements.txt
streamlit run app.py
```

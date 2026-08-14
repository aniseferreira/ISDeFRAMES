import json
import os

import streamlit as st
from huggingface_hub import InferenceClient
from transformers import pipeline


# ============================================================
# ISD & FrameNet — Streamlit
# Adapted from the Hugging Face Space:
# AniseF/ISD-e-Frames
# LICENÇA CC BY-SA 4.0
# ============================================================

st.set_page_config(
    page_title="Analisador Multilíngue ISD & FrameNet",
    page_icon="assets/favicon.png",
    layout="wide",
)

col_logo, col_titulo = st.columns([1, 8])

with col_logo:
    st.image("assets/icone.png", width=80)

with col_titulo:
    st.title("Analisador Linguístico Avançado (PT / FR)")
    st.subheader(
        "Arquitetura Textual do ISD (Bronckart) + "
        "Semântica de Frames (Fillmore)"
    )

# ------------------------------------------------------------
# 1. Hugging Face token
# ------------------------------------------------------------
def get_hf_token():
    """Read HF_TOKEN from Streamlit Secrets or environment."""
    try:
        token = st.secrets.get("HF_TOKEN")
        if token:
            return token
    except Exception:
        pass

    return os.environ.get("HF_TOKEN")


hf_token = get_hf_token()

if not hf_token:
    st.warning(
        "O token HF_TOKEN ainda não foi configurado. "
        "Configure-o nos Secrets do Streamlit Cloud antes de executar a análise."
    )


# ------------------------------------------------------------
# 2. Hugging Face Inference Client
# ------------------------------------------------------------
@st.cache_resource
def get_client(token):
    if not token:
        return None

    return InferenceClient(
        model="Qwen/Qwen2.5-7B-Instruct",
        token=token,
    )


client = get_client(hf_token)


# ------------------------------------------------------------
# 3. Classificador de Mundos Discursivos
# ------------------------------------------------------------
@st.cache_resource
def load_classifier():
    return pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    )


# O modelo é carregado somente quando a aplicação tem um texto
# para analisar, evitando download desnecessário ao abrir a página.
classifier = None


# ------------------------------------------------------------
# 4. Funções auxiliares
# ------------------------------------------------------------
def clean_json_response(text):
    """Remove cercas Markdown caso o LLM devolva ```json ... ```."""
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip().lower() in {"```json", "```"}:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return text


def run_chat(prompt, text, max_tokens):
    if client is None:
        raise RuntimeError(
            "HF_TOKEN não está configurado. "
            "Configure o token nos Secrets do Streamlit Cloud."
        )

    response = client.chat_completion(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content.strip()


# ------------------------------------------------------------
# 5. Barra lateral
# ------------------------------------------------------------
st.sidebar.header("📝 Contexto Inicial Declarado")

st.sidebar.markdown(
    "<p style='font-size: 16px; font-weight: bold; margin-bottom: 8px;'>"
    "Idioma do Corpus</p>",
    unsafe_allow_html=True,
)

idioma = st.sidebar.selectbox(
    "",
    ["Português", "Français"],
    label_visibility="collapsed",
)

st.sidebar.markdown(
    "<p style='font-size: 16px; font-weight: bold; margin-bottom: 8px;'>"
    "Estatuto do Autor/Emissor</p>",
    unsafe_allow_html=True,
)

autor = st.sidebar.text_input(
    "",
    placeholder="Ex.: Jornalista, Cientista...",
    label_visibility="collapsed",
)

st.sidebar.markdown(
    "<p style='font-size: 16px; font-weight: bold; margin-bottom: 8px;'>"
    "Objetivo Estimado</p>",
    unsafe_allow_html=True,
)

objetivo_input = st.sidebar.text_input(
    "",
    placeholder="Ex.: Persuadir, Denunciar...",
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "©️2026 Idealizado por Anise d'Orange Ferreira. CC BY-SA 4.0 "
    "Desenvolvimento realizado com assistência " 
    "do Gemini 1.5 Pro para a plataforma Hugging Face. " 
    "Assistido pelo GPT-5.6 Luna para a plataforma Streamlit usando GitHub. "
)


# ------------------------------------------------------------
# 6. Texto de entrada
# ------------------------------------------------------------
st.markdown(
    "<p style='font-size: 18px; font-weight: bold; margin-bottom: 5px;'>"
    "Insira o fragmento de texto para análise em Português ou Francês</p>",
    unsafe_allow_html=True,
)

texto_input = st.text_area(
    "",
    height=200,
    placeholder="Digite ou cole seu texto aqui...",
    label_visibility="collapsed",
)


# ------------------------------------------------------------
# 7. Análise
# ------------------------------------------------------------
if st.button("Analisar Texto Completo", type="primary"):

    if not texto_input.strip():
        st.error("Por favor, insira um texto para análise.")
        st.stop()

    if not hf_token:
        st.error(
            "HF_TOKEN não configurado. "
            "No Streamlit Cloud, abra Settings → Secrets e adicione o token."
        )
        st.stop()

    with st.spinner("Carregando o classificador multilíngue..."):
        classifier = load_classifier()

    col1, col2 = st.columns(2)

    # ========================================================
    # COLUNA 1 — ISD
    # ========================================================
    with col1:
        st.header("📊 Camada de Bronckart (ISD)")

        labels = [
            "Discurso Teórico / Discours Théorique",
            "Narração / Narration",
            "Discurso Interativo / Discours Interactif",
        ]

        try:
            res = classifier(
                texto_input,
                candidate_labels=labels,
            )
        except Exception as e:
            st.error(f"Erro no classificador de mundos discursivos: {e}")
            res = None

        prompt_isd = (
            "Você é um linguista sênior especialista no "
            "Interacionismo Sociodiscursivo (ISD) de Jean-Paul Bronckart "
            "e Jean-Michel Adam.\n"
            f"Analise o texto considerando que o autor declarado é "
            f"'{autor}' e o objetivo estimado é '{objetivo_input}'.\n"
            "Responda EXCLUSIVAMENTE em formato JSON estrito, sem markdown, "
            "sem introduções. "
            "Siga rigorosamente as seguintes diretrizes teóricas:\n\n"

            "1. 'contexto_produção': Mapeie o emissor (estatuto social), "
            "receptor (público-alvo), suporte (onde circula), o objetivo "
            "(efeito na sociedade) e o 'genero_textual' (gênero específico "
            "com base nas características sociodiscursivas).\n\n"

            "2. 'sequencias_textuais': Atribua porcentagens (0 a 100) "
            "para a presença das 5 sequências de Jean-Michel Adam: "
            "narrativa, descritiva, argumentativa, explicativa e injuntiva.\n\n"

            "3. 'mecanismos_textualizacao':\n"
            " - Foque apenas na amarração linear do texto. "
            "Não misture com vozes ou opiniões.\n"
            " - 'coesao_nominal': Analise as cadeias de referência "
            "anafórica. Como pronomes, sinônimos, repetições ou elipses "
            "retomam os referentes textuais. Dê exemplos exatos do texto.\n"
            " - 'coesao_verbal': Analise a ordenação e a correlação dos "
            "tempos verbais e seu papel na progressão textual.\n\n"

            "4. 'mecanismos_enunciativos':\n"
            " - Identifique como a subjetividade e as perspectivas são "
            "encenadas.\n"
            " - 'gerenciamento_vozes': Identifique a polifonia textual. "
            "Classifique as vozes segundo o ISD: Voz do Narrador/Expositor, "
            "Voz de Personagem, Voz de Instância Social ou Voz do Autor "
            "Empírico. Dê exemplos e indique se são diretas, indiretas "
            "ou implícitas.\n"
            " - 'modalizacoes': Classifique as marcas de avaliação, quando "
            "presentes, em Apreciativas, Lógicas, Deônticas ou Pragmáticas. "
            "Indique os itens lexicais exatos do texto.\n\n"

            "Estrutura EXATA do JSON esperado:\n"
            "{\n"
            '  "contexto_produção": {'
            '"emissor": "", "receptor": "", "suporte_circulacao": "", '
            '"objetivo_sociedade": "", "genero_textual": ""},\n'
            '  "sequencias_textuais": {'
            '"narrativa": 0, "descritiva": 0, "argumentativa": 0, '
            '"explicativa": 0, "injuntiva": 0},\n'
            '  "mecanismos_textualizacao": {'
            '"coesao_nominal": "", "coesao_verbal": ""},\n'
            '  "mecanismos_enunciativos": {'
            '"gerenciamento_vozes": "", "modalizacoes": ""}\n'
            "}"
        )

        with st.spinner("Decodificando camadas do ISD..."):
            try:
                texto_isd = run_chat(
                    prompt_isd,
                    texto_input,
                    max_tokens=800,
                )

                dados_isd = json.loads(clean_json_response(texto_isd))

                st.subheader("1. Contexto de Produção Pragmático")

                cp = dados_isd.get("contexto_produção", {})

                st.write(
                    "📂 **Gênero Textual Detectado:** "
                    f"{cp.get('genero_textual', 'Não identificado')}"
                )
                st.write(
                    "✍️ **Emissor (Estatuto):** "
                    f"{cp.get('emissor', 'Não identificado')}"
                )
                st.write(
                    "👥 **Receptor (Público-Alvo):** "
                    f"{cp.get('receptor', 'Não identificado')}"
                )
                st.write(
                    "📍 **Suporte e Circulação:** "
                    f"{cp.get('suporte_circulacao', 'Não identificado')}"
                )
                st.write(
                    "🎯 **Efeito Pretendido na Sociedade:** "
                    f"{cp.get('objetivo_sociedade', 'Não identificado')}"
                )

                st.divider()
                st.subheader("2. Infraestrutura: Mundos & Sequências")

                if res:
                    st.write(
                        "🔮 **Mundo Discursivo Predominante:** "
                        f"{res['labels'][0]} "
                        f"({res['scores'][0] * 100:.1f}%)"
                    )

                st.write("**Tipologia de Sequências (Adam):**")

                for seq, valor in dados_isd.get(
                    "sequencias_textuais", {}
                ).items():
                    try:
                        percentual = float(valor)
                    except (TypeError, ValueError):
                        percentual = 0.0

                    st.text(f"- Sequência {seq.capitalize()}:")
                    st.progress(max(0.0, min(1.0, percentual / 100)))

                st.divider()
                st.subheader("3. Mecanismos de Textualização")

                mt = dados_isd.get(
                    "mecanismos_textualizacao",
                    {},
                )

                st.write(
                    "🔗 **Coesão Nominal:** "
                    f"{mt.get('coesao_nominal', 'Não analisado')}"
                )
                st.write(
                    "⏳ **Coesão Verbal:** "
                    f"{mt.get('coesao_verbal', 'Não analisado')}"
                )

                st.divider()
                st.subheader("4. Mecanismos Enunciativos")

                me = dados_isd.get(
                    "mecanismos_enunciativos",
                    {},
                )

                st.write(
                    "🗣️ **Gerenciamento de Vozes:** "
                    f"{me.get('gerenciamento_vozes', 'Não analisado')}"
                )
                st.write(
                    "🦉 **Modalizações Detectadas:** "
                    f"{me.get('modalizacoes', 'Não analisado')}"
                )

            except Exception as e:
                st.error(f"Erro ao processar camada ISD: {e}")

    # ========================================================
    # COLUNA 2 — FRAME SEMANTICS
    # ========================================================
    with col2:
        st.header("🧠 Semântica de Frames Universal")

        prompt_frame = (
            "Você é um linguista computacional especialista na "
            "Semântica de Frames de Fillmore, utilizando estritamente "
            "as taxonomias oficiais da Berkeley FrameNet "
            "(para inglês/geral) e da Asfalda French FrameNet "
            "(para termos em francês).\n\n"

            "Instruções de análise:\n"
            "1. Identifique a Unidade Lexical (palavra-gatilho) "
            "principal do texto.\n"
            "2. Determine o nome oficial do Frame ativado "
            "(ex.: Commerce_buy, Statement, Motion).\n"
            "3. Mapeie os Elementos de Frame (FEs), extraindo "
            "as palavras exatas do texto original.\n"
            "4. Se o texto for em francês, use a correspondência "
            "conceitual validada pela Asfalda.\n\n"

            "Responda EXCLUSIVAMENTE com um objeto JSON válido, "
            "sem comentários e sem marcações markdown:\n"
            "{\n"
            '  "frame": "NOME_OFICIAL_DO_FRAME",\n'
            '  "unidade_lexical": "palavra_gatilho",\n'
            '  "elementos": {"Nome_Do_Elemento_Oficial": "trecho do texto"}\n'
            "}"
        )

        with st.spinner("Analisando frames..."):
            try:
                texto_frame = run_chat(
                    prompt_frame,
                    texto_input,
                    max_tokens=400,
                )

                dados_frame = json.loads(
                    clean_json_response(texto_frame)
                )

                st.success(
                    "🔓 **Frame Detectado:** "
                    f"`{dados_frame.get('frame', 'Desconhecido')}`"
                )

                st.write(
                    "**Unidade Lexical:** "
                    f"{dados_frame.get('unidade_lexical', 'Não identificada')}"
                )

                st.write("**Elementos mapeados no texto:**")

                for elemento, valor in dados_frame.get(
                    "elementos",
                    {},
                ).items():
                    st.write(f"- *{elemento}:* {valor}")

            except Exception as e:
                st.error(f"Erro ao processar Frame: {e}")

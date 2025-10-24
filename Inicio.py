import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# --- Configuración de la página ---
st.set_page_config(
    page_title="Demo TF-IDF Q&A",
    page_icon="💬",
    layout="centered"
)

# --- Estilos personalizados ---
st.markdown("""
    <style>
    /* Fondo general */
    .stApp {
        background: linear-gradient(135deg, #f7f7fa, #e8ebf7);
        color: #1b1b1b;
        font-family: "Inter", sans-serif;
    }

    /* Título */
    h1 {
        color: #5b32b4; /* violeta */
        text-align: center;
        font-weight: 800;
        padding-bottom: 0.5rem;
    }

    /* Subtítulos */
    h2, h3 {
        color: #30408d; /* azul oscuro */
        font-weight: 700;
        margin-top: 1rem;
    }

    /* Cuadros de texto */
    textarea, input {
        border-radius: 10px !important;
        border: 1px solid #b3b6d3 !important;
        background-color: #ffffff !important;
    }

    /* Botón principal */
    div.stButton > button {
        background-color: #5b32b4;
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        transition: all 0.3s ease;
        box-shadow: 0px 3px 6px rgba(0,0,0,0.2);
    }
    div.stButton > button:hover {
        background-color: #7b52d9;
        transform: scale(1.05);
        box-shadow: 0px 4px 8px rgba(0,0,0,0.25);
    }

    /* Tablas */
    .dataframe {
        border-radius: 8px;
        border: 1px solid #d1d3e0;
        overflow: hidden;
    }

    /* Textos */
    p, li {
        color: #2a2a2a !important;
        font-size: 1rem;
    }

    /* Explicaciones */
    .stMarkdown {
        color: #3c3c3c;
        font-size: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Título principal ---
st.title("💬 Demo de TF-IDF con Preguntas y Respuestas")

st.write("""
Cada línea se trata como un **documento** (puede ser una frase, un párrafo o un texto más largo).  
⚠️ Los documentos y las preguntas deben estar en **inglés**, ya que el análisis está configurado para ese idioma.  

La aplicación aplica normalización y *stemming* para que palabras como *playing* y *play* se consideren equivalentes.
""")

# --- Entrada de texto ---
text_input = st.text_area(
    "📝 Escribe tus documentos (uno por línea, en inglés):",
    "The dog barks loudly.\nThe cat meows at night.\nThe dog and the cat play together."
)

question = st.text_input("🔍 Escribe una pregunta (en inglés):", "Who is playing?")

# --- Configuración del stemmer ---
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text: str):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# --- Botón para ejecutar ---
if st.button("🚀 Calcular TF-IDF y buscar respuesta", use_container_width=True):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    if len(documents) < 1:
        st.warning("⚠️ Ingresa al menos un documento.")
    else:
        # Vectorización con stemming
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            stop_words="english",
            token_pattern=None
        )

        X = vectorizer.fit_transform(documents)

        # Matriz TF-IDF
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )

        st.subheader("📊 Matriz TF-IDF (stems)")
        st.dataframe(df_tfidf.round(3))

        # Similaridad coseno con la pregunta
        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()

        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]

        st.subheader("💡 Resultado de búsqueda")
        st.write(f"**Tu pregunta:** {question}")
        st.write(f"**Documento más relevante (Doc {best_idx+1}):** {best_doc}")
        st.success(f"**Puntaje de similitud:** {best_score:.3f}")

        sim_df = pd.DataFrame({
            "Documento": [f"Doc {i+1}" for i in range(len(documents))],
            "Texto": documents,
            "Similitud": similarities
        })
        st.subheader("📈 Puntajes de similitud (ordenados)")
        st.dataframe(sim_df.sort_values("Similitud", ascending=False))

        vocab = vectorizer.get_feature_names_out()
        q_stems = tokenize_and_stem(question)
        matched = [s for s in q_stems if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0]
        st.subheader("🧩 Stems de la pregunta presentes en el documento elegido")
        st.write(matched)






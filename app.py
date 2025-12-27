
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# -------------------- QA PAIRS (MULTI-INTENT) --------------------






# -------------------- QA PAIRS (AI / ML / DS / Career) --------------------

qa_pairs = [
    # AI Basics
    ("what is artificial intelligence",
     "Artificial Intelligence is the ability of machines to mimic human intelligence."),

    ("define ai",
     "AI enables machines to think, learn, and make decisions like humans."),

    ("applications of ai",
     "AI is used in healthcare, finance, self-driving cars, recommendation systems, and robotics."),

    ("who invented ai",
     "John McCarthy is known as the father of Artificial Intelligence."),

    # Machine Learning
    ("what is machine learning",
     "Machine Learning allows systems to learn from data without explicit programming."),

    ("types of machine learning",
     "There are three main types: Supervised, Unsupervised, and Reinforcement Learning."),

    ("difference between ml and dl",
     "Machine Learning uses algorithms to learn from data, while Deep Learning uses deep neural networks."),

    ("supervised learning examples",
     "Examples include spam detection, sentiment analysis, and house price prediction."),

    ("unsupervised learning examples",
     "Examples include customer segmentation, anomaly detection, and clustering."),

    ("reinforcement learning examples",
     "Examples include game-playing AI like AlphaGo, robotics, and recommendation systems."),

    # Deep Learning
    ("what is deep learning",
     "Deep Learning uses multi-layer neural networks to learn complex patterns from large datasets."),

    ("applications of deep learning",
     "DL is used in image recognition, NLP, speech recognition, self-driving cars, and medical diagnosis."),

    ("cnn definition",
     "Convolutional Neural Networks (CNNs) are used primarily for image data and computer vision tasks."),

    ("rnn definition",
     "Recurrent Neural Networks (RNNs) are used for sequential data like time series and text."),

    ("what is lstm",
     "LSTM (Long Short-Term Memory) is a type of RNN that can remember long-term dependencies."),

    # NLP
    ("what is nlp",
     "Natural Language Processing enables machines to understand, interpret, and generate human language."),

    ("applications of nlp",
     "NLP is used in chatbots, translation, sentiment analysis, text summarization, and search engines."),

    ("what is glove",
     "GloVe is a word embedding method based on global word co-occurrence that represents words as vectors."),

    ("what are word embeddings",
     "Word embeddings convert words into dense numerical vectors that capture semantic meaning."),

    ("what is bert",
     "BERT is a transformer-based model for understanding the context of words in NLP tasks."),

    ("what is transformers",
     "Transformers are neural networks that process sequential data in parallel using self-attention mechanisms."),

    # Data Science
    ("what is data science",
     "Data Science is the field of extracting insights from structured and unstructured data using statistics, ML, and programming."),

    ("skills required for data science",
     "Python/R, statistics, ML, data visualization, SQL, problem solving, and communication skills."),

    ("tools used in data science",
     "Popular tools include Python, R, Jupyter Notebook, Pandas, NumPy, Matplotlib, Tableau, and SQL."),

    ("difference between data science and machine learning",
     "Data Science is broader; ML is a subset focused on building predictive models."),

    ("what is big data",
     "Big Data refers to datasets that are too large or complex to process with traditional methods."),

    ("big data tools",
     "Hadoop, Spark, Hive, and Kafka are commonly used big data tools."),

    # AI/ML Career
    ("how to become a data scientist",
     "Learn Python/R, statistics, ML, data visualization, complete projects, and participate in Kaggle competitions."),

    ("how to become a machine learning engineer",
     "Focus on Python, ML, deep learning, data preprocessing, and production deployment skills."),

    ("career in ai",
     "AI offers roles like ML Engineer, Data Scientist, NLP Engineer, Computer Vision Engineer, AI Researcher."),

    ("how to learn ai",
     "Start with Python, then ML basics, data science, deep learning, and practice with projects."),

    ("high paying jobs in data science",
     "ML Engineer, Data Scientist, AI Researcher, Data Engineer, NLP Engineer are high-paying roles."),

    ("entry level jobs in ai",
     "AI/ML Intern, Data Analyst, Junior Data Scientist, Research Assistant in AI labs."),

    ("should i learn python for ai",
     "Yes! Python is the most widely used language for AI, ML, and Data Science."),

    ("should i learn statistics for data science",
     "Absolutely! Understanding statistics is crucial for data analysis and building ML models."),

    # Advanced Topics
    ("what is reinforcement learning",
     "Reinforcement Learning is a type of ML where an agent learns to make decisions by receiving rewards or penalties."),

    ("what is supervised learning",
     "Supervised learning uses labeled data to train models to predict outputs from inputs."),

    ("what is unsupervised learning",
     "Unsupervised learning finds patterns or structures in unlabeled data."),

    ("what is anomaly detection",
     "Anomaly detection identifies unusual patterns that do not conform to expected behavior."),

    ("what is computer vision",
     "Computer Vision enables machines to interpret and understand visual data from the world."),

    ("applications of computer vision",
     "CV is used in facial recognition, self-driving cars, medical imaging, and industrial inspection."),

    # Fun / Misc
    ("hello",
     "Hello! Ask me anything about AI, ML, or Data Science 🤖"),

    ("hi",
     "Hi there! How can I help you today?"),

    ("who are you",
     "I am an AI mentor bot built to answer your AI, ML, and Data Science questions 😄")
]















# -------------------- STOP WORDS --------------------

stop_words = {
    "a","about","above","after","again","against","all","am","an","and","any",
    "are","as","at","be","because","been","before","being","below","between",
    "both","but","by","can","could","did","do","does","doing","down","during",
    "each","few","for","from","further","had","has","have","having","he","her",
    "here","hers","herself","him","himself","his","how","i","if","in","into",
    "is","it","its","itself","just","me","more","most","my","myself","no",
    "nor","not","now","of","off","on","once","only","or","other","our","ours",
    "ourselves","out","over","own","same","she","should","so","some","such",
    "than","that","the","their","theirs","them","themselves","then","there",
    "these","they","this","those","through","to","too","under","until","up",
    "very","was","we","were","what","when","where","which","while","who",
    "whom","why","with","would","you","your","yours","yourself","yourselves"
}

# -------------------- FUNCTIONS --------------------

def load_glove_embeddings(path):
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    return embeddings


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    words = text.split()
    return [w for w in words if w not in stop_words]


def sentence_vector(words, embeddings, dim=300):
    vectors = [embeddings[w] for w in words if w in embeddings]
    if not vectors:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)


def get_best_answer(user_question, qa_pairs, embeddings, threshold=0.3):
    user_words = preprocess(user_question)
    user_vec = sentence_vector(user_words, embeddings)

    similarities = []
    for q, _ in qa_pairs:
        q_words = preprocess(q)
        q_vec = sentence_vector(q_words, embeddings)
        sim = cosine_similarity([user_vec], [q_vec])[0][0]
        similarities.append(sim)

    best_idx = np.argmax(similarities)

    if similarities[best_idx] < threshold:
        return "Sorry, I don't understand that yet 😔"

    return qa_pairs[best_idx][1]

# -------------------- STREAMLIT UI --------------------

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 AI Chatbot (GloVe + Cosine Similarity)")
st.write("Type **bye** to end the conversation")

@st.cache_resource
def load_embeddings():
    return load_glove_embeddings("dolma_300_2024_1.2M.100_combined.txt")

with st.spinner("Loading embeddings..."):
    embeddings = load_embeddings()

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask your question...")

if user_input:
    # Add user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    # Bye condition
    if user_input.lower() in ["bye", "exit", "quit"]:
        bot_reply = "Bye 👋 Keep learning AI, you're doing great 🚀"
    else:
        bot_reply = get_best_answer(user_input, qa_pairs, embeddings)

    # Add bot reply
    st.session_state.messages.append(
        {"role": "assistant", "content": bot_reply}
    )

    st.rerun()

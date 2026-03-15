import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

st.set_page_config(page_title="ML Chatbot", layout="centered")
st.title("Mini Project 2: Multi-Agent ML Chatbot")
st.caption("Powered by GPT-4.1-nano · Pinecone RAG · Multi-Agent Pipeline")

# ── Load API keys from local files ────
try:
    openai_api_key = open("open_ai_key.txt", "r", encoding="utf-8").read().strip()
    pinecone_api_key = open("pinecone_api.txt", "r", encoding="utf-8").read().strip()
except FileNotFoundError:
    try:
        openai_api_key = st.secrets["openai_api_key"]
        pinecone_api_key = st.secrets["pinecone_api_key"]
    except KeyError as e:
        st.error(f"Missing API key: {e}")
        st.stop()

PINECONE_INDEX_NAME = "miniproject2-machine-learning-textbook"


class Obnoxious_Agent:
    def __init__(self, client) -> None:
        self.client = client
        self.prompt = None
        self.set_prompt(
            "You are a content moderation assistant. "
            "Your task is to determine whether a user's message contains ANY obnoxious content, "
            "meaning rude, offensive, abusive, hateful, or disrespectful language — "
            "even if the message also contains a legitimate question. "
            "This includes profanity and swear words such as: fuck, shit, damn, hell, ass, bitch, crap, "
            "or any variation or censored form (e.g. f***, s***, wtf, wth). "
            "If ANY part of the message contains such language, reply with ONLY 'Yes'. "
            "Reply with ONLY 'No' if the entire message is respectful and appropriate. "
            "Do not include punctuation, explanation, or any other text."
        )

    def set_prompt(self, prompt):
        self.prompt = prompt

    def extract_action(self, response) -> bool:
        raw = response.choices[0].message.content
        normalized = raw.strip().lower().strip(".,!?")
        return normalized == "yes"

    def check_query(self, query) -> bool:
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user",   "content": query}
            ],
            temperature=0
        )
        return self.extract_action(response)


class Context_Rewriter_Agent:
    def __init__(self, openai_client):
        self.client = openai_client
        # [IMPROVED] Explicit instruction to replace all pronouns with specific concepts
        self.prompt = (
            "You are a query rewriting assistant. "
            "Given a conversation history and the user's latest message, "
            "rewrite the latest message into a single, self-contained question "
            "that can be understood without any prior context. "
            "IMPORTANT: Replace ALL pronouns (it, this, that, they, these, them) "
            "with the specific concept or term they refer to from the conversation history. "
            "IMPORTANT: Expand short or colloquial queries into full, descriptive questions. "  # ← new added
            "For example, 'what's supervised learning' should become "                          # ← new added
            "'What is supervised learning and how does it work?'  "                             # ← new added
            "If the latest message is already clear and standalone, return it as-is. "
            "Output ONLY the rewritten query, no explanation or extra text."
        )

    def rephrase(self, user_history: list, latest_query: str) -> str:
        history_text = ""
        for msg in user_history:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            history_text += f"{role}: {content}\n"

        user_input = (
            f"Conversation History:\n{history_text}\n"
            f"Latest User Message: {latest_query}\n\n"
            f"Rewrite the latest message into a standalone question:"
        )

        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user",   "content": user_input}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()


class Query_Agent:
    def __init__(self, pinecone_index, openai_client, embeddings) -> None:
        self.index = pinecone_index
        self.client = openai_client
        self.embeddings = embeddings
        self.prompt = None
        self.set_prompt(
            "You are a relevance-checking assistant. "
            "You have access to a knowledge base about Machine Learning. "
            "Given a user query, determine if it is related to Machine Learning or the knowledge base content. "
            "Reply with ONLY 'Yes' if the query is relevant, or ONLY 'No' if it is not. "
            "Do not include punctuation, explanation, or any other text."
        )

    def query_vector_store(self, query: str, k: int = 5, namespace: str = "ns2500") -> list:
        query_embedding = self.embeddings.embed_query(query)
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            namespace=namespace,
            include_metadata=True
        )
        docs = []
        for match in results["matches"]:
            text = match.get("metadata", {}).get("text", "")
            score = match.get("score", 0)
            if text:
                docs.append({"text": text, "score": score})
        return docs

    def set_prompt(self, prompt: str):
        self.prompt = prompt

    def extract_action(self, response, query=None) -> bool:
        raw = response.choices[0].message.content
        normalized = raw.strip().lower().strip(".,!?")
        return normalized == "yes"

    def is_relevant(self, query: str) -> bool:
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user",   "content": query}
            ],
            temperature=0
        )
        return self.extract_action(response, query)


class Answering_Agent:
    def __init__(self, openai_client) -> None:
        self.client = openai_client
        self.prompt = (
            "You are a helpful teaching assistant for a Machine Learning textbook. "
            "Answer the user's question using ONLY the provided document context. "
            "If the context does not contain enough information to answer, "
            "say 'I don't have enough information to answer this question.' "
            "Be concise, accurate, and helpful."
        )

    def generate_response(self, query: str, docs: list, conv_history: list, k: int = 5) -> str:
        context = ""
        for i, doc in enumerate(docs[:k]):
            context += f"[Document {i+1}]:\n{doc.get('text', '')}\n\n"

        history_text = ""
        for msg in conv_history:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            history_text += f"{role}: {content}\n"

        user_message = (
            f"Conversation History:\n{history_text}\n"
            f"Relevant Context:\n{context}"
            f"Current Question: {query}\n\n"
            f"Please answer the question based on the context provided."
        )

        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user",   "content": user_message}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()


class Relevant_Documents_Agent:
    def __init__(self, openai_client) -> None:
        self.client = openai_client
        self.prompt = (
            "You are a relevance evaluation assistant. "
            "Given a user query and a set of retrieved document chunks, "
            "determine whether the documents contain useful information to answer the query. "
            "Reply with ONLY 'Yes' if the documents are relevant, or ONLY 'No' if they are not. "
            "Do not include punctuation, explanation, or any other text."
        )

    def get_relevance(self, conversation: dict) -> bool:
        query = conversation.get("query", "")
        docs  = conversation.get("docs", [])

        docs_text = ""
        for i, doc in enumerate(docs):
            docs_text += f"Document {i+1}:\n{doc.get('text', '')}\n\n"

        user_message = (
            f"User Query: {query}\n\n"
            f"Retrieved Documents:\n{docs_text}"
            f"Are these documents relevant to the query?"
        )

        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user",   "content": user_message}
            ],
            temperature=0
        )
        raw = response.choices[0].message.content
        normalized = raw.strip().lower().strip(".,!?")
        return normalized == "yes"


class Head_Agent:
    def __init__(self, openai_key: str, pinecone_key: str, pinecone_index_name: str) -> None:
        self.openai_client = OpenAI(api_key=openai_key)
        pc = Pinecone(api_key=pinecone_key)
        self.pinecone_index = pc.Index(pinecone_index_name)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_key
        )
        self.conv_history = []
        self.obnoxious_agent     = None
        self.context_rewriter    = None
        self.query_agent         = None
        self.relevant_docs_agent = None
        self.answering_agent     = None
        self.setup_sub_agents()

    def setup_sub_agents(self):
        self.obnoxious_agent     = Obnoxious_Agent(self.openai_client)
        self.context_rewriter    = Context_Rewriter_Agent(self.openai_client)
        self.query_agent         = Query_Agent(self.pinecone_index, self.openai_client, self.embeddings)
        self.relevant_docs_agent = Relevant_Documents_Agent(self.openai_client)
        self.answering_agent     = Answering_Agent(self.openai_client)

    def is_small_talk(self, query: str) -> bool:
        response = self.openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": (
                    "Determine if the user's message is casual small talk, a greeting, "
                    "or general conversational filler with no technical content. "
                    "This includes phrases like 'Hello', 'Hi there', 'How are you', "
                    "'Good morning', 'Are you there?', 'What's up', or similar. "
                    "Reply with ONLY 'Yes' if it is small talk, or 'No' if it is not."
                )},
                {"role": "user", "content": query}
            ],
            temperature=0
        )
        raw = response.choices[0].message.content.strip().lower().strip(".,!?")
        return raw == "yes"

    # [NEW] Hybrid query decomposer: extracts only the ML-relevant part
    def decompose_ml_query(self, query: str) -> str | None:
        """
        Extract only the Machine Learning related question from a potentially hybrid query.
        Returns None if no ML content is found at all.
        Returns the original query if it is already entirely ML-related.
        Returns only the ML component if the query mixes ML with off-topic content.
        """
        response = self.openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": (
                    "You are a query decomposition assistant. "
                    "Your job is to extract the Machine Learning related question from the user's message. "
                    "Examples of ML topics: gradient descent, overfitting, neural networks, "
                    "supervised learning, regression, classification, backpropagation, SVM, etc. "
                    "If the message contains an ML question mixed with unrelated content, "
                    "return ONLY the ML question. "
                    "If the entire message is about ML, return it as-is. "
                    "If there is absolutely no ML content, return exactly 'NONE'. "
                    "Output only the question text, no explanation."
                )},
                {"role": "user", "content": query}
            ],
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        return None if result.upper() == "NONE" else result

    def expand_query(self, query: str) -> str:
        response = self.openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": (
                    "You are a query expansion assistant for a Machine Learning knowledge base. "
                    "Rewrite the user's question into a more detailed and descriptive version "
                    "that will improve similarity search retrieval. "
                    "Add relevant ML context and keywords while keeping the same meaning. "
                    "For example: 'what is supervised learning' → "
                    "'Explain supervised learning: definition, how it works, "
                    "labeled data, training process, and examples.' "
                    "Output ONLY the expanded query, no explanation."
                )},
                {"role": "user", "content": query}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    def chat(self, user_query: str) -> tuple[str, str]:
        # Returns (response_text, agent_path) for display in UI

        # Step 1: Check for obnoxious content
        if self.obnoxious_agent.check_query(user_query):
            return ("I'm sorry, I can't respond to that. Please keep the conversation respectful.",
                    "Obnoxious_Agent → Refused")

        # Step 2: Handle small talk directly
        if self.is_small_talk(user_query):
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You are a friendly ML assistant. Respond briefly to greetings and small talk."},
                    {"role": "user",   "content": user_query}
                ],
                temperature=0.7
            )
            answer = response.choices[0].message.content.strip()
            self.conv_history.append({"role": "user",      "content": user_query})
            self.conv_history.append({"role": "assistant", "content": answer})
            return (answer, "Small_Talk_Handler → Responded")

        # Step 3: Rewrite query for multi-turn clarity
        rewritten_query = self.context_rewriter.rephrase(self.conv_history, user_query)

        # Step 4: Decompose hybrid queries — extract only the ML component
        ml_query = self.decompose_ml_query(rewritten_query)
        if ml_query is None:
            return ("I'm sorry, I can only answer questions related to Machine Learning.",
                    "Query_Agent → Refused (Irrelevant)")

        hybrid_detected = ml_query.strip().lower() != rewritten_query.strip().lower()
        if hybrid_detected:
            rewritten_query = ml_query
            hybrid_note = "Hybrid_Decomposer → ML extracted → "
        else:
            hybrid_note = ""

        # Step 4.5: Confirm the extracted query is ML-relevant
        if not self.query_agent.is_relevant(rewritten_query):
            return ("I'm sorry, I can only answer questions related to Machine Learning.",
                    "Query_Agent → Refused (Irrelevant)")

        # Step 5: Retrieve documents from Pinecone
        expanded_query = self.expand_query(rewritten_query)
        docs = self.query_agent.query_vector_store(expanded_query, k=5, namespace="ns2500")

        # Step 6: Verify document relevance
        if not self.relevant_docs_agent.get_relevance({"query": rewritten_query, "docs": docs}):
            return ("I couldn't find relevant information in my knowledge base to answer that.",
                    "Relevant_Documents_Agent → Refused (No relevant docs)")

        # Step 7: Generate final answer
        answer = self.answering_agent.generate_response(
            rewritten_query, docs, self.conv_history
        )

        self.conv_history.append({"role": "user",      "content": user_query})
        self.conv_history.append({"role": "assistant", "content": answer})
        return (answer, f"{hybrid_note}Answering_Agent → Responded")


# Initialize Head_Agent
@st.cache_resource
def load_head_agent():
    return Head_Agent(
        openai_key=openai_api_key,
        pinecone_key=pinecone_api_key,
        pinecone_index_name=PINECONE_INDEX_NAME
    )

head_agent = load_head_agent()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: conversation controls
with st.sidebar:
    st.header("Controls")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        head_agent.conv_history = []
        st.rerun()
    st.divider()
    st.markdown("**Agent Pipeline:**")
    st.markdown(
        "1. Obnoxious Check\n"
        "2. Small Talk Check\n"
        "3. Context Rewriter\n"
        "4. Hybrid Decomposer\n"
        "4.5. Relevance Check\n"
        "5. Pinecone Retrieval\n"
        "6. Doc Relevance Check\n"
        "7. Answer Generation"
    )

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "agent_path" in message:
            st.caption(f"Agent path: {message['agent_path']}")

# Accept user input
if prompt := st.chat_input("Ask me anything about Machine Learning..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_text, agent_path = head_agent.chat(prompt)
        st.markdown(response_text)
        st.caption(f"Agent path: {agent_path}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "agent_path": agent_path
    })

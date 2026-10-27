import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from duckduckgo_search import DDGS
import re
from scraper import extract_article_content

# ==============================================================================
# Page Setup & Styling
# ==============================================================================
st.set_page_config(
    page_title="Credence | The EchoShield",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Credence: The EchoShield")
st.caption("Evidence-Based Multi-Stage News Verification System")

# ==============================================================================
# Resource Loading
# ==============================================================================
@st.cache_resource(show_spinner="Loading fine-tuned models and encoders...")
def init_models():
    repo_id = "Shrey-2310/credence-distilbert-base"
    
    # 1. Base DistilBERT
    tok = AutoTokenizer.from_pretrained(repo_id)
    clf = AutoModelForSequenceClassification.from_pretrained(repo_id)
    clf.eval()

    # 2. Evidence Sentence Matcher & NLI Stance Checker
    embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    nli = CrossEncoder("cross-encoder/nli-deberta-v3-small")

    return tok, clf, embed, nli

try:
    tokenizer, model, embedder, nli_model = init_models()
    models_ready = True
except Exception as e:
    st.error(f"Error loading models from Hugging Face Hub: {e}")
    models_ready = False

# ==============================================================================
# Inference & Retrieval Logic
# ==============================================================================
def get_baseline_score(text: str) -> float:
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        suspicion_prob = float(probs[0][1].item())  # Index 1 = Suspicious / Fake
    return suspicion_prob

def extract_candidate_claims(text: str, max_claims: int = 2) -> list:
    raw_sentences = re.split(r'(?<=[.!?]) +', text.strip())
    clean_claims = []
    for s in raw_sentences:
        clean = s.strip().replace("\n", " ")
        if len(clean.split()) >= 6 and not clean.endswith("?"):
            clean_claims.append(clean)
        if len(clean_claims) >= max_claims:
            break
    return clean_claims if clean_claims else [text[:150]]

def fetch_live_evidence(query: str, max_results: int = 5) -> list:
    evidence = []
    search_prompt = f"{query} news fact check"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(search_prompt, max_results=max_results))
            for r in results:
                snippet = r.get("body", "").strip()
                if snippet:
                    evidence.append({
                        "title": r.get("title", "News Source"),
                        "snippet": snippet,
                        "url": r.get("href", "#")
                    })
    except Exception as e:
        st.warning(f"DuckDuckGo search notice: {e}")
    return evidence

def verify_claim(claim: str, evidence_pool: list):
    if not evidence_pool:
        return None

    # Step 1: Semantic ranking
    claim_emb = embedder.encode(claim, convert_to_tensor=True)
    snippets = [e["snippet"] for e in evidence_pool]
    snippet_embs = embedder.encode(snippets, convert_to_tensor=True)

    sims = util.cos_sim(claim_emb, snippet_embs)[0]
    best_idx = int(torch.argmax(sims).item())
    best_match = evidence_pool[best_idx]
    best_sim = float(sims[best_idx].item())

    # Step 2: Cross-Encoder NLI
    nli_logits = nli_model.predict([(claim, best_match["snippet"])])[0]
    nli_probs = torch.softmax(torch.tensor(nli_logits), dim=0).tolist()
    
    labels = ["contradiction", "entailment", "neutral"]
    pred_label = labels[int(torch.tensor(nli_probs).argmax())]

    return {
        "claim": claim,
        "best_evidence": best_match["snippet"],
        "source_title": best_match["title"],
        "source_url": best_match["url"],
        "similarity": best_sim,
        "stance": pred_label,
        "p_contradiction": nli_probs[0],
        "p_entailment": nli_probs[1],
        "p_neutral": nli_probs[2]
    }

# ==============================================================================
# UI Component Layout
# ==============================================================================
col_left, col_right = st.columns([1.1, 1])
article_to_analyze = ""

with col_left:
    st.subheader("Input Method")
    tab_url, tab_text = st.tabs(["🔗 Paste Article URL", "📝 Direct Raw Text"])

    with tab_url:
        url_input = st.text_input("Enter News URL:", placeholder="https://www.theguardian.com/...")
        if st.button("Fetch & Extract Article", use_container_width=True):
            if url_input.strip():
                with st.spinner("Scraping webpage and removing boilerplate..."):
                    res = extract_article_content(url_input.strip())
                    if res["success"] and len(res["text"]) > 40:
                        st.session_state["extracted_text"] = res["text"]
                        st.session_state["extracted_title"] = res["title"]
                        st.success(f"Extracted: **{res['title']}**")
                    else:
                        st.error(f"Failed to scrape: {res.get('error', 'Text empty or blocked.')}")
            else:
                st.warning("Please provide a valid URL.")

        if "extracted_text" in st.session_state:
            article_to_analyze = st.session_state["extracted_text"]
            with st.expander("📄 View Extracted Content", expanded=False):
                st.write(article_to_analyze)

    with tab_text:
        text_input = st.text_area(
            "Paste excerpt or claim:",
            height=200,
            placeholder="e.g. World bank confirms all interest rates cut to zero starting tomorrow..."
        )
        if text_input.strip():
            article_to_analyze = text_input.strip()

    run_btn = st.button("Run Multi-Stage Verification 🛡️", type="primary", use_container_width=True)

# ==============================================================================
# Execution & Explainability Output
# ==============================================================================
if run_btn:
    if not models_ready:
        st.error("Models could not be initialized. Please check network/hub settings.")
    elif not article_to_analyze.strip():
        st.warning("Please enter text or fetch a URL first.")
    else:
        with st.spinner("Processing article through Credence pipeline..."):
            # 1. Base ML Suspicion
            base_suspicion = get_baseline_score(article_to_analyze)

            # 2. Claim Extraction
            extracted_claims = extract_candidate_claims(article_to_analyze)

            # 3. Evidence Match & Stance Verification
            results = []
            for claim in extracted_claims:
                evidence = fetch_live_evidence(claim)
                res = verify_claim(claim, evidence)
                if res:
                    results.append(res)

            # 4. Meta Fusion
            max_contra = max([r["p_contradiction"] for r in results], default=0.0)
            max_entail = max([r["p_entailment"] for r in results], default=0.0)

            final_suspicion = (0.35 * base_suspicion) + (0.50 * max_contra) + (0.15 * (1.0 - max_entail))
            final_suspicion = min(max(final_suspicion, 0.0), 1.0)

        with col_right:
            st.subheader("Verification Verdict")
            
            c1, c2 = st.columns(2)
            c1.metric("DistilBERT Suspicion", f"{base_suspicion * 100:.1f}%")
            c2.metric("Credibility Score", f"{(1.0 - final_suspicion) * 100:.1f}%")

            if final_suspicion >= 0.65:
                st.error(f"⚠️ **High Probability of Deception ({final_suspicion * 100:.1f}%)**")
            elif final_suspicion >= 0.40:
                st.warning(f"🟡 **Uncertain / Mixed Signals ({final_suspicion * 100:.1f}%)**")
            else:
                st.success(f"✅ **Likely Credible ({(1.0 - final_suspicion) * 100:.1f}%)**")

            st.progress(final_suspicion)

        st.divider()
        st.subheader("🔎 Explainability & Evidence Trace")

        if not results:
            st.info("No verifiable factual statements matched against external sources.")
        else:
            for i, item in enumerate(results):
                with st.container(border=True):
                    st.markdown(f"**Extracted Claim {i+1}:** *\"{item['claim']}\"*")
                    
                    badge = {
                        "contradiction": "❌ Contradicted by Evidence",
                        "entailment": "✅ Supported by Evidence",
                        "neutral": "⚖️ Inconclusive / Neutral"
                    }.get(item["stance"], "Unknown")
                    
                    top_conf = max(item['p_contradiction'], item['p_entailment'], item['p_neutral']) * 100
                    st.markdown(f"**Stance:** `{badge}` (Confidence: {top_conf:.1f}%)")
                    st.markdown(f"**Top Live Evidence:** \"{item['best_evidence']}\"")
                    st.markdown(f"**Source Document:** [{item['source_title']}]({item['source_url']})")
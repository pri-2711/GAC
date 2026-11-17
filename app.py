"""Gujarati Accent Classifier – Streamlit frontend.

Run from the project root:   streamlit run app.py

The app is a thin interface. Audio is standardised and classified by the project's real pipeline
(src/model_dev.AccentPredictor with the final Phase 6 MLP in results/model_development/final_model/bundle/);
nothing about feature extraction, normalisation or the model is re-implemented here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import altair as alt  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import frontend_utils as fu  # noqa: E402
import plot_utils as pu  # noqa: E402

st.set_page_config(page_title="Gujarati Accent Classifier", page_icon=":material/mic:", layout="wide", initial_sidebar_state="expanded")

NAVY, TEAL, SAND, INK, MUTED = "#14213D", "#0F766E", "#FFF7ED", "#1F2937", "#6B7280"
MIME = {"WAV": "audio/wav", "MP3": "audio/mpeg", "M4A": "audio/mp4", "OGG": "audio/ogg"}

CSS = f"""
<style>
html, body, [class*="css"] {{ font-family: "Segoe UI", "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif; color: {INK}; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; color: {NAVY}; }}
.hero {{ background: linear-gradient(120deg, {NAVY} 0%, #1D3557 55%, {TEAL} 130%); color: #fff; border-radius: 20px; padding: 2.2rem 2.4rem; margin-bottom: 1.4rem; box-shadow: 0 10px 30px rgba(20,33,61,.22); }}
.hero h1 {{ color: #fff; font-size: 2.35rem; margin: 0 0 .35rem 0; }}
.hero p {{ color: #E2E8F0; font-size: 1.06rem; margin: 0; max-width: 780px; line-height: 1.55; }}
.badge {{ display: inline-block; background: rgba(255,255,255,.16); color: #fff; border: 1px solid rgba(255,255,255,.3); border-radius: 999px; padding: .18rem .75rem; font-size: .78rem; margin: 0 .4rem .7rem 0; letter-spacing: .03em; }}
.card {{ background: #fff; border: 1px solid #E5E7EB; border-radius: 16px; padding: 1.2rem 1.4rem; box-shadow: 0 2px 10px rgba(15,23,42,.05); margin-bottom: 1rem; }}
.card h3 {{ margin-top: 0; font-size: 1.12rem; }}
.card p, .card li {{ line-height: 1.55; }}
.accent-card {{ border-radius: 16px; padding: 1.05rem 1.2rem; background: #fff; border: 1px solid #E5E7EB; border-top: 6px solid var(--c); box-shadow: 0 2px 10px rgba(15,23,42,.05); height: 100%; }}
.accent-card .name {{ font-size: 1.2rem; font-weight: 700; color: {NAVY}; }}
.accent-card .region {{ color: {MUTED}; font-size: .92rem; margin-top: .1rem; }}
.result {{ border-radius: 20px; padding: 1.6rem 1.8rem; background: linear-gradient(135deg, #fff 0%, {SAND} 100%); border: 1px solid #FED7AA; border-left: 10px solid var(--c); box-shadow: 0 8px 24px rgba(15,23,42,.08); margin: .8rem 0 1rem 0; }}
.result .label {{ text-transform: uppercase; letter-spacing: .12em; font-size: .74rem; color: {MUTED}; font-weight: 600; }}
.result .accent {{ font-size: 2.6rem; font-weight: 800; color: var(--c); line-height: 1.1; margin: .15rem 0 .3rem 0; }}
.result .region {{ font-size: 1.2rem; color: {NAVY}; font-weight: 600; }}
.result .score {{ color: {MUTED}; font-size: .93rem; margin-top: .55rem; }}
.warn {{ background: #FEF3C7; border: 1px solid #F59E0B; border-left: 6px solid #D97706; color: #78350F; border-radius: 12px; padding: .85rem 1.1rem; font-size: .95rem; line-height: 1.5; margin: .6rem 0 1rem 0; }}
.chip {{ display: inline-block; background: #F1F5F9; border: 1px solid #E2E8F0; color: {NAVY}; border-radius: 999px; padding: .22rem .8rem; font-size: .84rem; margin: 0 .4rem .4rem 0; }}
.stat {{ text-align: center; background: #fff; border: 1px solid #E5E7EB; border-radius: 16px; padding: 1rem .6rem; box-shadow: 0 2px 10px rgba(15,23,42,.05); }}
.stat .num {{ font-size: 2.1rem; font-weight: 800; color: {TEAL}; line-height: 1.1; }}
.stat .cap {{ color: {MUTED}; font-size: .88rem; }}
.step {{ display: flex; gap: .8rem; align-items: flex-start; margin-bottom: .55rem; }}
.step .n {{ min-width: 1.9rem; height: 1.9rem; border-radius: 50%; background: {TEAL}; color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: .9rem; }}
.footer {{ color: {MUTED}; font-size: .82rem; text-align: center; margin-top: 2.4rem; border-top: 1px solid #E5E7EB; padding-top: 1rem; }}
section[data-testid="stSidebar"] {{ background: #F8FAFC; border-right: 1px solid #E5E7EB; }}
/* sidebar tab-style navigation: inactive tab = plain row, active tab = tinted row with a teal left bar */
section[data-testid="stSidebar"] div.stButton > button {{ justify-content: flex-start; text-align: left; background: transparent; color: {INK}; border: none; border-left: 4px solid transparent; border-radius: 10px; padding: .6rem .9rem; font-weight: 500; font-size: 1rem; box-shadow: none; }}
section[data-testid="stSidebar"] div.stButton > button > div {{ justify-content: flex-start; }}
section[data-testid="stSidebar"] div.stButton > button:hover {{ background: #F0FDFA; color: {TEAL}; border-left-color: #99F6E4; }}
section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {{ background: #F0FDFA; color: {TEAL}; border-left: 4px solid {TEAL}; font-weight: 700; padding: .6rem .9rem; font-size: 1rem; }}
section[data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover {{ background: #CCFBF1; color: #115E59; }}
div[data-testid="stFileUploader"] section {{ border: 2px dashed {TEAL}; border-radius: 16px; background: #F0FDFA; }}
div.stButton > button[kind="primary"] {{ background: {TEAL}; border: none; border-radius: 12px; padding: .7rem 1.6rem; font-weight: 700; font-size: 1.05rem; }}
div.stButton > button[kind="primary"]:hover {{ background: #115E59; }}
@media (max-width: 800px) {{ .hero {{ padding: 1.4rem 1.2rem; }} .hero h1 {{ font-size: 1.7rem; }} .result .accent {{ font-size: 2rem; }} }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def html(s: str) -> None:
    st.markdown(" ".join(line.strip() for line in s.strip().splitlines()), unsafe_allow_html=True)


WARNING_TEXT = ("<b>Experimental prototype – not validated on unseen speakers.</b> The model was trained on only 35 recordings from 5 speakers "
                "(one speaker per accent). Its output reflects similarity to those five voices and recording setups; it does <b>not</b> show "
                "that a new speaker’s accent has been recognised.")


@st.cache_resource(show_spinner="Loading the trained model…")
def get_predictor():
    return fu.load_predictor()


try:
    predictor = get_predictor()
    facts = fu.model_facts(predictor)
except Exception as exc:                                              # model bundle missing / broken
    st.error(f"The trained model could not be loaded from `results/model_development/final_model/bundle/`. Details: {exc}")
    st.stop()

COLORS = pu.accent_colors(facts["accents"])
DATA = fu.dataset_facts()

# --------------------------------------------------------------------------- sidebar
PAGES = {"Home": ":material/home:", "Classify audio": ":material/headphones:", "About the model": ":material/psychology:", "Dataset & limitations": ":material/bar_chart:"}
st.session_state.setdefault("page", "Home")


def _go(name: str) -> None:
    st.session_state["page"] = name


with st.sidebar:
    st.markdown("### :material/mic: Gujarati Accent Classifier")
    for _name, _icon in PAGES.items():
        st.button(_name, key=f"nav_{_name}", icon=_icon, use_container_width=True,
                  type="primary" if st.session_state["page"] == _name else "secondary", on_click=_go, args=(_name,))
    page = st.session_state["page"]
    st.markdown("---")
    st.caption("**Model**")
    st.caption(f"Final MLP · {facts['n_features']} acoustic features · {facts['n_trainable']:,} trainable parameters · trained from scratch")
    st.caption("**Supported files**: " + ", ".join(e.upper().lstrip(".") for e in fu.SUPPORTED_EXTENSIONS))
    html(f'<div class="warn" style="font-size:.82rem;padding:.6rem .8rem;">△ Experimental prototype. Not validated on unseen speakers.</div>')


def hero(title: str, subtitle: str, badges: list[str]) -> None:
    html(f'<div class="hero">{"".join(f"<span class=badge>{b}</span>" for b in badges)}<h1>{title}</h1><p>{subtitle}</p></div>')


# --------------------------------------------------------------------------- pages
def page_home() -> None:
    hero("Gujarati Accent Classifier",
         "A research prototype that listens to a short Gujarati speech recording and estimates which of five Gujarati accents "
         "it is most similar to, together with the accent’s region of Gujarat. It was built from scratch as a university NLP "
         "research project: audio standardisation, acoustic feature extraction, analysis and a small neural network.",
         ["NLP research project", "Speech · MFCC features", "Trained from scratch"])
    c1, c2, c3 = st.columns(3)
    for col, num, cap in ((c1, DATA["n_recordings"], "recordings in the dataset"), (c2, DATA["n_accents"], "Gujarati accents"), (c3, DATA["n_speakers"], "speakers (one per accent)")):
        col.markdown(f'<div class="stat"><div class="num">{num}</div><div class="cap">{cap}</div></div>', unsafe_allow_html=True)
    st.markdown("### Supported accents and their regions")
    cols = st.columns(len(facts["accents"]))
    for col, a in zip(cols, facts["accents"]):
        col.markdown(f'<div class="accent-card" style="--c:{COLORS[a]}"><div class="name">{a}</div><div class="region">⌖ {facts["accent_to_region"][a]}</div></div>', unsafe_allow_html=True)
    st.markdown("")
    left, right = st.columns([3, 2])
    with left:
        html("""<div class="card"><h3>How it works</h3>
        <div class="step"><div class="n">1</div><div><b>Upload</b> a speech recording (WAV, MP3, M4A or OGG).</div></div>
        <div class="step"><div class="n">2</div><div>The audio is <b>standardised</b> (16 kHz, mono) exactly as in the project’s data preparation.</div></div>
        <div class="step"><div class="n">3</div><div><b>78 acoustic features</b> (MFCC summaries) are extracted and normalised with the training statistics.</div></div>
        <div class="step"><div class="n">4</div><div>A small neural network predicts the <b>region</b> and the <b>accent</b>; the accent is chosen among the accents of the predicted region.</div></div></div>""")
    with right:
        html("""<div class="card"><h3>Please read</h3><p>This is an <b>initial exploratory study</b>. Every accent in the dataset
        is represented by <b>one speaker</b>, so the model cannot tell an accent from an individual voice. Treat results as a demonstration of the
        pipeline, not as an accent test.</p></div>""")
    html(f'<div class="warn">{WARNING_TEXT}</div>')
    st.info("Open the :material/headphones: **Classify audio** tab in the sidebar to try it.")


def waveform_figure(y: np.ndarray, sr: int):
    n = len(y)
    t = np.arange(n) / sr
    bins = min(2400, n)
    edges = np.linspace(0, n, bins + 1, dtype=int)
    mins = np.array([y[a:b].min() for a, b in zip(edges[:-1], edges[1:])])
    maxs = np.array([y[a:b].max() for a, b in zip(edges[:-1], edges[1:])])
    ts = (edges[:-1] + edges[1:]) / 2 / sr
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.fill_between(ts, mins, maxs, color=TEAL, linewidth=0, alpha=.9)
    ax.axhline(0, color="#94A3B8", lw=.6)
    ax.set(xlim=(0, t[-1] if n else 1), ylim=(-1.02, 1.02), xlabel="Time (s)", ylabel="Amplitude")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


def prob_chart(names: list[str], probs: list[float], regions: dict[str, str] | None, title: str) -> alt.Chart:
    df = pd.DataFrame({"Class": names, "Score": probs, "Region": [regions.get(n, "") if regions else "" for n in names]})
    df["Label"] = df.Score.map(lambda v: f"{v:.1%}")
    colors = {**COLORS}
    base = alt.Chart(df, title=title).encode(y=alt.Y("Class:N", sort="-x", title=None, axis=alt.Axis(labelFontSize=13)))
    bars = base.mark_bar(cornerRadiusEnd=6, size=26).encode(
        x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title="model output score (softmax)")),
        color=alt.Color("Class:N", scale=alt.Scale(domain=list(colors), range=list(colors.values())), legend=None),
        tooltip=["Class", alt.Tooltip("Score:Q", format=".4f"), "Region"])
    text = base.mark_text(align="left", dx=6, fontSize=12, color=INK).encode(x="Score:Q", text="Label:N")
    return (bars + text).properties(height=34 * len(df) + 40)


def show_result(res: dict) -> None:
    c = COLORS.get(res["accent"], TEAL)
    html(f"""<div class="result" style="--c:{c}"><div class="label">Predicted accent</div><div class="accent">{res['accent']}</div>
    <div class="region">⌖ {res['region']}</div>
    <div class="score">Model output score for this accent: <b>{res['accent_head_probability']:.1%}</b> · for this region: <b>{res['region_confidence']:.1%}</b>
    (softmax scores of a small network – <u>not</u> calibrated confidence)</div></div>""")
    html(f'<div class="warn">{WARNING_TEXT}</div>')
    acc_names = list(res["accent_probabilities"])
    ch1, ch2 = st.columns(2)
    with ch1:
        st.altair_chart(prob_chart(acc_names, [res["accent_probabilities"][a] for a in acc_names], None, "Accent head – output scores"), use_container_width=True)
    with ch2:
        reg_names = list(res["region_probabilities"])
        reg_to_color = {r: COLORS[facts["region_to_accents"][r][0]] for r in reg_names}
        df = pd.DataFrame({"Class": reg_names, "Score": [res["region_probabilities"][r] for r in reg_names]})
        df["Label"] = df.Score.map(lambda v: f"{v:.1%}")
        base = alt.Chart(df, title="Region head – output scores").encode(y=alt.Y("Class:N", sort="-x", title=None, axis=alt.Axis(labelFontSize=13)))
        bars = base.mark_bar(cornerRadiusEnd=6, size=26).encode(x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title="model output score (softmax)")),
                                                                color=alt.Color("Class:N", scale=alt.Scale(domain=list(reg_to_color), range=list(reg_to_color.values())), legend=None))
        text = base.mark_text(align="left", dx=6, fontSize=12, color=INK).encode(x="Score:Q", text="Label:N")
        st.altair_chart((bars + text).properties(height=34 * len(df) + 40), use_container_width=True)
    st.caption("These bars are the raw outputs of the two softmax heads. They add up to 100% by construction and tend to be close to 0% or 100% because the network "
               "was trained on only five voices; they are **not** calibrated probabilities and should not be read as the chance of being correct.")
    with st.expander("How the final decision was made (hierarchical decoding)"):
        st.markdown(
            f"1. **Region** = highest-scoring region: **{res['region']}** (score {res['region_confidence']:.1%}).\n"
            f"2. The accents belonging to that region were considered: **{', '.join(facts['region_to_accents'][res['region']])}**.\n"
            f"3. **Accent** = highest-scoring accent among them: **{res['accent']}** (score within the region {res['accent_confidence_within_region']:.1%}; "
            f"raw accent-head score {res['accent_head_probability']:.1%}).\n\n"
            f"The accent head on its own would have chosen **{res['accent_head_argmax']}**. "
            "Because each region currently contains exactly one accent, step 3 is trivial; the app cannot yet distinguish several accents inside one region.")
    with st.expander("Raw model output"):
        st.json({k: v for k, v in res.items()})


def page_classify() -> None:
    hero("Classify a recording", "Upload a short recording of Gujarati speech. The model returns the most similar accent and its region.",
         ["WAV · MP3 · M4A · OGG", "1 – 180 s", "best: 8–18 s of clear speech"])
    html(f'<div class="warn">{WARNING_TEXT}</div>')
    up = st.file_uploader("Drag and drop a recording here, or browse", type=[e.lstrip(".") for e in fu.SUPPORTED_EXTENSIONS], accept_multiple_files=False,
                          help=f"WAV, MP3, M4A or OGG, up to {fu.MAX_UPLOAD_MB} MB.")
    if up is None:
        st.markdown("")
        html("""<div class="card"><h3>Tips for a meaningful test</h3><ul>
        <li>Use <b>8–18 seconds</b> of natural speech (the training recordings are this long).</li>
        <li>One speaker, little background noise, no music.</li>
        <li>Remember that the model only knows five voices – see <i>Dataset &amp; limitations</i>.</li></ul></div>""")
        return
    data = up.getvalue()
    key = f"{up.name}:{fu.digest_of(data)}"
    if st.session_state.get("prepared_key") != key:
        st.session_state.pop("prepared", None)
        st.session_state["prepared_key"] = key
        st.session_state["prepare_error"] = None
        with st.spinner("Reading and standardising the audio…"):
            try:
                st.session_state["prepared"] = fu.prepare_upload(up.name, data, predictor)
            except fu.AudioProblem as exc:
                st.session_state["prepare_error"] = str(exc)
            except Exception as exc:                                    # never show a stack trace to the user
                st.session_state["prepare_error"] = f"Unexpected problem while reading the file: {exc}"
    if st.session_state.get("prepare_error"):
        st.error(st.session_state["prepare_error"])
        return
    prep: fu.PreparedAudio = st.session_state["prepared"]

    st.markdown("#### 1 · Your recording")
    st.audio(data, format=MIME.get(prep.original_format, "audio/wav"))
    sr_txt = f"{prep.original_sample_rate / 1000:g} kHz" if prep.original_sample_rate else "unknown rate"
    ch = prep.original_channels
    ch_txt = {1: "mono", 2: "stereo"}.get(ch, f"{ch} channels") if ch is not None else ""
    html(f'<span class="chip">▢ {prep.name}</span><span class="chip">{prep.original_format} · {sr_txt} · {ch_txt}</span>'
         f'<span class="chip">◷ {prep.duration_s:.1f} s</span><span class="chip">→ standardised to 16 kHz mono</span>'
         f'<span class="chip">level {prep.rms_dbfs:.0f} dBFS RMS</span>')
    for w in prep.warnings:
        st.warning(w)
    st.pyplot(waveform_figure(prep.signal, prep.sample_rate), clear_figure=True, use_container_width=True)

    st.markdown("#### 2 · Classify")
    store = st.session_state.setdefault("results", {})
    if st.button("Classify accent", icon=":material/search:", type="primary"):
        with st.spinner("Analysing the recording with the trained model…"):
            try:
                store[key] = fu.classify(prep, predictor)
            except Exception as exc:
                st.error(f"The model could not process this recording: {exc}")
    if key in store:
        st.markdown("#### 3 · Result")
        show_result(store[key])


def page_model() -> None:
    hero("About the model", "A small neural network (a multilayer perceptron) that reads 78 numbers summarising how a recording sounds.",
         ["Multilayer perceptron", "78 MFCC features", "Trained from scratch"])
    a, b = st.columns([3, 2])
    with a:
        html("""<div class="card"><h3>The 78 numerical acoustic features</h3>
        <p>Speech is cut into 25 ms frames every 10 ms. For each frame 13 <b>MFCCs</b> (Mel-frequency cepstral coefficients) describe the
        shape of the sound spectrum – roughly, how the vocal tract is shaping the sound. The recording is then summarised by the <b>mean</b> and the
        <b>standard deviation</b> over all frames:</p>
        <ul><li><b>13 MFCCs</b> × (mean, std) = 26 numbers – the average spectral shape and how much it varies</li>
        <li><b>13 delta-MFCCs</b> × (mean, std) = 26 numbers – how fast the spectral shape changes</li>
        <li><b>13 delta-delta-MFCCs</b> × (mean, std) = 26 numbers – how that change itself changes</li></ul>
        <p>26 + 26 + 26 = <b>78 features</b> per recording, computed by the project’s Phase 3 code. Before prediction each feature is standardised with the
        mean and spread of the training recordings (saved with the model).</p></div>""")
        html(f"""<div class="card"><h3>The MLP and its two prediction heads</h3>
        <p>The final model is a small <b>multilayer perceptron</b> (a stack of fully connected layers) with {facts['n_trainable']:,} trainable parameters:</p>
        <ul><li>78 features → <b>Dense 64</b> (ReLU) → batch normalisation → dropout 0.30</li>
        <li>→ <b>Dense 32</b> (ReLU) → dropout 0.20</li>
        <li>→ shared <b>Dense 64</b> (ReLU) → dropout 0.40</li>
        <li>→ two <b>softmax heads</b>: a <b>region head</b> ({len(facts['regions'])} regions) and an <b>accent head</b> ({len(facts['accents'])} accents).</li></ul>
        <p><b>Hierarchical decision:</b> the region is chosen first; the accent is then chosen only among the accents that belong to that region. Today each
        region contains exactly one accent, so the two heads are equivalent and the app cannot yet separate several accents <i>within</i> a region.</p>
        <p>The model was <b>trained entirely from scratch</b> on the project’s own recordings – no pretrained model, weights or embeddings are used.
        The deployed model was fitted on {facts['n_fit']} recordings, with {facts['n_validation']} more used to pick the best training epoch.</p></div>""")
    with b:
        comp = facts["comparison"]
        score = ""
        if comp is not None and "mlp_only" in comp.index:
            score = (f"In the project’s 7-fold cross-validation this configuration classified <b>{comp.loc['mlp_only', 'pooled_accent_accuracy']:.0%}</b> of held-out "
                     "recordings correctly – but held-out recordings came from the <i>same five speakers</i>.")
        html(f"""<div class="card"><h3>What the scores mean</h3><p>{score}</p>
        <p>That is a <b>within-speaker</b> result. It says the pipeline works mechanically; it says nothing about speakers the model has never heard.</p></div>""")
        html("""<div class="card"><h3>Processing pipeline</h3>
        <div class="step"><div class="n">1</div><div>Decode and convert with FFmpeg to 16 kHz mono (no filtering or trimming)</div></div>
        <div class="step"><div class="n">2</div><div>Extract MFCC, delta and delta-delta per frame; take mean and std</div></div>
        <div class="step"><div class="n">3</div><div>Standardise with the saved training statistics</div></div>
        <div class="step"><div class="n">4</div><div>MLP → region scores and accent scores → hierarchical decision</div></div></div>""")


def page_dataset() -> None:
    hero("Dataset and limitations", "What the model learned from – and why that limits what it can claim.", ["Initial five-accent exploratory study"])
    c1, c2, c3, c4 = st.columns(4)
    for col, num, cap in ((c1, DATA["n_recordings"], "recordings"), (c2, DATA["n_accents"], "accents"), (c3, DATA["n_speakers"], "speakers"), (c4, DATA["per_speaker"], "recordings per speaker")):
        col.markdown(f'<div class="stat"><div class="num">{num}</div><div class="cap">{cap}</div></div>', unsafe_allow_html=True)
    st.markdown("")
    per = DATA["per_accent"].rename(columns={"region": "Region", "accent": "Accent", "recordings": "Recordings", "speakers": "Speakers", "format": "Original format"})
    st.dataframe(per, hide_index=True, use_container_width=True)
    html(f'<div class="warn">{WARNING_TEXT}</div>')
    html(f"""<div class="card"><h3>The key limitation: one speaker per accent</h3>
    <p>Each of the {DATA['n_accents']} accents is represented by <b>a single speaker</b> ({DATA['per_speaker']} recordings each). Accent, speaker identity, microphone, room and even
    the spoken wording are therefore completely mixed together. When the model recognises a recording it may be recognising <i>that person’s voice or recording setup</i>
    rather than a regional accent.</p>
    <ul><li><b>No speaker-independent evaluation is possible</b> with this data: held-out test recordings come from the same people as the training recordings.</li>
    <li>Near-perfect scores in the project’s cross-validation are therefore <b>expected</b> and are <b>not</b> evidence that the accents will be recognised in new speakers.</li>
    <li>A new speaker will always be assigned to the <b>most similar of the five known voices</b>, even if their accent is none of these five.</li>
    <li>Each region contains one accent, so distinguishing several accents <b>within</b> a region has <b>not</b> been demonstrated.</li>
    <li><b>Speaker gender and voice characteristics:</b> differences in speaker gender and gender-associated vocal characteristics, such as pitch and voice quality, could potentially
    influence the acoustic features and lead to biased accent predictions. <b>Gender-independent performance has not been established</b>, because the dataset does not have enough speaker
    diversity (one speaker per accent). This is a potential risk that has not been tested; it is <b>not</b> an experimentally demonstrated bias.</li>
    <li>Recording conditions differ between the five groups (format, sample rate, noise level); the model may use them as shortcuts.</li></ul></div>""")
    html("""<div class="card"><h3>Future scope</h3><ul>
    <li>More speakers per accent and more recordings per speaker; same-device, quiet recordings.</li>
    <li>More regions and accent varieties within Gujarat; a larger, balanced dataset.</li>
    <li>Speaker-independent evaluation (train and test on different speakers) and a more generalisable classifier.</li></ul></div>""")


{"Home": page_home, "Classify audio": page_classify, "About the model": page_model, "Dataset & limitations": page_dataset}[page]()
st.markdown('<div class="footer">Gujarati Accent Classifier · research prototype (Phases 1–7) · experimental, not validated on unseen speakers</div>', unsafe_allow_html=True)

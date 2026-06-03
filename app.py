"""
NZ Native Fish Presence Predictor — Multi-Species Streamlit App
================================================================
Features:
  - 10 native fish species with visual card selector
  - Region-based auto-fill environmental parameters
  - Multi-model comparison (kNN, RF, XGBoost, Ensemble)
  - Interactive NZ sampling site map
  - Model performance dashboard
  - Probability gauge + interpretation
"""
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(
    page_title="NZ Native Fish Predictor",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': 'ML-powered NZ native fish habitat prediction — Massey University 158.755'
    }
)

# Block Streamlit 'c' key clear-cache — must run BEFORE any Streamlit component
st.markdown("""
<script>
(function() {
  // Intercept Streamlit's CLEAR_CACHE custom event at the dispatch level
  var _origDispatch = EventTarget.prototype.dispatchEvent;
  EventTarget.prototype.dispatchEvent = function(event) {
    if (event && event.type === 'CLEAR_CACHE') {
      return true;
    }
    return _origDispatch.call(this, event);
  };

  // Backup: block bare 'c' key at keyboard level
  function blockC(e) {
    if ((e.key === 'c' || e.key === 'C' || e.code === 'KeyC') &&
        !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
      var tag = (e.target && e.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea' || (e.target && e.target.isContentEditable)) return;
      e.stopImmediatePropagation();
      e.preventDefault();
      return false;
    }
  }
  document.addEventListener('keydown', blockC, {capture: true, passive: false});
  window.addEventListener('keydown', blockC, {capture: true, passive: false});

  // Hide toolbar hint
  setInterval(function() {
    var el = document.querySelector('[data-testid="stStatusWidget"]');
    if (el) el.style.display = 'none';
  }, 400);
})();
</script>
""", unsafe_allow_html=True)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

# ============================================================
# CSS Variables + Full Theme
# ============================================================
st.markdown("""<style>
:root {
  --bg-deep: #050510;
  --bg-card: rgba(12, 12, 30, 0.75);
  --accent-cyan: #00D4FF;
  --accent-purple: #7B2FFF;
  --accent-glow-cyan: rgba(0, 212, 255, 0.25);
  --accent-glow-purple: rgba(123, 47, 255, 0.2);
  --danger-red: #FF4466;
  --warning-amber: #FFB347;
  --text-primary: #E8E8F0;
  --text-secondary: #8888AA;
  --border-subtle: rgba(255, 255, 255, 0.06);
  --border-glow: rgba(0, 212, 255, 0.35);
  --glass-blur: blur(16px);
}

/* === Cyber background: animated grid + drifting orbs + scan lines === */
.stApp {
  background: #050510;
}

/* Layer 1 — animated grid with glow pulse */
.stApp::before {
  content: '';
  position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  background-image:
    linear-gradient(rgba(0, 212, 255, 0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 212, 255, 0.04) 1px, transparent 1px);
  background-size: 80px 80px;
  pointer-events: none; z-index: 0;
  animation: gridDrift 30s linear infinite, gridGlow 4s ease-in-out infinite;
}
@keyframes gridDrift {
  0%   { transform: translate(0, 0); }
  100% { transform: translate(80px, 80px); }
}
@keyframes gridGlow {
  0%, 100% { opacity: 0.5; }
  50%      { opacity: 1; }
}

/* Layer 2 — drifting orbs (CSS-only, no extra DOM) */
.stApp::after {
  content: '';
  position: fixed; top: -30%; left: -15%; right: -15%; bottom: -30%;
  background:
    /* Orb 1 — cyan, drifts right */
    radial-gradient(ellipse 600px 400px at 25% 30%, rgba(0, 212, 255, 0.08) 0%, transparent 70%),
    /* Orb 2 — purple, drifts left */
    radial-gradient(ellipse 500px 350px at 70% 60%, rgba(123, 47, 255, 0.07) 0%, transparent 65%),
    /* Orb 3 — cyan, top-right */
    radial-gradient(ellipse 450px 300px at 80% 15%, rgba(0, 212, 255, 0.05) 0%, transparent 60%),
    /* Orb 4 — purple, bottom-left */
    radial-gradient(ellipse 400px 250px at 15% 85%, rgba(123, 47, 255, 0.05) 0%, transparent 60%);
  pointer-events: none; z-index: 0;
  animation: orbsDrift 20s ease-in-out infinite, orbsPulse 6s ease-in-out infinite;
}
@keyframes orbsDrift {
  0%   { transform: translate(0, 0) rotate(0deg); }
  25%  { transform: translate(3%, -2%) rotate(1deg); }
  50%  { transform: translate(-1%, 3%) rotate(-0.5deg); }
  75%  { transform: translate(-3%, -1%) rotate(0.5deg); }
  100% { transform: translate(0, 0) rotate(0deg); }
}
@keyframes orbsPulse {
  0%, 100% { opacity: 0.5; }
  50%      { opacity: 0.9; }
}

/* Layer 3 — scanning beam (extra div needed, injected once) */
.scan-beam {
  position: fixed; top: -2px; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg,
    transparent 0%, rgba(0,212,255,0.15) 30%, rgba(123,47,255,0.3) 50%,
    rgba(0,212,255,0.15) 70%, transparent 100%);
  pointer-events: none; z-index: 0;
  animation: scanDown 8s linear infinite;
  box-shadow: 0 0 40px rgba(0,212,255,0.15), 0 0 80px rgba(123,47,255,0.08);
}
@keyframes scanDown {
  0%   { top: -4px; opacity: 0; }
  5%   { opacity: 0.8; }
  95%  { opacity: 0.8; }
  100% { top: 100vh; opacity: 0; }
}

/* Layer 4 — data particles */
.data-particles {
  position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  background-image:
    radial-gradient(1.5px 1.5px at 12% 22%, rgba(0,212,255,0.35), transparent),
    radial-gradient(1px 1px at 28% 45%, rgba(123,47,255,0.3), transparent),
    radial-gradient(1.5px 1.5px at 45% 15%, rgba(0,212,255,0.25), transparent),
    radial-gradient(1px 1px at 55% 72%, rgba(123,47,255,0.3), transparent),
    radial-gradient(1.5px 1.5px at 68% 38%, rgba(0,212,255,0.25), transparent),
    radial-gradient(1px 1px at 75% 85%, rgba(0,212,255,0.2), transparent),
    radial-gradient(1.5px 1.5px at 85% 55%, rgba(123,47,255,0.3), transparent),
    radial-gradient(1px 1px at 92% 12%, rgba(0,212,255,0.25), transparent),
    radial-gradient(1.5px 1.5px at 18% 65%, rgba(0,212,255,0.2), transparent),
    radial-gradient(1px 1px at 35% 90%, rgba(123,47,255,0.25), transparent);
  background-size: 200px 200px;
  pointer-events: none; z-index: 0;
  animation: particlesFloat 15s linear infinite;
}
@keyframes particlesFloat {
  0%   { transform: translateY(0); opacity: 0.6; }
  100% { transform: translateY(-200px); opacity: 1; }
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #07071A 0%, #0C0C22 50%, #08081C 100%) !important;
  border-right: 1px solid rgba(255, 255, 255, 0.04) !important;
}
[data-testid="stHeader"] { background: transparent !important; }

/* === Glassmorphism cards with glowing border === */
.glass-card {
  background: var(--bg-card);
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
  border: 1px solid var(--border-subtle);
  border-radius: 14px;
  padding: 0.9rem 0.6rem;
  text-align: center;
  transition: all 0.3s cubic-bezier(0.22, 0.61, 0.36, 1);
  position: relative;
  overflow: hidden;
  min-height: 72px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  box-sizing: border-box;
}
.glass-card::before {
  content: '';
  position: absolute; top: 0; left: 5%; right: 5%;
  height: 1.5px;
  background: linear-gradient(90deg, transparent, var(--accent-cyan), var(--accent-purple), transparent);
  opacity: 0.7;
}
.glass-card:hover {
  transform: translateY(-4px);
  border-color: var(--border-glow);
  box-shadow: 0 0 30px var(--accent-glow-cyan), 0 8px 30px rgba(0,0,0,0.3);
}
.glass-card .stat-value {
  font-size: 1.6rem; font-weight: 800;
  background: linear-gradient(135deg, #00D4FF, #7B2FFF);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1.15;
}
.glass-card .stat-label {
  font-size: 0.68rem; color: var(--text-secondary);
  margin-top: 0.15rem;
  line-height: 1.1;
  letter-spacing: 0.5px;
}

/* === Gauge meter === */
.gauge-container {
  background: var(--bg-card);
  backdrop-filter: var(--glass-blur);
  border: 1px solid var(--border-subtle);
  border-radius: 18px;
  padding: 1.5rem 2rem;
  text-align: center;
  margin: 0.5rem 0;
}
.gauge-track {
  height: 20px;
  border-radius: 12px;
  background: linear-gradient(90deg, #FF4466 0%, #FFB347 25%, #FFE44D 50%, #00D4FF 75%, #00FF88 100%);
  position: relative;
  margin: 1rem 0 0.3rem 0;
  box-shadow: inset 0 2px 8px rgba(0,0,0,0.5), 0 0 20px rgba(0,212,255,0.1);
}
.gauge-fill {
  height: 20px;
  border-radius: 12px;
  background: rgba(5, 5, 16, 0.88);
  position: absolute;
  right: 0; top: 0;
  transition: width 0.7s cubic-bezier(0.22, 0.61, 0.36, 1);
}
.gauge-needle {
  position: absolute;
  top: -8px;
  width: 4px;
  height: 36px;
  border-radius: 2px;
  background: #FFFFFF;
  box-shadow: 0 0 16px rgba(255,255,255,0.7), 0 0 4px rgba(0,212,255,0.5);
  transition: left 0.7s cubic-bezier(0.22, 0.61, 0.36, 1);
  z-index: 2;
}
.gauge-labels {
  display: flex; justify-content: space-between;
  font-size: 0.62rem; color: var(--text-secondary);
  margin-top: 0.3rem;
  letter-spacing: 0.3px;
}
.gauge-value {
  font-size: 3rem; font-weight: 900;
  background: linear-gradient(135deg, #00D4FF 0%, #7B2FFF 70%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0.3rem 0;
  filter: drop-shadow(0 0 12px rgba(0, 212, 255, 0.4));
}

/* === Sidebar selectbox accent === */
div[data-testid="stSidebar"] div[data-testid="stSelectbox"] {
  border: none !important;
}

/* === Form === */
div[data-testid="stForm"] {
  background: var(--bg-card);
  backdrop-filter: var(--glass-blur);
  border: 1px solid var(--border-subtle) !important;
  border-radius: 18px !important;
  padding: 1.2rem !important;
}
div[data-testid="stFormSubmitButton"] button {
  background: linear-gradient(135deg, #00D4FF, #7B2FFF) !important;
  border: none !important;
  color: #FFFFFF !important;
  font-weight: 700 !important;
  font-size: 1rem !important;
  padding: 0.65rem 2rem !important;
  border-radius: 10px !important;
  transition: all 0.3s !important;
  letter-spacing: 0.5px;
  box-shadow: 0 0 20px rgba(0, 212, 255, 0.25);
}
div[data-testid="stFormSubmitButton"] button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 0 35px rgba(0, 212, 255, 0.4), 0 0 20px rgba(123, 47, 255, 0.3) !important;
}

/* === Tabs === */
div[data-testid="stTabs"] { margin-top: 0.5rem; }
button[data-baseweb="tab"] {
  font-size: 0.85rem !important;
  font-weight: 600 !important;
  padding: 0.5rem 1.2rem !important;
  border-radius: 8px 8px 0 0 !important;
  transition: all 0.25s;
  color: #8888AA !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  background: rgba(0, 212, 255, 0.08) !important;
  border-bottom: 3px solid var(--accent-cyan) !important;
  color: #00D4FF !important;
}

/* === Loading fish === */
.fish-loader {
  display: flex; align-items: center; justify-content: center;
  gap: 8px; padding: 2rem;
}
.fish-loader .fish-dot {
  font-size: 1.6rem;
  animation: fishBounce 0.6s ease-in-out infinite;
}
.fish-loader .fish-dot:nth-child(2) { animation-delay: 0.1s; }
.fish-loader .fish-dot:nth-child(3) { animation-delay: 0.2s; }
.fish-loader .fish-dot:nth-child(4) { animation-delay: 0.3s; }
.fish-loader .fish-dot:nth-child(5) { animation-delay: 0.4s; }
@keyframes fishBounce {
  0%, 100% { transform: translateY(0) scale(1); }
  50% { transform: translateY(-16px) scale(1.2); }
}

/* === Widgets === */
div[data-testid="stSlider"] label {
  color: var(--text-primary) !important;
  font-weight: 500 !important;
  font-size: 0.8rem !important;
}
div[data-testid="stSelectbox"] label {
  color: var(--text-primary) !important;
  font-weight: 500 !important;
}

/* Global metric */
div[data-testid="stMetric"] {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: 12px;
  padding: 0.5rem 0.8rem !important;
}
div[data-testid="stMetric"] label {
  color: var(--text-secondary) !important;
  font-size: 0.7rem !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
  color: #00D4FF !important;
  font-weight: 700 !important;
}

/* Sidebar compact stats — custom cards, NOT st.metric */
.sidebar-stat-row {
  display: flex; gap: 6px; margin: 0.4rem 0;
}
.sidebar-stat-card {
  flex: 1;
  background: rgba(20, 20, 45, 0.5);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 10px;
  padding: 0.4rem 0.2rem;
  text-align: center;
  transition: all 0.25s;
}
.sidebar-stat-card:hover {
  border-color: rgba(0, 212, 255, 0.3);
  box-shadow: 0 0 12px rgba(0, 212, 255, 0.08);
}
.sidebar-stat-card .stat-num {
  font-size: 0.95rem; font-weight: 700;
  background: linear-gradient(135deg, #00D4FF, #7B2FFF);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1.1;
}
.sidebar-stat-card .stat-desc {
  font-size: 0.52rem; color: #8888AA;
  margin-top: 1px;
  letter-spacing: 0.3px;
}

/* Expander */
details[data-testid="stExpander"] {
  background: var(--bg-card);
  backdrop-filter: var(--glass-blur);
  border: 1px solid var(--border-subtle) !important;
  border-radius: 12px !important;
  margin: 0.5rem 0;
}

/* Smooth image transition */
div[data-testid="stSidebar"] img {
  transition: opacity 0.25s ease-out;
}
</style>

<!-- Background effects: scanning beam + data particles (2 divs, minimal DOM) -->
<div class="scan-beam"></div>
<div class="data-particles"></div>
""", unsafe_allow_html=True)

# ============================================================
# Species definitions
# ============================================================
SPECIES_LIST = {
    'banded_kokopu': 'Banded Kokopu (whitebait)',
    'longfin_eel': 'Longfin Eel (tuna)',
    'shortfin_eel': 'Shortfin Eel',
    'inanga': 'Inanga (whitebait)',
    'koura': 'Koura (freshwater crayfish)',
    'common_bully': 'Common Bully',
    'upland_bully': 'Upland Bully',
    'redfin_bully': 'Redfin Bully',
    'koaro': 'Koaro (whitebait)',
    'torrentfish': 'Torrentfish',
}

SPECIES_EMOJI = {
    'banded_kokopu': '🐟', 'longfin_eel': '🐍', 'shortfin_eel': '🪱',
    'inanga': '🐟', 'koura': '🦞', 'common_bully': '🐠',
    'upland_bully': '🐠', 'redfin_bully': '🐡', 'koaro': '🐟',
    'torrentfish': '🐡',
}

# Short names for sidebar buttons (keep them narrow & uniform)
SPECIES_SHORT = {
    'banded_kokopu': '🐟 Kokopu',
    'longfin_eel': '🐍 Longfin',
    'shortfin_eel': '🪱 Shortfin',
    'inanga': '🐟 Inanga',
    'koura': '🦞 Koura',
    'common_bully': '🐠 C.Bully',
    'upland_bully': '🐠 U.Bully',
    'redfin_bully': '🐡 R.Bully',
    'koaro': '🐟 Koaro',
    'torrentfish': '🐡 Torrent',
}

SPECIES_IMAGES = {
    'banded_kokopu': 'https://upload.wikimedia.org/wikipedia/commons/6/6b/Galaxias_fasciatus.jpg',
    'longfin_eel': 'https://upload.wikimedia.org/wikipedia/commons/4/4c/Anguilla_dieffenbachii.jpg',
    'shortfin_eel': 'https://upload.wikimedia.org/wikipedia/commons/6/6a/Anguilla_australis.jpg',
    'inanga': 'https://upload.wikimedia.org/wikipedia/commons/7/7d/Galaxias_maculatus.jpg',
    'koura': 'https://upload.wikimedia.org/wikipedia/commons/4/4e/Paranephrops_planifrons.jpg',
    'common_bully': 'https://upload.wikimedia.org/wikipedia/commons/b/b5/Gobiomorphus_cotidianus.jpg',
    'redfin_bully': 'https://upload.wikimedia.org/wikipedia/commons/7/7a/Gobiomorphus_huttoni.jpg',
    'torrentfish': 'https://upload.wikimedia.org/wikipedia/commons/d/d2/Cheimarrichthys_fosteri.jpg',
}

SPECIES_RATES = {
    'longfin_eel': 25.3, 'shortfin_eel': 16.6, 'common_bully': 11.1,
    'koura': 9.2, 'upland_bully': 10.8, 'inanga': 8.2,
    'redfin_bully': 7.7, 'banded_kokopu': 8.0, 'koaro': 6.1, 'torrentfish': 4.0,
}

# ============================================================
# Cached model/data loading
# ============================================================
@st.cache_resource
def load_scaler(species: str = ""):
    if species:
        sp_path = os.path.join(MODEL_DIR, species, "scaler.pkl")
        if os.path.exists(sp_path):
            return joblib.load(sp_path)
    root_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if os.path.exists(root_path):
        return joblib.load(root_path)
    raise FileNotFoundError(f"No scaler found for species='{species}'.")

@st.cache_resource
def load_features():
    return joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

@st.cache_resource
def load_encoders():
    le_dict = {}
    for col in ['Region', 'recLandcover']:
        le_path = os.path.join(MODEL_DIR, f'le_{col}.pkl')
        if os.path.exists(le_path):
            le_dict[col] = joblib.load(le_path)
    return le_dict

def load_model_for_species(species, model_type):
    path = os.path.join(MODEL_DIR, species, f'model_{model_type}.pkl')
    if os.path.exists(path):
        return joblib.load(path)
    if species == 'banded_kokopu':
        fallback = {'knn': 'model_knn.pkl', 'rf': 'model_rf.pkl',
                    'xgb': 'model_xgb.pkl', 'nb': 'model_nb.pkl'}
        fb_path = os.path.join(MODEL_DIR, fallback.get(model_type, ''))
        if os.path.exists(fb_path):
            return joblib.load(fb_path)
    return None

@st.cache_resource
def load_region_species_rates():
    path = os.path.join(PROJECT_DIR, 'data', 'nz_fish_multispecies.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    species_cols = list(SPECIES_LIST.keys())
    rates = {}
    for region in df['Region'].unique():
        region_df = df[df['Region'] == region]
        rates[region] = {}
        for sp in species_cols:
            if sp in df.columns:
                rates[region][sp] = region_df[sp].mean() * 100
    return rates

@st.cache_resource
def load_multispecies_results():
    path = os.path.join(MODEL_DIR, 'multispecies_results.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_resource
def load_map_data(species):
    path = os.path.join(PROJECT_DIR, 'data', 'nz_fish_multispecies.csv')
    if not os.path.exists(path):
        path = os.path.join(PROJECT_DIR, 'data', 'nz_fish_water_merged.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        target_col = species if species in df.columns else 'is_target'
        sites = df[['Latitude', 'Longitude', 'Region', target_col]].dropna(subset=['Latitude', 'Longitude'])
        sites = sites.rename(columns={target_col: 'is_target'})
        if len(sites) > 500:
            sites = sites.sample(500, random_state=42)
        return sites
    return None

# ============================================================
# Load assets
# ============================================================
with st.spinner("Loading app..."):
    FEATURES = load_features()
    le_dict = load_encoders()
    region_ref = pd.read_csv(os.path.join(PROJECT_DIR, 'data', 'nz_regions_reference.csv'), index_col=0)
    region_species_rates = load_region_species_rates()
    ms_results = load_multispecies_results()

FEATURE_LABELS = {
    'minimumElevation': ('Minimum Elevation', 'm', (0, 2000)),
    'distanceOcean': ('Distance to Ocean', 'km', (0, 1500)),
    'ASPM': ('ASPM Score', '', (0.0, 1.0)),
    'MCI': ('MCI Score', '', (40.0, 150.0)),
    'PercentageEPTTaxa': ('EPT Taxa', '%', (0.0, 100.0)),
    'QMCI': ('QMCI Score', '', (1.0, 9.0)),
    'TaxaRichness': ('Taxa Richness', '', (0.0, 40.0)),
    'distance': ('Spatial Distance', 'km', (0.0, 50.0)),
}

nn_features = [f for f in FEATURES if f not in ('Region_enc', 'recLandcover_enc')]

# ============================================================
# Init session state
# ============================================================
if 'prev_species' not in st.session_state:
    st.session_state['prev_species'] = 'banded_kokopu'
if 'prev_region' not in st.session_state:
    st.session_state['prev_region'] = ''
if 'has_prediction' not in st.session_state:
    st.session_state['has_prediction'] = False
if 'all_probs' not in st.session_state:
    st.session_state['all_probs'] = {}
if 'species_key' not in st.session_state:
    st.session_state['species_key'] = 'banded_kokopu'
if 'prediction_species' not in st.session_state:
    st.session_state['prediction_species'] = ''
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = {}
if 'selected_region' not in st.session_state:
    st.session_state['selected_region'] = ''
if 'selected_landcover' not in st.session_state:
    st.session_state['selected_landcover'] = ''

# ============================================================
# Feature 10: Navigation anchors
# ============================================================
st.markdown("""
<div style="display:flex;gap:1.5rem;justify-content:center;flex-wrap:wrap;padding:0.3rem 0 0.6rem 0;
            font-size:0.8rem;color:var(--text-secondary);border-bottom:1px solid rgba(26,58,82,0.5);margin-bottom:0.8rem;">
  <a href="#prediction" style="color:#8888AA;text-decoration:none;transition:color 0.2s;"
     onmouseover="this.style.color='#00D4FF'" onmouseout="this.style.color='#8888AA'">🎯 Predict</a>
  <a href="#analysis" style="color:#8888AA;text-decoration:none;transition:color 0.2s;"
     onmouseover="this.style.color='#00D4FF'" onmouseout="this.style.color='#8888AA'">📈 Analysis</a>
  <a href="#map" style="color:#8888AA;text-decoration:none;transition:color 0.2s;"
     onmouseover="this.style.color='#00D4FF'" onmouseout="this.style.color='#8888AA'">🗺️ Map</a>
  <a href="#performance" style="color:#8888AA;text-decoration:none;transition:color 0.2s;"
     onmouseover="this.style.color='#00D4FF'" onmouseout="this.style.color='#8888AA'">📊 Performance</a>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Header with Feature 1: Glassmorphism stat cards
# ============================================================
st.markdown("""
<div style="text-align:center;margin-bottom:0.5rem;">
  <div style="font-size:2.2rem;font-weight:800;margin-bottom:0.2rem;
    background:linear-gradient(135deg,#00D4FF 0%,#7B2FFF 70%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    filter:drop-shadow(0 0 18px rgba(0,212,255,0.35));">
    🐟 NZ Native Fish Presence Predictor
  </div>
  <div style="font-size:0.95rem;color:#8888AA;margin-bottom:1rem;">
    Predicting native freshwater fish distribution across Aotearoa NZ using machine learning
  </div>
</div>
""", unsafe_allow_html=True)

# Glassmorphism stat cards
cols = st.columns(4)
stat_items = [
    ("10", "Species"),
    ("5", "Algorithms"),
    ("8", "Parameters"),
    ("62K", "Samples"),
]
for col, (val, label) in zip(cols, stat_items):
    col.markdown(f"""
    <div class="glass-card">
      <div class="stat-value">{val}</div>
      <div class="stat-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# Sidebar
# ============================================================
st.sidebar.markdown("""
<div style="text-align:center;padding:0.3rem 0 0.8rem 0;border-bottom:1px solid rgba(26,58,82,0.5);margin-bottom:0.5rem;">
  <img src="https://upload.wikimedia.org/wikipedia/en/5/5e/Massey_University_Logo.svg"
       style="height:40px;margin-bottom:0.3rem;" alt="Massey University">
  <div style="font-size:0.7rem;color:#8888AA;">Te Kunenga ki Purehuroa</div>
  <div style="font-size:0.65rem;color:#8888AA;">158.755 · Project 4 · 2026 S1</div>
</div>
""", unsafe_allow_html=True)

# ---- Species selector (native selectbox for speed) ----
st.sidebar.markdown("**🐟 Select Species**")

species_key = st.sidebar.selectbox(
    "Native Fish Species",
    list(SPECIES_LIST.keys()),
    format_func=lambda x: SPECIES_LIST[x],
    index=list(SPECIES_LIST.keys()).index(st.session_state['species_key'])
    if st.session_state['species_key'] in SPECIES_LIST else 0,
    key="species_selectbox"
)

# Track species changes
if species_key != st.session_state['prev_species']:
    st.session_state['prev_species'] = species_key
    st.session_state['has_prediction'] = False

species_name = SPECIES_LIST[species_key]
base_rate = SPECIES_RATES.get(species_key, 8.0)

# Show fish image
local_img = os.path.join(PROJECT_DIR, 'data', 'images', f'{species_key}.jpg')
if os.path.exists(local_img):
    st.sidebar.image(local_img, caption=species_name, width='stretch')
elif SPECIES_IMAGES.get(species_key):
    st.sidebar.image(SPECIES_IMAGES[species_key], caption=species_name, width='stretch')

st.sidebar.caption(f"Historical presence rate: **{base_rate:.1f}%** of NZ sampling sites")

st.sidebar.markdown("---")
st.sidebar.header("📍 Location & Environment")

REGIONS_SORTED = sorted(le_dict['Region'].classes_.tolist()) if 'Region' in le_dict else []
LANDCOVERS = sorted(le_dict['recLandcover'].classes_.tolist()) if 'recLandcover' in le_dict else []

use_location = st.sidebar.checkbox("🏙️ Select by Region (auto-fill)", value=True)

defaults = {}
selected_region = REGIONS_SORTED[0] if REGIONS_SORTED else "auckland"
region_changed = False
region_match = None

if use_location and region_ref is not None and REGIONS_SORTED:
    selected_region = st.sidebar.selectbox("Region", REGIONS_SORTED)

    if st.session_state['prev_region'] != selected_region:
        region_changed = True
        st.session_state['prev_region'] = selected_region

    region_match = None
    for idx in region_ref.index:
        idx_norm = idx.lower().replace('ā', 'a').replace('ō', 'o').replace('ū', 'u')
        if selected_region.lower() == idx_norm:
            region_match = idx
            break

    if region_match:
        row = region_ref.loc[region_match]
        for feat in FEATURES:
            if feat in row.index and pd.notna(row[feat]):
                defaults[feat] = float(row[feat])
        sp_rate = None
        if region_species_rates and region_match:
            sp_rate = region_species_rates.get(region_match, {}).get(species_key, None)

        if sp_rate is not None:
            st.sidebar.caption(f"📊 {int(row['sample_count']):,} samples | **{species_name}** rate: **{sp_rate:.1f}%**")
        else:
            st.sidebar.caption(f"📊 {int(row['sample_count']):,} samples")

# Force slider values to match new region defaults when region changes
if region_changed:
    for feat in FEATURES:
        if feat in defaults:
            st.session_state[f"param_{feat}"] = defaults[feat]

# Quick stats in sidebar — custom HTML for full control
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div class="sidebar-stat-row">
  <div class="sidebar-stat-card">
    <div class="stat-num">{len(FEATURES)}</div>
    <div class="stat-desc">features</div>
  </div>
  <div class="sidebar-stat-card">
    <div class="stat-num">62K</div>
    <div class="stat-desc">samples</div>
  </div>
  <div class="sidebar-stat-card">
    <div class="stat-num">{len(SPECIES_LIST)}</div>
    <div class="stat-desc">species</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Retrain section
st.sidebar.markdown("---")
with st.sidebar.expander("🔄 Retrain Models", expanded=False):
    st.warning("⚠️ Training takes ~15 min for all 10 species")
    train_all = st.checkbox("Retrain single-species models too", value=False)
    if st.button("🚀 Retrain All Species Models", use_container_width=True):
        import subprocess, sys, time
        scripts = []
        if train_all:
            scripts.append(["train_all_models.py", "Single-species (kNN+NB+RF+XGBoost)"])
        scripts.append(["train_multispecies.py", "Multi-species (10 fish, RF+XGBoost+kNN)"])
        log_placeholder = st.empty()
        all_logs = []
        for script, desc in scripts:
            log_placeholder.info(f"Running {desc}...")
            proc = subprocess.Popen(
                [sys.executable, "-X", "utf8", script],
                cwd=PROJECT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )
            for line in proc.stdout:
                all_logs.append(line)
                log_placeholder.code("".join(all_logs[-8:]), language="text")
            proc.wait()
            if proc.returncode == 0:
                all_logs.append(f"\n✅ {desc} completed!\n")
            else:
                all_logs.append(f"\n❌ {desc} failed with code {proc.returncode}\n")
            log_placeholder.code("".join(all_logs[-10:]), language="text")
        st.cache_resource.clear()
        st.cache_data.clear()
        log_placeholder.success("✅ All training complete! Refresh the page to use new models.")
        if st.button("🔄 Refresh Page"):
            st.rerun()

st.sidebar.caption("Models loaded on-demand per species.")

# ============================================================
# Feature 5: Main area with Tabs
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Prediction",
    "📈 Analysis",
    "🗺️ NZ Map",
    "📊 Performance"
])

# ============================================================
# TAB 1: Prediction (Feature 2: gauge + Feature 4: form auto-predict)
# ============================================================
with tab1:
    st.markdown('<div id="prediction"></div>', unsafe_allow_html=True)

    # Form for parameter input
    with st.form("prediction_form", clear_on_submit=False):
        st.markdown(f"### 🎯 Predict: {species_name}")

        # Parameters in 2 columns
        pc1, pc2 = st.columns(2)

        user_input = {}
        half = len(nn_features) // 2
        for i, feat in enumerate(nn_features):
            col = pc1 if i < half else pc2
            label, unit, (lo, hi) = FEATURE_LABELS.get(feat, (feat, '', (0.0, 100.0)))
            display_label = f"{label} ({unit})" if unit else label
            default_val = defaults.get(feat, float(lo + hi) / 2)
            default_val = max(lo, min(hi, default_val))
            step = max(0.01, (hi - lo) / 100)
            user_input[feat] = col.slider(
                display_label,
                min_value=float(lo), max_value=float(hi),
                value=float(round(default_val, 2)),
                step=float(round(step, 3)),
                key=f"param_{feat}"
            )

        # Land cover — auto-fill from region reference when region changes
        default_lc = LANDCOVERS[0] if LANDCOVERS else "Pasture"
        if region_match and 'recLandcover' in region_ref.columns:
            ref_lc = region_ref.loc[region_match, 'recLandcover']
            if pd.notna(ref_lc) and ref_lc in LANDCOVERS:
                default_lc = ref_lc
        if region_changed and default_lc in LANDCOVERS:
            st.session_state['selected_landcover'] = default_lc
        if st.session_state.get('selected_landcover', '') not in LANDCOVERS:
            st.session_state['selected_landcover'] = default_lc

        selected_landcover = LANDCOVERS[0] if LANDCOVERS else "Pasture"
        if LANDCOVERS:
            selected_landcover = st.selectbox(
                "Land Cover (surrounding vegetation)",
                LANDCOVERS,
                index=LANDCOVERS.index(st.session_state['selected_landcover'])
                if st.session_state['selected_landcover'] in LANDCOVERS else 0,
                help="Dominant land use around the river site. Auto-filled to the most common type in this region."
            )
            st.session_state['selected_landcover'] = selected_landcover

        submitted = st.form_submit_button("🔍 Run Prediction", type="primary", use_container_width=True)

    # Auto-trigger on region change only (species change is instant UI swap, no auto-predict)
    if not submitted and st.session_state.get('has_prediction') and region_changed:
        submitted = True

    # Track if results are for a different species than currently selected
    results_stale = (
        st.session_state.get('has_prediction') and
        st.session_state.get('prediction_species', '') != species_key
    )

    if submitted:
        st.session_state['prev_species'] = species_key
        with st.spinner(f"Computing predictions for {species_name}..."):
            input_vals = [user_input[f] for f in nn_features]
            region_enc = le_dict['Region'].transform([selected_region])[0] if 'Region' in le_dict else 0
            lc_enc = le_dict['recLandcover'].transform([selected_landcover])[0] if 'recLandcover' in le_dict else 0
            X_input = np.array([input_vals + [region_enc, lc_enc]])
            scaler = load_scaler(species_key)
            X_scaled = scaler.transform(X_input)

            all_probs = {}
            for model_type, display_name in [
                ('knn', 'kNN'), ('nb', 'Naive Bayes'), ('rf', 'Random Forest'), ('xgb', 'XGBoost')
            ]:
                model = load_model_for_species(species_key, model_type)
                if model is not None:
                    prob = model.predict_proba(X_scaled)[0, 1]
                    all_probs[display_name] = prob

            ensemble_probs = [all_probs[n] for n in ['kNN', 'Random Forest', 'XGBoost'] if n in all_probs]
            if ensemble_probs:
                all_probs['Ensemble'] = float(np.mean(ensemble_probs))

            st.session_state['all_probs'] = all_probs
            st.session_state['has_prediction'] = True
            st.session_state['prediction_species'] = species_key
            st.session_state['user_input'] = user_input
            st.session_state['selected_region'] = selected_region
            st.session_state['selected_landcover'] = selected_landcover
            st.session_state['_just_predicted'] = True

    # Auto-switch to Analysis tab after prediction
    if st.session_state.get('_just_predicted'):
        st.session_state['_just_predicted'] = False
        st.markdown("""
        <script>
        (function switchTab() {
            var tabs = document.querySelectorAll('button[data-baseweb="tab"]');
            if (tabs.length < 2) { setTimeout(switchTab, 50); return; }
            tabs[1].click();
        })();
        </script>
        """, unsafe_allow_html=True)

    # Parameter explanations (Tab 1 since that's where params are)
    with st.expander("📖 What do these parameters mean?", expanded=False):
        st.markdown("""
        📏 **Elevation** — height above sea level. Lower = warmer, easier for migratory fish to reach.

        🌊 **Distance to Ocean** — how far the site is from the coast. Many NZ native fish migrate between rivers and the sea.

        🐛 **MCI** — Macroinvertebrate Community Index. Higher = cleaner water, more aquatic insects.

        🐛 **QMCI** — Quantitative MCI. Weighted by abundance of each macroinvertebrate species.

        🪰 **EPT Taxa** — % of mayflies, stoneflies, caddisflies. Pollution-sensitive insects. Higher = better water quality.

        🧪 **ASPM** — Average Score Per Metric. Macroinvertebrate-based water quality indicator.

        🔬 **Taxa Richness** — number of different macroinvertebrate types found. More types = healthier stream.

        📍 **Spatial Distance** — distance between fish sampling site and nearest water quality monitoring station.

        **Land Cover:**
        🌿 **Native vegetation** — best habitat: shade, leaf litter, clean water.
        🐄 **Pasture** — moderate: livestock can erode banks, add nutrients.
        🌲 **Plantation forest** — mixed: managed forestry, clear-felling sends sediment into streams.
        🌳 **Exotic forest** — mixed: non-native tree cover, variable water quality.
        🏙️ **Urban** — worst: stormwater runoff, concrete channels, warmer water.
        """)

    # Show prediction results
    if st.session_state.get('has_prediction'):

        # Stale results warning when species changed but prediction not re-run
        if results_stale:
            old_name = SPECIES_LIST.get(
                st.session_state.get('prediction_species', ''),
                'another species'
            )
            st.warning(
                f"⚠️ Results below are for **{old_name}**. "
                f"Click **Run Prediction** to update for **{species_name}**."
            )

        all_probs = st.session_state.get('all_probs', {})
        main_prob = all_probs.get('Ensemble', list(all_probs.values())[-1] if all_probs else 0.5)
        base_rate = SPECIES_RATES.get(species_key, 8.0)

        # ---- Feature 2: Gauge meter ----
        pct = int(main_prob * 100)
        gauge_html = f"""
        <div class="gauge-container">
          <div style="font-size:1rem;color:#E8E8F0;font-weight:600;margin-bottom:0.2rem;">
            Habitat Suitability Score
          </div>
          <div class="gauge-value">{pct}%</div>
          <div class="gauge-track">
            <div class="gauge-fill" style="width:{100-pct}%;"></div>
            <div class="gauge-needle" style="left:{pct}%;"></div>
          </div>
          <div class="gauge-labels">
            <span>🟢 Yes</span><span>🟡 Likely</span><span>🟠 Unlikely</span><span>🔴 No</span>
          </div>
        </div>
        """
        st.markdown(gauge_html, unsafe_allow_html=True)

        # ---- Feature 7: Metrics with delta ----
        mc1, mc2, mc3 = st.columns(3)
        if main_prob >= 0.7:
            status = "Yes 🟢"
        elif main_prob >= 0.5:
            status = "Likely 🟡"
        elif main_prob >= 0.3:
            status = "Unlikely 🟠"
        else:
            status = "No 🔴"

        delta = main_prob - (base_rate / 100.0)
        mc1.metric("Status", status)
        mc2.metric("Habitat Suitability", f"{main_prob:.1%}",
                   delta=f"{delta:+.1%} vs avg" if abs(delta) > 0.01 else None)
        confidence = ("High" if abs(main_prob - 0.5) > 0.3 else
                      "Medium" if abs(main_prob - 0.5) > 0.15 else "Low")
        mc3.metric("Confidence", confidence)

        # Interpretation
        if main_prob >= 0.7:
            st.success(f"🟢 **High Suitability** — Most similar NZ sites had **{species_name}**. Highly recommended for field surveys.")
        elif main_prob >= 0.5:
            st.info(f"🟡 **Moderate Suitability** — About half of similar sites had **{species_name}**. Worth investigating.")
        elif main_prob >= 0.3:
            st.warning(f"🟠 **Below-Average** — A minority of similar sites had **{species_name}**. Better sites likely exist.")
        else:
            st.error(f"🔴 **Low Suitability** — Similar sites rarely had **{species_name}**. A different region is recommended.")

        with st.expander("💡 How to interpret this probability", expanded=False):
            st.markdown(f"""
            **This is NOT a weather forecast — it's spatial habitat suitability.**

            **How it works:**
            1. You provide environmental parameters (elevation, distance to ocean, water quality, etc.)
            2. The model compares them against **62,507 historical sampling sites** across NZ
            3. It answers: *"At sites with similar environmental conditions, how often was **{species_name}** found?"*

            **Example interpretation:**
            - **Probability = 80%** → ~80% of similar sites had this fish → great place to look
            - **Probability = 50%** → ~50% of similar sites had it → worth trying
            - **Probability = 15%** → only ~15% of similar sites had it → unlikely
            - **Probability = 3%** → almost no similar sites had it → look elsewhere

            **Key environmental factors:**
            - **Elevation**: lower-elevation streams tend to have more native fish
            - **Distance to ocean**: many NZ native fish are migratory (diadromous)
            - **MCI / QMCI**: macroinvertebrate indices reflecting water quality and stream health
            """)
    else:
        st.info("👆 Adjust parameters above and click **Run Prediction** to see results.")

# ============================================================
# TAB 2: Analysis — Rich comparison charts
# ============================================================
with tab2:
    st.markdown('<div id="analysis"></div>', unsafe_allow_html=True)

    if st.session_state.get('has_prediction'):
        # When species changed but prediction not re-run, skip heavy charts
        if results_stale:
            old_name = SPECIES_LIST.get(
                st.session_state.get('prediction_species', ''), 'another species'
            )
            st.info(
                f"📊 Charts below show results for **{old_name}**. "
                f"Click **Run Prediction** to refresh analysis for **{species_name}**."
            )
            # Show simplified view — just the bar chart, no heavy subplots
            all_probs = st.session_state.get('all_probs', {})
            names = list(all_probs.keys())
            probs = list(all_probs.values())
            main_prob = all_probs.get('Ensemble', probs[-1] if probs else 0.5)
            colors = ['#00D4FF' if p >= 0.5 else '#FF4466' for p in probs]
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Bar(y=names, x=probs, orientation='h',
                marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.08)', width=1)),
                text=[f'{p:.1%}' for p in probs], textposition='outside',
                textfont=dict(color='#E8E8F0', size=13), showlegend=False))
            fig.add_vline(x=0.5, line_dash='dash', line_color='rgba(255,255,255,0.2)')
            fig.update_layout(
                xaxis=dict(range=[0,1], color='#8888AA', gridcolor='rgba(255,255,255,0.04)', tickformat='.0%'),
                yaxis=dict(color='#8888AA'), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=50, t=10, b=10), height=250)
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            knn_prob_stale = all_probs.get('kNN', -1)
            if knn_prob_stale >= 0.99 or knn_prob_stale <= 0.01:
                st.caption(
                    f"⚠️ kNN returned {knn_prob_stale:.0%}. With k=5 and SMOTE, "
                    "all neighbours can land in one class. The number looks certain "
                    "but isn't."
                )
            st.caption("⚠️ Run a fresh prediction to see the full analysis dashboard.")
        else:
            # Full rich analysis (only when results are fresh)
            all_probs = st.session_state.get('all_probs', {})
            user_input = st.session_state.get('user_input', {})
            main_prob = all_probs.get('Ensemble', list(all_probs.values())[-1] if all_probs else 0.5)
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

        # ========================================================
        # ROW 1: Ensemble Donut (left) + Model Bars (right)
        # ========================================================
        st.markdown("### 📈 Model Analysis Dashboard")
        c_left, c_right = st.columns([1, 1.2])

        with c_left:
            # ---- Donut gauge ----
            pct = main_prob * 100
            if main_prob >= 0.7:
                gauge_color = '#00FF88'
            elif main_prob >= 0.5:
                gauge_color = '#00D4FF'
            elif main_prob >= 0.3:
                gauge_color = '#FFB347'
            else:
                gauge_color = '#FF4466'

            fig_donut = go.Figure()
            fig_donut.add_trace(go.Pie(
                values=[pct, 100 - pct],
                hole=0.72,
                marker=dict(colors=[gauge_color, 'rgba(255,255,255,0.04)']),
                textinfo='none',
                hoverinfo='none',
                sort=False,
                showlegend=False,
            ))
            fig_donut.add_annotation(
                text=f"<b style='font-size:2rem;color:{gauge_color}'>{pct:.0f}%</b>",
                showarrow=False, x=0.5, y=0.5,
            )
            fig_donut.add_annotation(
                text=f"<span style='font-size:0.7rem;color:#8888AA'>Ensemble Score</span>",
                showarrow=False, x=0.5, y=0.38,
            )
            fig_donut.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10),
                height=260,
            )
            st.plotly_chart(fig_donut, width='stretch', config={'displayModeBar': False})

            # Agreement badge — count base models only (exclude Ensemble itself)
            base_probs = {k: v for k, v in all_probs.items() if k != 'Ensemble'}
            agree = sum(1 for p in base_probs.values() if (p >= 0.5) == (main_prob >= 0.5))
            total = len(base_probs)
            st.markdown(f"""
            <div style="background:rgba(20,20,45,0.5);border:1px solid {gauge_color}44;border-radius:12px;
                        padding:0.6rem;text-align:center;margin-top:-0.5rem;">
              <span style="color:#8888AA;font-size:0.75rem;">Model Consensus: </span>
              <span style="color:{gauge_color};font-weight:700;font-size:0.9rem;">{agree}/{total}</span>
            </div>
            """, unsafe_allow_html=True)

        with c_right:
            # ---- Vertical bar chart with glow dots ----
            names = list(all_probs.keys())
            probs = list(all_probs.values())
            colors = ['#00FF88' if p >= 0.7 else '#00D4FF' if p >= 0.5 else '#FFB347' if p >= 0.3 else '#FF4466'
                      for p in probs]

            fig_bars = go.Figure()
            fig_bars.add_trace(go.Bar(
                y=names, x=probs, orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(255,255,255,0.08)', width=1),
                    opacity=0.85,
                ),
                text=[f'  {p:.1%}  ' for p in probs],
                textposition='outside',
                textfont=dict(color='#E8E8F0', size=13, family='monospace'),
                hovertemplate='<b>%{y}</b>: %{x:.2%}<extra></extra>',
                width=0.55, showlegend=False,
            ))
            # Add glow dots at bar tips
            fig_bars.add_trace(go.Scatter(
                x=[p + 0.01 for p in probs], y=names,
                mode='markers',
                marker=dict(size=10, color=colors, line=dict(color='#FFFFFF', width=1.5),
                           symbol='diamond'),
                hoverinfo='none',
                showlegend=False,
            ))
            fig_bars.add_vline(x=0.5, line_dash='dash', line_color='rgba(255,255,255,0.2)',
                               line_width=2.5, annotation_text='50% threshold',
                               annotation_font_color='#8888AA', annotation_font_size=10)
            fig_bars.update_layout(
                xaxis=dict(range=[-0.02, 1.08], title='', color='#8888AA',
                           gridcolor='rgba(255,255,255,0.04)', tickformat='.0%',
                           zerolinecolor='rgba(255,255,255,0.06)'),
                yaxis=dict(color='#E8E8F0', gridcolor='rgba(255,255,255,0.02)',
                           categoryorder='array', categoryarray=list(reversed(names))),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#8888AA'),
                margin=dict(l=10, r=60, t=10, b=10),
                height=280,
            )
            st.plotly_chart(fig_bars, width='stretch', config={'displayModeBar': False})

        # ========================================================
        # ROW 2: Feature Deviation Waterfall (left) + Model Spread (right)
        # ========================================================
        st.markdown("---")
        st.markdown("### 🔬 Feature Impact & Model Spread")

        c_f1, c_f2 = st.columns([1.2, 1])

        with c_f1:
            # ---- Feature deviation waterfall ----
            norm_vals = []
            display_feats = []
            for feat in nn_features:
                label, unit, (lo, hi) = FEATURE_LABELS.get(feat, (feat, '', (0, 100)))
                val = user_input.get(feat, 0)
                norm = (val - lo) / (hi - lo) if hi > lo else 0.5
                norm_vals.append(max(0.0, min(1.0, norm)))
                display_feats.append(label)

            pct_vals = [v * 100 for v in norm_vals]  # 0–100 scale

            fig_dev = go.Figure()
            colors_fp = ['#00D4FF' if v >= 50 else '#FF4466' for v in pct_vals]
            fig_dev.add_trace(go.Bar(
                y=display_feats, x=pct_vals,
                orientation='h',
                marker=dict(
                    color=colors_fp,
                    line=dict(color='rgba(255,255,255,0.06)', width=1),
                    opacity=0.9,
                ),
                text=[f'{v:.0f}%' for v in pct_vals],
                textposition='outside',
                textfont=dict(color='#E8E8F0', size=11),
                hovertemplate='%{y}: <b>%{x:.0f}%</b> of range<extra></extra>',
                width=0.6, showlegend=False,
            ))
            fig_dev.add_vline(x=50, line_color='rgba(255,255,255,0.3)', line_width=2.5,
                              line_dash='dash', annotation_text='Midpoint (50%)',
                              annotation_font_color='#8888AA', annotation_font_size=10)
            fig_dev.update_layout(
                xaxis=dict(range=[-2, 102], title='', color='#8888AA',
                           gridcolor='rgba(255,255,255,0.03)', tickformat='d',
                           ticksuffix='%', zeroline=False),
                yaxis=dict(color='#E8E8F0', autorange='reversed',
                           gridcolor='rgba(255,255,255,0.02)'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#8888AA'),
                margin=dict(l=10, r=65, t=30, b=20),
                height=310,
                title=dict(text='📊 Parameter Position Within Valid Range', font=dict(color='#E8E8F0', size=13)),
            )
            st.plotly_chart(fig_dev, width='stretch', config={'displayModeBar': False})
            st.caption(
                "**How to read:** Each bar shows where your input falls within that parameter's "
                "possible range (0% = minimum, 100% = maximum). "
                "The dashed line at 50% = the midpoint. "
                "**Red** = below midpoint (e.g. low elevation, low MCI); "
                "**Cyan** = above midpoint (e.g. high elevation, high MCI). "
                "Values near 0% or 100% mean your site is extreme for that parameter."
            )

        with c_f2:
            # ---- Model spread / range visualization ----
            prob_list = list(all_probs.values())
            if len(prob_list) >= 2:
                prob_min = min(prob_list)
                prob_max = max(prob_list)
                prob_range = prob_max - prob_min
                model_names = list(all_probs.keys())

                # Scatter range plot
                fig_range = go.Figure()
                # Range bar
                fig_range.add_trace(go.Scatter(
                    x=[prob_min, prob_max], y=['All Models', 'All Models'],
                    mode='lines',
                    line=dict(color='rgba(0, 212, 255, 0.4)', width=8),
                    hoverinfo='none',
                    showlegend=False,
                ))
                # Min/Max dots
                fig_range.add_trace(go.Scatter(
                    x=[prob_min], y=['All Models'],
                    mode='markers+text',
                    marker=dict(size=16, color='#FF4466', symbol='triangle-left'),
                    text=[f'  {prob_min:.1%}'],
                    textposition='middle right',
                    textfont=dict(color='#FF4466', size=11),
                    hoverinfo='none',
                    showlegend=False,
                ))
                fig_range.add_trace(go.Scatter(
                    x=[prob_max], y=['All Models'],
                    mode='markers+text',
                    marker=dict(size=16, color='#00FF88', symbol='triangle-right'),
                    text=[f'{prob_max:.1%}  '],
                    textposition='middle left',
                    textfont=dict(color='#00FF88', size=11),
                    hoverinfo='none',
                    showlegend=False,
                ))
                # Ensemble diamond
                fig_range.add_trace(go.Scatter(
                    x=[main_prob], y=['All Models'],
                    mode='markers+text',
                    marker=dict(size=17, color='#FFFFFF', symbol='diamond',
                               line=dict(color='#00D4FF', width=2.5)),
                    text=[f'<b> {main_prob:.1%}</b>'],
                    textposition='top center',
                    textfont=dict(color='#FFFFFF', size=13),
                    hoverinfo='none',
                    showlegend=False,
                    hovertemplate='Ensemble: %{x:.1%}',
                ))
                # Per-model dots
                for i, (name, prob) in enumerate(all_probs.items()):
                    if name != 'Ensemble':
                        fig_range.add_trace(go.Scatter(
                            x=[prob], y=[f'  {name}'],
                            mode='markers+text',
                            marker=dict(size=9, color='#00D4FF', opacity=0.7),
                            text=[f'{prob:.1%}'],
                            textposition='middle right',
                            textfont=dict(color='#8888AA', size=9),
                            hoverinfo='none',
                            showlegend=False,
                        ))

                fig_range.update_layout(
                    xaxis=dict(range=[-0.08, 1.08], title='', color='#8888AA',
                               tickformat='.0%', gridcolor='rgba(255,255,255,0.03)',
                               zeroline=False),
                    yaxis=dict(color='#8888AA', gridcolor='rgba(255,255,255,0.02)',
                               automargin=True),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=60, r=60, t=30, b=20),
                    height=310,
                    title=dict(text='🎯 Model Spread & Ensemble', font=dict(color='#E8E8F0', size=13)),
                )
                st.plotly_chart(fig_range, width='stretch', config={'displayModeBar': False})

                # Spread stat
                st.markdown(f"""
                <div style="background:rgba(20,20,45,0.5);border:1px solid rgba(255,255,255,0.06);border-radius:10px;
                            padding:0.5rem 0.6rem;text-align:center;margin-top:-0.5rem;">
                  <span style="color:#8888AA;font-size:0.7rem;">Spread </span>
                  <span style="color:#00D4FF;font-weight:700;">{prob_range:.1%}</span>
                  <span style="color:#8888AA;font-size:0.7rem;"> | Range: </span>
                  <span style="color:#FF4466;">{prob_min:.1%}</span>
                  <span style="color:#8888AA;"> – </span>
                  <span style="color:#00FF88;">{prob_max:.1%}</span>
                </div>
                """, unsafe_allow_html=True)
            st.caption(
                "💡 Spread = Max − Min: narrow spread → models agree; wide spread → high uncertainty. "
                "White ◆ marks the Ensemble (voting average), the best single estimate."
            )

            # Note when kNN hits extreme probability
            knn_prob = all_probs.get('kNN', -1)
            if knn_prob >= 0.99 or knn_prob <= 0.01:
                st.caption(
                    f"⚠️ kNN returned {knn_prob:.0%}. With k=5 and SMOTE, all "
                    "nearest neighbours can belong to one class. This is a known "
                    "limitation — the model is not as certain as the number suggests."
                )

    else:
        st.info("Run a prediction first to see model comparison and parameter analysis.")

# ============================================================
# TAB 3: NZ Map
# ============================================================
with tab3:
    st.markdown('<div id="map"></div>', unsafe_allow_html=True)
    st.markdown("### 🗺️ NZ Sampling Sites")

    map_sites = load_map_data(species_key)
    if map_sites is not None:
        try:
            import folium
            from streamlit_folium import folium_static

            m = folium.Map(location=[-41.0, 172.5], zoom_start=5.5,
                           tiles='CartoDB voyager', control_scale=True)

            for _, row in map_sites.iterrows():
                color = '#00D4FF' if row['is_target'] == 1 else '#FF4466'
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=3.5, color=color, fill=True,
                    fillColor=color, fillOpacity=0.7, weight=1.0
                ).add_to(m)

            st.caption(f"🟢 {species_name} Present | 🔴 Absent | Up to 500 sites shown")
            folium_static(m, width=1050, height=480)
        except ImportError:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor('#050510')
            ax.set_facecolor('#050510')
            for _, row in map_sites.iterrows():
                c = '#00D4FF' if row['is_target'] == 1 else '#FF4466'
                ax.scatter(row['Longitude'], row['Latitude'], c=c, alpha=0.6, s=8)
            ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
            ax.set_title(f'NZ Sampling Sites — {species_name}', color='#E8E8F0')
            ax.tick_params(colors='#8888AA')
            for spine in ax.spines.values(): spine.set_color('#222244')
            st.pyplot(fig)
    else:
        st.warning("Map data not available.")

# ============================================================
# TAB 4: Performance + Regional Stats + About
# ============================================================
with tab4:
    st.markdown('<div id="performance"></div>', unsafe_allow_html=True)

    st.markdown("### 📊 Multi-Species Model Performance")
    if ms_results is not None:
        st.markdown(f"**Currently selected: {species_name}**")
        num_cols = ms_results.select_dtypes(include='number').columns.tolist()
        st.dataframe(
            ms_results.style.format('{:.3f}', subset=num_cols),
            width='stretch'
        )
    else:
        st.success(f"✅ Models trained for {len(SPECIES_LIST)} species")

    st.markdown("---")
    st.markdown("### 📋 Regional Statistics")
    if region_ref is not None:
        display_cols = ['sample_count', 'fish_presence_rate', 'minimumElevation',
                        'distanceOcean', 'MCI', 'QMCI']
        st.dataframe(
            region_ref[display_cols].style.format({
                'fish_presence_rate': '{:.1f}%',
                'minimumElevation': '{:.0f}m',
                'distanceOcean': '{:.1f}km',
                'MCI': '{:.1f}',
                'QMCI': '{:.2f}'
            }),
            width='stretch'
        )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(f"""
    **Authors:** Xiaotao Wu, Ruoyang Hou, Xuanhui Li

    **158.755 — Data Science & Machine Learning** | Massey University | 2026 S1 | Project 4

    **10 Native Species:** Longfin Eel, Shortfin Eel, Common Bully, Koura, Upland Bully,
    Inanga, Redfin Bully, Banded Kokopu, Koaro, Torrentfish

    **Algorithms:** kNN, Naive Bayes, Random Forest, XGBoost, Voting Ensemble

    **Data:** NIWA NZFFD (182,833 raw) + LAWA (87,852 raw) → merged 62,507 sites via QGIS spatial join

    **What the probability means:** This is a **spatial habitat suitability score** —
    "At sites with similar environmental conditions across NZ, what proportion had this species?"

    ⚠️ Academic project only — not for regulatory or conservation decisions.
    """)

# ============================================================
# Footer JS (suppress Streamlit 'c' key clear-cache shortcut)
# ============================================================

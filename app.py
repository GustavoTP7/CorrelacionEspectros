import streamlit as st
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr, spearmanr
import plotly.graph_objects as go

def load_spectrum(uploaded_file):
    content = uploaded_file.read().decode('utf-8', errors='ignore').splitlines()
    data = []
    for line in content:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                wl, refl = float(parts[0]), float(parts[1])
                data.append((wl, refl))
            except:
                continue
    arr = np.array(data)
    return (arr[:,0], arr[:,1]) if arr.size else (np.array([]), np.array([]))

def detect_peaks(wl, y, prominence=0.01, min_dist_nm=10.0):
    if wl.size < 2 or y.size < 2:
        return np.array([])
    delta = wl[1] - wl[0]
    dist = max(1, int(round(min_dist_nm / delta)))
    idx, _ = signal.find_peaks(-y, prominence=prominence, distance=dist)
    return wl[idx]

def calculate_metrics(rep_pat, rep_sam):
    y1, y2 = rep_pat, rep_sam
    m = {
        'Pearson': pearsonr(y1, y2)[0],
        'Spearman': spearmanr(y1, y2)[0]
    }
    norm1, norm2 = np.linalg.norm(y1), np.linalg.norm(y2)
    m['Cosine'] = float(np.dot(y1, y2)/(norm1*norm2)) if norm1 and norm2 else 0.0
    cos_angle = np.clip(np.dot(y1, y2)/(norm1*norm2), -1.0, 1.0)
    m['SAM (deg)'] = float(np.degrees(np.arccos(cos_angle)))
    diff = y1 - y2
    m['RMSE'] = float(np.sqrt(np.mean(diff**2)))
    max_pat = max(np.max(y1), 1e-6)
    m['RMSE relativo vs Patrón'] = m['RMSE'] / max_pat
    return m

def improved_similarity(m):
    cos, sam = m['Cosine'], m['SAM (deg)']
    rmse_score = max(0, 1 - m['RMSE relativo vs Patrón'])
    pr, sr = m['Pearson'], m['Spearman']
    return (0.45*cos + 0.25*max(0, 1-sam/180) + 0.15*rmse_score +
            0.10*((pr+1)/2) + 0.05*((sr+1)/2)) * 100

st.title("Comparador Espectral - Interactivo (Fix)")
st.sidebar.header("Carga de espectros")
pat_files = st.sidebar.file_uploader("Patrones", type=['txt','csv','asd'], accept_multiple_files=True)
sam_files = st.sidebar.file_uploader("Muestras", type=['txt','csv','asd'], accept_multiple_files=True)

if pat_files and sam_files:
    # Load and align
    patterns = [load_spectrum(f) for f in pat_files]
    base_wl, _ = patterns[0]
    aligned_p = [
        np.interp(base_wl, wl, y) if wl.shape != base_wl.shape or not np.allclose(wl, base_wl) else y
        for wl, y in patterns
    ]
    rep_pat = np.mean(aligned_p, axis=0)
    samples = [load_spectrum(f) for f in sam_files]
    aligned_s = [
        np.interp(base_wl, wl, y) if wl.shape != base_wl.shape or not np.allclose(wl, base_wl) else y
        for wl, y in samples
    ]
    rep_sam = np.mean(aligned_s, axis=0)

    # Peaks and metrics
    p_pat = detect_peaks(base_wl, rep_pat)
    p_sam = detect_peaks(base_wl, rep_sam)
    peaks = np.union1d(p_pat, p_sam)
    m = calculate_metrics(rep_pat, rep_sam)
    total = improved_similarity(m)
    m['Similitud total (%)'] = total

    # Metrics table
    df_metrics = pd.DataFrame.from_dict(m, orient='index', columns=['Valor']).round(4)
    st.subheader("Métricas")
    st.table(df_metrics)

    # Peaks table
    prot = pd.Series(rep_pat, index=base_wl)
    smpl = pd.Series(rep_sam, index=base_wl)
    df_peaks = pd.DataFrame({
        'Wavelength (Patrón)': p_pat,
        'Reflectancia (Patrón)': prot.loc[p_pat].values,
        'Wavelength (Muestra)': p_sam,
        'Reflectancia (Muestra)': smpl.loc[p_sam].values
    }).round(4)
    st.subheader("Picos")
    st.table(df_peaks)

    # Interactive plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=base_wl, y=rep_pat, mode='lines', name='Patrón', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=base_wl, y=rep_sam, mode='lines', name='Muestra', line=dict(width=2, dash='dash')))
    ymin, ymax = min(rep_pat.min(), rep_sam.min()), max(rep_pat.max(), rep_sam.max())
    for x in peaks:
        fig.add_shape(type='line', x0=x, y0=ymin, x1=x, y1=ymax,
                      line=dict(color='gray', dash='dash'))
    fig.update_traces(hovertemplate='Wavelength: %{x:.2f}<br>Reflectance: %{y:.4f}')
    fig.update_layout(title=f"Similitud: {total:.2f} %", xaxis_title="Longitud de onda (nm)", yaxis_title="Reflectancia", hovermode='x')
    st.subheader("Gráfico Interactivo")
    st.plotly_chart(fig, use_container_width=True)

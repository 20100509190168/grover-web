import sys
sys.setrecursionlimit(10000)

import matplotlib
matplotlib.use('Agg')

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import time
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# ---------- 页面设置 ----------
st.set_page_config(page_title="Grover Quantum Search", layout="wide", page_icon="🔍")
st.title("🔍 Grover Quantum Search Algorithm Visualization")
st.markdown("---")

# ---------- 缓存模拟函数 ----------
@st.cache_data(show_spinner=False)
def run_ideal(n, target, k, shots=1024):
    qc = QuantumCircuit(n, n)
    for q in range(n):
        qc.h(q)
    target_bin = format(target, f'0{n}b')[::-1]
    for _ in range(k):
        # Oracle
        for q, bit in enumerate(target_bin):
            if bit == '0':
                qc.x(q)
        if n == 2:
            qc.cz(0, 1)
        elif n == 3:
            qc.h(2)
            qc.ccx(0, 1, 2)
            qc.h(2)
        for q, bit in enumerate(target_bin):
            if bit == '0':
                qc.x(q)
        # 扩散算符
        for q in range(n):
            qc.h(q)
            qc.x(q)
        if n == 2:
            qc.cz(0, 1)
        elif n == 3:
            qc.h(2)
            qc.ccx(0, 1, 2)
            qc.h(2)
        for q in range(n):
            qc.x(q)
            qc.h(q)
    qc.measure(range(n), range(n))
    sim = AerSimulator()
    counts = sim.run(qc, shots=shots).result().get_counts()
    return counts

def run_noisy(n, target, k, noise_p, shots=1024):
    qc = QuantumCircuit(n, n)
    for q in range(n):
        qc.h(q)
    target_bin = format(target, f'0{n}b')[::-1]
    for _ in range(k):
        for q, bit in enumerate(target_bin):
            if bit == '0':
                qc.x(q)
        if n == 2:
            qc.cz(0, 1)
        elif n == 3:
            qc.h(2)
            qc.ccx(0, 1, 2)
            qc.h(2)
        for q, bit in enumerate(target_bin):
            if bit == '0':
                qc.x(q)
        for q in range(n):
            qc.h(q)
            qc.x(q)
        if n == 2:
            qc.cz(0, 1)
        elif n == 3:
            qc.h(2)
            qc.ccx(0, 1, 2)
            qc.h(2)
        for q in range(n):
            qc.x(q)
            qc.h(q)
    qc.measure(range(n), range(n))
    noise_model = NoiseModel()
    if noise_p > 0:
        err1 = depolarizing_error(noise_p, 1)
        err2 = depolarizing_error(noise_p, 2)
        noise_model.add_all_qubit_quantum_error(err1, ['h', 'x', 'z'])
        noise_model.add_all_qubit_quantum_error(err2, ['cz', 'ccx'])
    sim = AerSimulator(noise_model=noise_model)
    counts = sim.run(qc, shots=shots).result().get_counts()
    return counts

# ---------- 绘图辅助函数 ----------
def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_bloch(n, k):
    N = 2**n
    theta = np.arcsin(1/np.sqrt(N))
    fig, ax = plt.subplots(figsize=(4,4))
    ax.add_artist(plt.Circle((0,0), 1, fill=False, color='gray'))
    ax.axhline(0, color='gray', alpha=0.3)
    ax.axvline(0, color='gray', alpha=0.3)
    ax.arrow(0,0, 0,1, head_width=0.06, fc='blue')
    ax.text(0.1, 1.05, '|t⟩', color='blue')
    s_angle = np.pi/2 - theta
    ax.arrow(0,0, np.cos(s_angle), np.sin(s_angle), head_width=0.06, fc='green', linestyle='dashed')
    ax.text(np.cos(s_angle)*1.1, np.sin(s_angle)*1.1, '|s⟩', color='green')
    cur_angle = np.pi/2 - (theta + 2*theta*k)
    ax.arrow(0,0, np.cos(cur_angle), np.sin(cur_angle), head_width=0.08, fc='red', linewidth=2)
    ax.set_xlim(-1.3,1.3); ax.set_ylim(-1.3,1.3)
    ax.set_aspect('equal')
    ax.set_title(f'Bloch Disk (k={k})')
    return fig_to_img(fig)

def plot_counts(counts):
    fig, ax = plt.subplots()
    states = sorted(counts.keys())
    vals = [counts[s] for s in states]
    ax.bar(states, vals, color='#1f77b4', edgecolor='white')
    ax.set_title('Measurement Distribution'); ax.set_ylabel('Counts')
    return fig_to_img(fig)

def plot_success_curve(rates):
    fig, ax = plt.subplots()
    ks = list(rates.keys()); vs = list(rates.values())
    ax.plot(ks, vs, 'o-', color='purple', markersize=8, linewidth=2)
    ax.set_xlabel('Iterations k'); ax.set_ylabel('Success Rate')
    ax.set_xticks(ks); ax.grid(True, alpha=0.3); ax.set_ylim(0,1.05)
    return fig_to_img(fig)

def plot_amplitude(counts, n):
    fig, ax = plt.subplots()
    total = sum(counts.values())
    states = [f'|{i:0{n}b}⟩' for i in range(2**n)]
    amps = [np.sqrt(counts.get(s,0)/total) for s in states]
    ax.bar(states, amps, color='orange', edgecolor='white')
    ax.set_title('Probability Amplitudes'); ax.set_ylabel('Magnitude')
    return fig_to_img(fig)

def plot_noise_curve(data, N):
    fig, ax = plt.subplots()
    ax.set_xlim(0,0.22); ax.set_ylim(0,1.05)
    ax.set_xlabel('Noise Strength p'); ax.set_ylabel('Success Rate')
    if data:
        ps = [d[0] for d in data]
        means = [d[1] for d in data]
        stds = [d[2] for d in data]
        ax.errorbar(ps, means, yerr=stds, fmt='o-', color='red', capsize=3)
        ax.axhline(y=1/N, color='gray', linestyle='--', label='Random Guess')
        ax.legend(); ax.grid(True, alpha=0.3)
    return fig_to_img(fig)

def plot_complexity(N, optimal_k):
    """经典 vs 量子查询复杂度柱状图"""
    fig, ax = plt.subplots(figsize=(5,4))
    labels = ['Classical\n(worst case)', 'Grover\n(optimal)']
    values = [N-1, optimal_k]
    bars = ax.bar(labels, values, color=['gray', 'red'], edgecolor='white')
    ax.set_ylabel('Queries / Iterations')
    ax.set_title('Quantum Speedup')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(val), ha='center', fontsize=12)
    return fig_to_img(fig)

# ---------- 状态管理 ----------
if 'k' not in st.session_state:
    st.session_state.k = 0
if 'noise_data' not in st.session_state:
    st.session_state.noise_data = []

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("⚙️ Configuration")
    n = st.selectbox("Number of Qubits", [2, 3], index=0)
    N = 2**n
    target = st.selectbox("Target State", list(range(N)),
                          format_func=lambda x: f"|{x:0{n}b}⟩")
    optimal_k = int(np.floor(np.pi/4 * np.sqrt(N)))

    st.markdown("---")
    st.subheader("🎮 Ideal Iteration")
    col1, col2 = st.columns(2)
    with col1:
        step = st.button("➕ Step", use_container_width=True)
    with col2:
        reset = st.button("🔄 Reset", use_container_width=True)
    auto = st.checkbox("▶️ Auto Play")

    st.markdown("---")
    st.subheader("🌪️ Noise Experiment")
    noise_p = st.slider("Depolarizing Noise p", 0.0, 0.2, 0.0, 0.01)
    noisy_btn = st.button("🔬 Run Noisy", use_container_width=True)
    clear_btn = st.button("🗑️ Clear Noise Data", use_container_width=True)

    st.markdown("---")
    st.info(f"💡 Optimal iterations: **{optimal_k}**")

# ---------- 按钮逻辑 ----------
if step:
    st.session_state.k += 1
if reset:
    st.session_state.k = 0
if auto:
    if st.session_state.k < optimal_k + 2:
        time.sleep(0.8)
        st.session_state.k += 1
        st.rerun()

k = st.session_state.k
target_str = format(target, f'0{n}b')
counts = run_ideal(n, target, k)
success = counts.get(target_str, 0) / 1024

# 成功率曲线数据
rates = {}
for i in range(4):
    cnt = run_ideal(n, target, i)
    rates[i] = cnt.get(target_str, 0) / 1024

# 噪声实验
if noisy_btn:
    cnt_n = run_noisy(n, target, optimal_k, noise_p)
    succ_n = cnt_n.get(target_str, 0) / 1024
    st.session_state.noise_data.append((noise_p, succ_n, 0))
    st.sidebar.success(f"p={noise_p:.2f}, success={succ_n:.1%}")

if clear_btn:
    st.session_state.noise_data.clear()

# ---------- 主界面显示 ----------
theta_deg = np.degrees(np.arcsin(1/np.sqrt(N)))
angle = 2 * theta_deg * k
theory = np.sin((2*k+1)*np.arcsin(1/np.sqrt(N)))**2

c1, c2, c3, c4 = st.columns(4)
c1.metric("Iteration k", k)
c2.metric("Total Rotation", f"{angle:.1f}°")
c3.metric("Theoretical Success", f"{theory:.1%}")
c4.metric("Measured Success", f"{success:.1%}")

st.markdown("---")

# 第一行图表
left1, right1 = st.columns(2)
with left1:
    st.subheader("🔵 Bloch Disk")
    st.image(plot_bloch(n, k), width=600)
with right1:
    st.subheader("📊 Measurement Distribution")
    st.image(plot_counts(counts), width=600)

# 第二行图表
left2, right2 = st.columns(2)
with left2:
    st.subheader("📈 Success Rate vs Iterations")
    st.image(plot_success_curve(rates), width=600)
with right2:
    st.subheader("🌪️ Noise Impact Curve")
    st.image(plot_noise_curve(st.session_state.noise_data, N), width=600)

# 第三行图表
left3, right3 = st.columns(2)
with left3:
    st.subheader("🔬 Probability Amplitudes")
    st.image(plot_amplitude(counts, n), width=600)
with right3:
    st.subheader("⚖️ Complexity Comparison")
    st.image(plot_complexity(N, optimal_k), width=600)

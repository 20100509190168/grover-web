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
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

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
        noise_model.add_all_qubit_quantum_error(err1, ['h', 'x', 'z'])
        err2 = depolarizing_error(noise_p, 2)
        noise_model.add_all_qubit_quantum_error(err2, ['cz'])
        err3 = depolarizing_error(noise_p, 3)
        noise_model.add_all_qubit_quantum_error(err3, ['ccx'])

    sim = AerSimulator(noise_model=noise_model)
    counts = sim.run(qc, shots=shots).result().get_counts()
    return counts

def run_readout_noisy(n, target, k, readout_p, shots=1024):
    """只加读出噪声的模拟，用于测量误差缓解对比"""
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
    if readout_p > 0:
        readout_error = ReadoutError([[1-readout_p, readout_p],
                                      [readout_p, 1-readout_p]])
        noise_model.add_all_qubit_readout_error(readout_error)
    sim = AerSimulator(noise_model=noise_model)
    counts = sim.run(qc, shots=shots).result().get_counts()
    return counts

def apply_readout_mitigation(counts, readout_p):
    """使用线性反演进行测量误差缓解（基于 numpy，兼容 Qiskit 1.x）"""
    n = len(next(iter(counts.keys())))  # 比特数
    dim = 2 ** n

    # 构造单比特混淆矩阵
    M1 = np.array([[1 - readout_p, readout_p],
                   [readout_p, 1 - readout_p]])
    # 构造完整混淆矩阵（Kronecker 积）
    M = M1.copy()
    for _ in range(n - 1):
        M = np.kron(M, M1)

    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return counts  # 矩阵不可逆时返回原始结果

    # 构建测量概率向量
    p_meas = np.zeros(dim)
    states = [format(i, f'0{n}b') for i in range(dim)]
    total = sum(counts.values())
    for i, s in enumerate(states):
        p_meas[i] = counts.get(s, 0) / total

    # 线性反演校正
    p_corr = M_inv @ p_meas
    p_corr = np.clip(p_corr, 0, None)  # 去除负概率
    p_sum = p_corr.sum()
    if p_sum > 0:
        p_corr = p_corr / p_sum

    # 转换回 counts 字典（按原始总测量次数缩放）
    corrected_counts = {}
    for i, s in enumerate(states):
        corrected_counts[s] = int(round(p_corr[i] * total))

    return corrected_counts

def batch_noise_scan(n, target, optimal_k, p_values, repeat=5, shots=1024):
    """批量噪声扫描，返回 (p, mean, std) 列表"""
    results = []
    target_str = format(target, f'0{n}b')
    for p in p_values:
        success_list = []
        for _ in range(repeat):
            cnt = run_noisy(n, target, optimal_k, p, shots)
            success_list.append(cnt.get(target_str, 0) / shots)
        mean = np.mean(success_list)
        std = np.std(success_list)
        results.append((p, mean, std))
    return results

# ---------- 绘图辅助函数 ----------
def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
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
    fig, ax = plt.subplots(figsize=(4,3.5))
    states = sorted(counts.keys())
    vals = [counts[s] for s in states]
    ax.bar(states, vals, color='#1f77b4', edgecolor='white')
    ax.set_title('Measurement Distribution'); ax.set_ylabel('Counts')
    return fig_to_img(fig)

def plot_success_curve(rates):
    fig, ax = plt.subplots(figsize=(4,3.5))
    ks = list(rates.keys()); vs = list(rates.values())
    ax.plot(ks, vs, 'o-', color='purple', markersize=8, linewidth=2)
    ax.set_xlabel('Iterations k'); ax.set_ylabel('Success Rate')
    ax.set_xticks(ks); ax.grid(True, alpha=0.3); ax.set_ylim(0,1.05)
    return fig_to_img(fig)

def plot_amplitude(counts, n):
    fig, ax = plt.subplots(figsize=(4,3.5))
    total = sum(counts.values())
    binary_states = [format(i, f'0{n}b') for i in range(2**n)]
    amps = [np.sqrt(counts.get(s, 0) / total) for s in binary_states]
    labels = [f'|{s}⟩' for s in binary_states]
    ax.bar(labels, amps, color='orange', edgecolor='white')
    ax.set_title('Probability Amplitudes'); ax.set_ylabel('Magnitude')
    return fig_to_img(fig)

def plot_noise_curve(data, N):
    fig, ax = plt.subplots(figsize=(4,3.5))
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
    fig, ax = plt.subplots(figsize=(4,3.5))
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
    st.subheader("📊 Batch Noise Scan")
    batch_btn = st.button("🚀 Run Batch Scan (p=0~0.2)", use_container_width=True)

    st.markdown("---")
    st.subheader("📉 Error Mitigation")
    mit_btn = st.button("🛡️ Compare Mitigation", use_container_width=True)

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

rates = {}
for i in range(4):
    cnt = run_ideal(n, target, i)
    rates[i] = cnt.get(target_str, 0) / 1024

if noisy_btn:
    cnt_n = run_noisy(n, target, optimal_k, noise_p)
    succ_n = cnt_n.get(target_str, 0) / 1024
    st.session_state.noise_data.append((noise_p, succ_n, 0))
    st.sidebar.success(f"p={noise_p:.2f}, success={succ_n:.1%}")

if clear_btn:
    st.session_state.noise_data.clear()

if batch_btn:
    p_values = np.arange(0.0, 0.21, 0.02)
    noise_data = batch_noise_scan(n, target, optimal_k, p_values)
    st.session_state.noise_data = noise_data
    st.sidebar.success("✅ Batch scan complete!")

if mit_btn:
    readout_p = 0.1  # 固定 10% 读出错误率
    cnt_noisy = run_readout_noisy(n, target, optimal_k, readout_p)
    cnt_mit = apply_readout_mitigation(cnt_noisy, readout_p)
    succ_noisy = cnt_noisy.get(target_str, 0) / 1024
    total_mit = sum(cnt_mit.values())
    succ_mit = cnt_mit.get(target_str, 0) / total_mit if total_mit > 0 else 0
    st.session_state.mit_result = (succ_noisy, succ_mit, readout_p)
    st.sidebar.success(f"Mitigation: {succ_noisy:.1%} → {succ_mit:.1%}")

# 计算指标
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
left1, right1 = st.columns(2, gap="small")
with left1:
    st.subheader("🔵 Bloch Disk")
    st.image(plot_bloch(n, k), width=380)
with right1:
    st.subheader("📊 Measurement Distribution")
    st.image(plot_counts(counts), width=380)

# 第二行图表
left2, right2 = st.columns(2, gap="small")
with left2:
    st.subheader("📈 Success Rate vs Iterations")
    st.image(plot_success_curve(rates), width=380)
with right2:
    st.subheader("🌪️ Noise Impact Curve")
    st.image(plot_noise_curve(st.session_state.noise_data, N), width=380)

# 第三行图表
left3, right3 = st.columns(2, gap="small")
with left3:
    st.subheader("🔬 Probability Amplitudes")
    st.image(plot_amplitude(counts, n), width=380)
with right3:
    st.subheader("⚖️ Complexity Comparison")
    st.image(plot_complexity(N, optimal_k), width=380)

# 误差缓解结果展示
if 'mit_result' in st.session_state and st.session_state.mit_result is not None:
    st.markdown("---")
    st.subheader("📉 Measurement Error Mitigation Result")
    succ_noisy, succ_mit, rp = st.session_state.mit_result
    fig, ax = plt.subplots(figsize=(4,3))
    labels = ['Noisy', 'Mitigated']
    values = [succ_noisy, succ_mit]
    bars = ax.bar(labels, values, color=['#D32F2F', '#4CAF50'], edgecolor='white')
    ax.set_ylabel('Success Rate')
    ax.set_ylim(0,1.05)
    ax.set_title(f'Readout Error p={rp:.2f}')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f'{val:.1%}', ha='center')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    st.image(buf, width=400)

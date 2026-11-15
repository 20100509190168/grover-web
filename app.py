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
from qiskit_aer.noise import (
    NoiseModel, depolarizing_error, ReadoutError,
    thermal_relaxation_error
)

# ---------- 页面设置 ----------
st.set_page_config(page_title="Grover Quantum Search", layout="wide", page_icon="🔍")
st.title("🔍 Grover Quantum Search Algorithm Visualization")
st.markdown("---")

# ---------- 线路构建（支持 2/3/4 比特） ----------
def build_grover_circuit(n, target, k):
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
        else:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        for q, bit in enumerate(target_bin):
            if bit == '0':
                qc.x(q)
        # Diffusion
        for q in range(n):
            qc.h(q)
            qc.x(q)
        if n == 2:
            qc.cz(0, 1)
        else:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        for q in range(n):
            qc.x(q)
            qc.h(q)
    qc.measure(range(n), range(n))
    return qc

# ---------- 理想模拟 ----------
@st.cache_data(show_spinner=False)
def run_ideal(n, target, k, shots=1024):
    qc = build_grover_circuit(n, target, k)
    sim = AerSimulator()
    counts = sim.run(qc, shots=shots).result().get_counts()
    return counts

# ---------- 物理复合噪声：热弛豫 + 读出误差 ----------
def build_physical_noise(noise_p, n, gate_time_1q=50.0, gate_time_2q=300.0):
    T1_ideal, T2_ideal = 80.0, 60.0
    T1 = T1_ideal / (1 + noise_p * 8.0)
    T2 = min(T2_ideal / (1 + noise_p * 8.0), 2 * T1)

    nm = NoiseModel()

    # 单比特门
    err_1q = thermal_relaxation_error(T1, T2, gate_time_1q)
    nm.add_all_qubit_quantum_error(err_1q, ['h', 'x', 'z'])

    # 双比特门
    err_2q = thermal_relaxation_error(T1, T2, gate_time_2q).tensor(
        thermal_relaxation_error(T1, T2, gate_time_2q))
    nm.add_all_qubit_quantum_error(err_2q, ['cz', 'cx'])

    # 三比特门
    err_3q = thermal_relaxation_error(T1, T2, gate_time_2q).tensor(
        thermal_relaxation_error(T1, T2, gate_time_2q)).tensor(
        thermal_relaxation_error(T1, T2, gate_time_2q))
    nm.add_all_qubit_quantum_error(err_3q, ['ccx'])

    # 读出误差
    readout_p = min(noise_p * 0.4, 0.15)
    if readout_p > 0:
        nm.add_all_qubit_readout_error(
            ReadoutError([[1 - readout_p, readout_p],
                          [readout_p, 1 - readout_p]])
        )
    return nm

def run_noisy(n, target, k, noise_p, shots=1024):
    qc = build_grover_circuit(n, target, k)
    nm = build_physical_noise(noise_p, n)
    sim = AerSimulator(noise_model=nm)
    counts = sim.run(qc, shots=shots).result().get_counts()
    return counts

# ---------- 仅读出噪声（用于误差缓解对比） ----------
def run_readout_noisy(n, target, k, readout_p, shots=1024):
    qc = build_grover_circuit(n, target, k)
    nm = NoiseModel()
    if readout_p > 0:
        nm.add_all_qubit_readout_error(
            ReadoutError([[1 - readout_p, readout_p],
                          [readout_p, 1 - readout_p]])
        )
    sim = AerSimulator(noise_model=nm)
    counts = sim.run(qc, shots=shots).result().get_counts()
    return counts

def apply_readout_mitigation(counts, readout_p):
    n = len(next(iter(counts.keys())))
    dim = 2 ** n
    M1 = np.array([[1 - readout_p, readout_p],
                   [readout_p, 1 - readout_p]])
    M = M1.copy()
    for _ in range(n - 1):
        M = np.kron(M, M1)
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return counts

    states = [format(i, f'0{n}b') for i in range(dim)]
    total = sum(counts.values())
    p_meas = np.array([counts.get(s, 0) / total for s in states])
    p_corr = np.clip(M_inv @ p_meas, 0, None)
    p_sum = p_corr.sum()
    if p_sum > 0:
        p_corr = p_corr / p_sum
    return {s: int(round(p_corr[i] * total)) for i, s in enumerate(states)}

# ---------- 保真度计算 ----------
def state_fidelity(counts_noisy, counts_ideal):
    states = list(counts_ideal.keys())
    total_i = sum(counts_ideal.values())
    total_n = sum(counts_noisy.values())
    fid = 0.0
    for s in states:
        pi = counts_ideal.get(s, 0) / total_i
        pn = counts_noisy.get(s, 0) / total_n
        fid += np.sqrt(pi * pn)
    return fid

# ---------- 批量噪声扫描 ----------
@st.cache_data(show_spinner=False)
def batch_noise_scan(n, target, optimal_k, p_values_tuple, repeat=20, shots=512):
    p_values = list(p_values_tuple)
    target_str = format(target, f'0{n}b')
    ideal_counts = run_ideal(n, target, optimal_k, shots)
    results = []
    for p in p_values:
        success_list = []
        agg_counts = {}
        for _ in range(repeat):
            cnt = run_noisy(n, target, optimal_k, p, shots)
            success_list.append(cnt.get(target_str, 0) / shots)
            for s, v in cnt.items():
                agg_counts[s] = agg_counts.get(s, 0) + v
        mean = np.mean(success_list)
        std = np.std(success_list)
        fid = state_fidelity(agg_counts, ideal_counts)
        results.append((p, mean, std, fid))
    return results

# ---------- 收敛性诊断 ----------
@st.cache_data(show_spinner=False)
def convergence_diagnostic(n, target, optimal_k, p_fixed=0.08,
                           M_list=(5, 10, 20, 50, 100), shots=512):
    target_str = format(target, f'0{n}b')
    results = []
    for M in M_list:
        successes = []
        for _ in range(M):
            cnt = run_noisy(n, target, optimal_k, p_fixed, shots)
            successes.append(cnt.get(target_str, 0) / shots)
        mean = np.mean(successes)
        boot = [np.random.choice(successes, size=M, replace=True).mean()
                for _ in range(500)]
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        results.append((M, mean, ci_low, ci_high))
    return results

# ---------- 绘图辅助 ----------
def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_bloch(n, k):
    N = 2 ** n
    theta = np.arcsin(1 / np.sqrt(N))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.add_artist(plt.Circle((0, 0), 1, fill=False, color='gray'))
    ax.axhline(0, color='gray', alpha=0.3)
    ax.axvline(0, color='gray', alpha=0.3)
    ax.arrow(0, 0, 0, 1, head_width=0.06, fc='blue')
    ax.text(0.1, 1.05, '|t⟩', color='blue')
    s_angle = np.pi / 2 - theta
    ax.arrow(0, 0, np.cos(s_angle), np.sin(s_angle), head_width=0.06,
             fc='green', linestyle='dashed')
    ax.text(np.cos(s_angle) * 1.1, np.sin(s_angle) * 1.1, '|s⟩', color='green')
    cur_angle = np.pi / 2 - (theta + 2 * theta * k)
    ax.arrow(0, 0, np.cos(cur_angle), np.sin(cur_angle), head_width=0.08,
             fc='red', linewidth=2)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_title(f'Bloch Disk (k={k})')
    return fig_to_img(fig)

def plot_counts(counts):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    states = sorted(counts.keys())
    vals = [counts[s] for s in states]
    ax.bar(states, vals, color='#1f77b4', edgecolor='white')
    ax.set_title('Measurement Distribution'); ax.set_ylabel('Counts')
    plt.xticks(rotation=45 if len(states) > 8 else 0)
    return fig_to_img(fig)

def plot_success_curve(rates):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ks = list(rates.keys()); vs = list(rates.values())
    ax.plot(ks, vs, 'o-', color='purple', markersize=8, linewidth=2)
    ax.set_xlabel('Iterations k'); ax.set_ylabel('Success Rate')
    ax.set_xticks(ks); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)
    return fig_to_img(fig)

def plot_amplitude(counts, n):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    total = sum(counts.values())
    binary_states = [format(i, f'0{n}b') for i in range(2 ** n)]
    amps = [np.sqrt(counts.get(s, 0) / total) for s in binary_states]
    labels = [f'|{s}⟩' for s in binary_states]
    ax.bar(labels, amps, color='orange', edgecolor='white')
    ax.set_title('Probability Amplitudes'); ax.set_ylabel('Magnitude')
    plt.xticks(rotation=45 if len(labels) > 8 else 0)
    return fig_to_img(fig)

def plot_noise_curve(data, N):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.set_xlim(0, 0.22); ax.set_ylim(0, 1.05)
    ax.set_xlabel('Noise Strength p'); ax.set_ylabel('Success Rate')
    if data:
        ps = [d[0] for d in data]
        means = [d[1] for d in data]
        stds = [d[2] for d in data]
        ax.errorbar(ps, means, yerr=stds, fmt='o-', color='red', capsize=3)
        ax.axhline(y=1 / N, color='gray', linestyle='--', label='Random Guess')
        ax.legend(); ax.grid(True, alpha=0.3)
    return fig_to_img(fig)

def plot_complexity(N, optimal_k):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    labels = ['Classical\n(worst case)', 'Grover\n(optimal)']
    values = [N - 1, optimal_k]
    bars = ax.bar(labels, values, color=['gray', 'red'], edgecolor='white')
    ax.set_ylabel('Queries / Iterations')
    ax.set_title('Quantum Speedup')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(val), ha='center', fontsize=12)
    return fig_to_img(fig)

def plot_advantage_ratio(data, n, optimal_k):
    N = 2 ** n
    classical_bound = optimal_k / N
    fig, ax = plt.subplots(figsize=(4, 3.5))
    if data:
        ps = [d[0] for d in data]
        ratios = [d[1] / classical_bound for d in data]
        ax.plot(ps, ratios, 's-', color='#6A1B9A', markersize=5)
        ax.axhline(1.0, color='gray', linestyle='--', label='Classical bound')
        ax.legend()
    ax.set_xlabel('Noise p'); ax.set_ylabel('Quantum / Classical')
    ax.set_title(f'Advantage Ratio (k={optimal_k}, N={N})')
    ax.grid(True, alpha=0.3)
    return fig_to_img(fig)

def plot_fidelity_curve(data):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    if data:
        ps = [d[0] for d in data]
        fids = [d[3] for d in data]
        ax.plot(ps, fids, 'd-', color='#00897B', markersize=6)
    ax.set_xlabel('Noise p'); ax.set_ylabel('State Fidelity')
    ax.set_title('Output Fidelity vs Noise')
    ax.grid(True, alpha=0.3)
    return fig_to_img(fig)

def plot_convergence(results):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    if results:
        Ms = [r[0] for r in results]
        widths = [(r[3] - r[2]) / 2 for r in results]
        ax.plot(Ms, widths, 'o-', color='#E64A19')
        ax.set_xscale('log')
    ax.set_xlabel('Repeat count M'); ax.set_ylabel('95% CI half-width')
    ax.set_title('Convergence Diagnostic')
    ax.grid(True, alpha=0.3)
    return fig_to_img(fig)

# ---------- 状态管理 ----------
if 'k' not in st.session_state:
    st.session_state.k = 0
if 'noise_data' not in st.session_state:
    st.session_state.noise_data = []
if 'conv_result' not in st.session_state:
    st.session_state.conv_result = None
if 'auto_playing' not in st.session_state:
    st.session_state.auto_playing = False

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("⚙️ Configuration")
    n = st.selectbox("Number of Qubits", [2, 3, 4], index=0)
    N = 2 ** n
    target = st.selectbox("Target State", list(range(N)),
                          format_func=lambda x: f"|{x:0{n}b}⟩")
    optimal_k = int(np.floor(np.pi / 4 * np.sqrt(N)))

    st.markdown("---")
    st.subheader("🎮 Ideal Iteration")
    col1, col2 = st.columns(2)
    with col1:
        step = st.button("➕ Step", use_container_width=True)
    with col2:
        reset = st.button("🔄 Reset", use_container_width=True)
    
    auto_btn = st.button("▶️ Auto Play", use_container_width=True)

    st.markdown("---")
    st.subheader("🌪️ Noise Experiment (Manual)")
    noise_p = st.slider("Physical Noise p", 0.0, 0.2, 0.0, 0.01)
    noisy_btn = st.button("🔬 Run Noisy", use_container_width=True)
    clear_btn = st.button("🗑️ Clear Noise Data", use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Batch Noise Scan")
    repeat_m = st.selectbox("Repeat M (for Batch)", [5, 10, 20, 50], index=2)
    batch_btn = st.button("🚀 Run Batch Scan (p=0~0.2)", use_container_width=True)

    st.markdown("---")
    st.subheader("📉 Error Mitigation")
    mit_btn = st.button("🛡️ Compare Mitigation", use_container_width=True)

    st.markdown("---")
    st.subheader("📐 Convergence Check")
    conv_btn = st.button("🔍 Run Convergence (p=0.08)", use_container_width=True)

    st.markdown("---")
    st.info(f"💡 Optimal iterations: **{optimal_k}**")

# ---------- 按钮逻辑 ----------
if step or reset:
    st.session_state.auto_playing = False

if step:
    st.session_state.k += 1
if reset:
    st.session_state.k = 0
if auto_btn:
    st.session_state.auto_playing = True

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
    ideal_counts = run_ideal(n, target, optimal_k, 1024)
    fid = state_fidelity(cnt_n, ideal_counts)
    st.session_state.noise_data.append((noise_p, succ_n, 0, fid))
    st.sidebar.success(f"p={noise_p:.2f}, success={succ_n:.1%}, fid={fid:.3f}")

if clear_btn:
    st.session_state.noise_data.clear()
    st.session_state.conv_result = None

if batch_btn:
    p_values = tuple(np.round(np.arange(0.0, 0.21, 0.02), 2))
    with st.spinner("Running batch noise scan..."):
        noise_data = batch_noise_scan(n, target, optimal_k, p_values,
                                      repeat=repeat_m)
    st.session_state.noise_data = noise_data
    st.sidebar.success("✅ Batch scan complete!")

if mit_btn:
    readout_p = 0.1
    cnt_noisy = run_readout_noisy(n, target, optimal_k, readout_p)
    cnt_mit = apply_readout_mitigation(cnt_noisy, readout_p)
    succ_noisy = cnt_noisy.get(target_str, 0) / 1024
    total_mit = sum(cnt_mit.values())
    succ_mit = cnt_mit.get(target_str, 0) / total_mit if total_mit > 0 else 0
    st.session_state.mit_result = (succ_noisy, succ_mit, readout_p)
    st.sidebar.success(f"Mitigation: {succ_noisy:.1%} → {succ_mit:.1%}")

if conv_btn:
    with st.spinner("Running convergence diagnostic..."):
        conv = convergence_diagnostic(n, target, optimal_k, p_fixed=0.08,
                                      M_list=(5, 10, 20, 50, 100))
    st.session_state.conv_result = conv
    st.sidebar.success("✅ Convergence diagnostic complete!")

# ---------- 指标卡片 ----------
theta_deg = np.degrees(np.arcsin(1 / np.sqrt(N)))
angle = 2 * theta_deg * k
theory = np.sin((2 * k + 1) * np.arcsin(1 / np.sqrt(N))) ** 2

c1, c2, c3, c4 = st.columns(4)
c1.metric("Iteration k", k)
c2.metric("Total Rotation", f"{angle:.1f}°")
c3.metric("Theoretical Success", f"{theory:.1%}")
c4.metric("Measured Success", f"{success:.1%}")

st.markdown("---")

# ---------- 第一行 ----------
left1, right1 = st.columns(2, gap="small")
with left1:
    st.subheader("🔵 Bloch Disk")
    st.image(plot_bloch(n, k), width=380)
with right1:
    st.subheader("📊 Measurement Distribution")
    st.image(plot_counts(counts), width=380)

# ---------- 第二行 ----------
left2, right2 = st.columns(2, gap="small")
with left2:
    st.subheader("📈 Success Rate vs Iterations")
    st.image(plot_success_curve(rates), width=380)
with right2:
    st.subheader("🌪️ Noise Impact Curve")
    st.image(plot_noise_curve(st.session_state.noise_data, N), width=380)

# ---------- 第三行 ----------
left3, right3 = st.columns(2, gap="small")
with left3:
    st.subheader("🔬 Probability Amplitudes")
    st.image(plot_amplitude(counts, n), width=380)
with right3:
    st.subheader("⚖️ Complexity Comparison")
    st.image(plot_complexity(N, optimal_k), width=380)

# ---------- 第四行：高级指标 ----------
left4, right4 = st.columns(2, gap="small")
with left4:
    st.subheader("📐 Quantum / Classical Advantage Ratio")
    st.image(plot_advantage_ratio(st.session_state.noise_data, n, optimal_k),
             width=380)
with right4:
    st.subheader("🎯 State Fidelity vs Noise")
    st.image(plot_fidelity_curve(st.session_state.noise_data), width=380)

# ---------- 误差缓解结果 ----------
if 'mit_result' in st.session_state and st.session_state.mit_result is not None:
    st.markdown("---")
    st.subheader("📉 Measurement Error Mitigation Result")
    succ_noisy, succ_mit, rp = st.session_state.mit_result
    fig, ax = plt.subplots(figsize=(4, 3))
    labels = ['Noisy', 'Mitigated']
    values = [succ_noisy, succ_mit]
    bars = ax.bar(labels, values, color=['#D32F2F', '#4CAF50'], edgecolor='white')
    ax.set_ylabel('Success Rate')
    ax.set_ylim(0, 1.05)
    ax.set_title(f'Readout Error p={rp:.2f}')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    st.image(buf, width=400)

# ---------- 收敛性诊断结果 ----------
if st.session_state.conv_result is not None:
    st.markdown("---")
    st.subheader("📐 Convergence Diagnostic (p=0.08)")
    st.image(plot_convergence(st.session_state.conv_result), width=420)
    st.caption("横轴为重复次数 M，纵轴为成功率 95% 置信区间半宽。曲线趋于平稳说明样本量已足够。")


# ---------- 自动播放核心逻辑（必须放在文件最底部！） ----------
if st.session_state.auto_playing:
    if st.session_state.k < optimal_k + 1:
        st.info(f"▶️ Auto playing... 当前迭代次数 k = {st.session_state.k}，请稍候...")
        time.sleep(0.8)  # 停留 0.8 秒，播放更连贯
        st.session_state.k += 1
        st.rerun()
    else:
        st.session_state.auto_playing = False
        st.sidebar.success("✅ Auto Play Finished!")
        st.rerun()

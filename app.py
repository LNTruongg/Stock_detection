import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime
import base64

# ============================================================
# CẤU HÌNH TRANG
# ============================================================
st.set_page_config(
    page_title="📈 Anomaly Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# TIÊU ĐỀ & CSS TÙY CHỈNH
# ============================================================
st.markdown("""
<style>
    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        padding: 1.2rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: #fff;
        font-weight: 700;
        font-size: 2.4rem;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .main-header p {
        color: #a0c4ff;
        margin: 4px 0 0 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    /* Cards */
    .stat-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(8px);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        transition: 0.2s;
    }
    .stat-card:hover {
        transform: translateY(-3px);
        border-color: rgba(100, 180, 255, 0.3);
    }
    .stat-card .label {
        color: #aaa;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-card .value {
        color: #fff;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .stat-card .sub {
        color: #8ab4f8;
        font-size: 0.9rem;
    }
    /* Severity badges */
    .badge-high {
        background: #e53935;
        color: #fff;
        padding: 2px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-medium {
        background: #fb8c00;
        color: #fff;
        padding: 2px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-low {
        background: #f6d32d;
        color: #1a1a2e;
        padding: 2px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-normal {
        background: #2a2a3e;
        color: #aaa;
        padding: 2px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255,255,255,0.06);
        text-align: center;
        color: #555;
        font-size: 0.85rem;
    }
    /* Tabs style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background: rgba(255,255,255,0.03);
        padding: 8px 16px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        color: #aaa;
        padding: 8px 16px;
        border-radius: 8px;
        transition: 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #fff;
        background: rgba(255,255,255,0.05);
    }
    .stTabs [aria-selected="true"] {
        color: #fff !important;
        background: rgba(66, 165, 245, 0.2) !important;
    }
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: #2a3a5a;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #3a5a8a;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# TIÊU ĐỀ
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>📈 Anomaly Detection Dashboard</h1>
    <p>Phân tích bất thường giao dịch cổ phiếu &nbsp;•&nbsp; Dữ liệu theo ngày</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("df_final_output.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    try:
        clusters = pd.read_csv("clusters.csv")
        clusters['start_date'] = pd.to_datetime(clusters['start_date'])
        clusters['end_date'] = pd.to_datetime(clusters['end_date'])
        has_clusters = True
    except:
        clusters = pd.DataFrame()
        has_clusters = False

    try:
        rule = pd.read_csv("df_rule.csv")
        rule['Date'] = pd.to_datetime(rule['Date'])
    except:
        rule = pd.DataFrame()

    try:
        ml = pd.read_csv("df_ml.csv")
        ml['Date'] = pd.to_datetime(ml['Date'])
    except:
        ml = pd.DataFrame()

    return df, clusters, rule, ml, has_clusters

df, clusters, rule, ml, has_clusters = load_data()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Điều khiển")

    # Filter by severity
    severity_filter = st.multiselect(
        "Mức độ bất thường",
        options=['High', 'Medium', 'Low', 'Normal'],
        default=['High', 'Medium', 'Low']
    )

    # Date range
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    date_range = st.date_input(
        "Khoảng thời gian",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Apply filters
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask_date = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
    else:
        mask_date = pd.Series([True] * len(df))

    mask_sev = df['severity'].isin(severity_filter)
    df_filtered = df[mask_date & mask_sev].copy()

    # Stats in sidebar
    st.markdown("---")
    st.markdown("### 📊 Thống kê nhanh")
    n_yes = len(df_filtered[df_filtered['label'] == 'Yes'])
    n_total = len(df_filtered)
    pct = n_yes / n_total * 100 if n_total else 0

    col1, col2 = st.columns(2)
    col1.metric("Ngày bất thường", f"{n_yes:,}")
    col2.metric("Tỉ lệ", f"{pct:.1f}%")

    st.caption(f"Tổng ngày: {n_total:,}")

# ============================================================
# MAIN DASHBOARD
# ============================================================

# ---- HÀNG 1: 4 thẻ chỉ số ----
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat-card">
        <div class="label">🔴 Mức High</div>
        <div class="value">{}</div>
        <div class="sub">Ngày bất thường nghiêm trọng</div>
    </div>
    """.format(len(df_filtered[df_filtered['severity'] == 'High'])), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-card">
        <div class="label">🟠 Mức Medium</div>
        <div class="value">{}</div>
        <div class="sub">Ngày bất thường trung bình</div>
    </div>
    """.format(len(df_filtered[df_filtered['severity'] == 'Medium'])), unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-card">
        <div class="label">🟡 Mức Low</div>
        <div class="value">{}</div>
        <div class="sub">Ngày bất thường nhẹ</div>
    </div>
    """.format(len(df_filtered[df_filtered['severity'] == 'Low'])), unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-card">
        <div class="label">📈 Điểm TB</div>
        <div class="value">{:.3f}</div>
        <div class="sub">Anomaly score trung bình</div>
    </div>
    """.format(df_filtered['anomaly_score'].mean()), unsafe_allow_html=True)


# ---- HÀNG 2: Biểu đồ giá & Anomaly ----
st.markdown("---")
st.markdown("### 📉 Giá đóng cửa & Các điểm bất thường")

# Chuẩn bị dữ liệu cho plotly
df_plot = df_filtered.copy()
df_plot['date_str'] = df_plot['Date'].dt.strftime('%Y-%m-%d')

# Tạo figure
fig = go.Figure()

# Line giá
fig.add_trace(go.Scatter(
    x=df_plot['Date'],
    y=df_plot['Close'],
    mode='lines',
    name='Giá đóng cửa',
    line=dict(color='#42a5f5', width=2),
    hovertemplate='%{x}<br>Giá: %{y:.2f}<extra></extra>'
))

# Thêm vùng cluster nếu có
if has_clusters and not clusters.empty:
    for _, row in clusters.iterrows():
        fig.add_vrect(
            x0=row['start_date'],
            x1=row['end_date'],
            fillcolor='rgba(144, 202, 249, 0.12)',
            line_width=0,
            layer='below',
            annotation_text=f"Cluster {row['cluster_id']}",
            annotation_position="top left",
            annotation_font_size=10,
            annotation_font_color='#8ab4f8'
        )

# Scatter các điểm bất thường theo severity
colors_sev = {'High': '#e53935', 'Medium': '#fb8c00', 'Low': '#f6d32d'}
for sev in ['High', 'Medium', 'Low']:
    tmp = df_plot[df_plot['severity'] == sev]
    if not tmp.empty:
        fig.add_trace(go.Scatter(
            x=tmp['Date'],
            y=tmp['Close'],
            mode='markers',
            name=sev,
            marker=dict(
                size=10,
                color=colors_sev[sev],
                symbol='circle',
                line=dict(width=1, color='#fff')
            ),
            hovertemplate='%{x}<br>Giá: %{y:.2f}<br>Điểm: %{customdata:.3f}<extra></extra>',
            customdata=tmp['anomaly_score']
        ))

fig.update_layout(
    template='plotly_dark',
    xaxis=dict(title='Ngày', showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
    yaxis=dict(title='Giá đóng cửa', showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
    hovermode='x unified',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    margin=dict(l=20, r=20, t=20, b=20),
    height=450,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


# ---- HÀNG 3: 2 biểu đồ cạnh nhau ----
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🏆 Top 15 ngày bất thường nhất")
    top15 = df_filtered.nlargest(15, 'anomaly_score').sort_values('anomaly_score')
    top15['label_date'] = top15['Date'].dt.strftime('%d/%m/%Y')

    fig_top = go.Figure()
    fig_top.add_trace(go.Bar(
        y=top15['label_date'],
        x=top15['anomaly_score'],
        orientation='h',
        marker=dict(
            color=top15['anomaly_score'],
            colorscale='Reds',
            showscale=False,
            line=dict(width=0)
        ),
        text=top15['anomaly_score'].round(3),
        textposition='outside',
        textfont=dict(size=11, color='#ccc'),
        hovertemplate='<b>%{y}</b><br>Điểm: %{x:.3f}<extra></extra>'
    ))

    fig_top.update_layout(
        template='plotly_dark',
        xaxis=dict(title='Anomaly Score', showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title='', showgrid=False),
        height=400,
        margin=dict(l=10, r=30, t=10, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_top, use_container_width=True, config={'displayModeBar': False})

with col2:
    st.markdown("### 📊 Phân bố mức độ bất thường")
    sev_counts = df_filtered['severity'].value_counts().reindex(['High', 'Medium', 'Low', 'Normal'], fill_value=0)

    fig_pie = go.Figure(data=[go.Pie(
        labels=['High', 'Medium', 'Low', 'Normal'],
        values=sev_counts.values,
        marker=dict(colors=['#e53935', '#fb8c00', '#f6d32d', '#2a2a3e']),
        textinfo='label+percent',
        textposition='inside',
        hole=0.45,
        pull=[0.05, 0.03, 0.02, 0],
        showlegend=False
    )])

    fig_pie.update_layout(
        template='plotly_dark',
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        annotations=[dict(text=f"{len(df_filtered[df_filtered['label']=='Yes'])}", x=0.5, y=0.5, font_size=28, font_color='#fff', showarrow=False)]
    )
    st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})


# ---- HÀNG 4: Heatmap lịch ----
st.markdown("---")
st.markdown("### 📅 Bản đồ nhiệt: Ngày trong tháng × Tháng")

# Tạo pivot
df_cal = df_filtered.copy()
df_cal['Tháng'] = df_cal['Date'].dt.month
df_cal['Ngày'] = df_cal['Date'].dt.day
pivot = df_cal.pivot_table(index='Ngày', columns='Tháng', values='anomaly_score', aggfunc='mean')

fig_heat = go.Figure(data=go.Heatmap(
    z=pivot.values,
    x=pivot.columns,
    y=pivot.index,
    colorscale='YlOrRd',
    zmin=0,
    zmax=pivot.values.max() if pivot.values.size > 0 else 1,
    hovertemplate='Ngày %{y}, Tháng %{x}<br>Điểm TB: %{z:.3f}<extra></extra>'
))

fig_heat.update_layout(
    template='plotly_dark',
    xaxis=dict(title='Tháng', tickmode='linear', tick0=1, dtick=1),
    yaxis=dict(title='Ngày trong tháng', tickmode='linear', tick0=1, dtick=2),
    height=420,
    margin=dict(l=20, r=20, t=20, b=20),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)
st.plotly_chart(fig_heat, use_container_width=True, config={'displayModeBar': False})


# ---- HÀNG 5: Heatmap ret_z vs vol_z ----
st.markdown("---")
st.markdown("### 🔥 Tương quan: Biến động giá (ret_z) vs Biến động khối lượng (vol_z)")

# Lọc dữ liệu
df_scatter = df_filtered[['ret_z', 'vol_z', 'anomaly_score', 'severity']].dropna()
df_scatter = df_scatter[(df_scatter['ret_z'].abs() < 5) & (df_scatter['vol_z'] < 5)]

if not df_scatter.empty:
    fig_scatter = go.Figure()

    # Density heatmap
    fig_scatter.add_trace(go.Histogram2dContour(
        x=df_scatter['ret_z'],
        y=df_scatter['vol_z'],
        colorscale='Blues',
        contours=dict(coloring='heatmap'),
        hovertemplate='ret_z: %{x:.2f}<br>vol_z: %{y:.2f}<br>Mật độ: %{z}<extra></extra>'
    ))

    # Đường ngưỡng
    for thr in [1.5, 2, 3]:
        fig_scatter.add_hline(y=thr, line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dash'))
        fig_scatter.add_hline(y=-thr, line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dash'))
        fig_scatter.add_vline(x=thr, line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dash'))
        fig_scatter.add_vline(x=-thr, line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dash'))

    # Scatter điểm bất thường
    tmp_high = df_scatter[df_scatter['severity'] == 'High']
    tmp_med = df_scatter[df_scatter['severity'] == 'Medium']
    tmp_low = df_scatter[df_scatter['severity'] == 'Low']

    fig_scatter.add_trace(go.Scatter(
        x=tmp_high['ret_z'], y=tmp_high['vol_z'],
        mode='markers',
        name='High',
        marker=dict(color='#e53935', size=8, line=dict(width=1, color='#fff')),
        hovertemplate='ret_z: %{x:.2f}<br>vol_z: %{y:.2f}<extra></extra>'
    ))
    fig_scatter.add_trace(go.Scatter(
        x=tmp_med['ret_z'], y=tmp_med['vol_z'],
        mode='markers',
        name='Medium',
        marker=dict(color='#fb8c00', size=7, line=dict(width=1, color='#fff')),
        hovertemplate='ret_z: %{x:.2f}<br>vol_z: %{y:.2f}<extra></extra>'
    ))
    fig_scatter.add_trace(go.Scatter(
        x=tmp_low['ret_z'], y=tmp_low['vol_z'],
        mode='markers',
        name='Low',
        marker=dict(color='#f6d32d', size=6, line=dict(width=1, color='#fff')),
        hovertemplate='ret_z: %{x:.2f}<br>vol_z: %{y:.2f}<extra></extra>'
    ))

    fig_scatter.update_layout(
        template='plotly_dark',
        xaxis=dict(title='ret_z (biến động giá)', showgrid=True, gridcolor='rgba(255,255,255,0.05)', range=[-4.5, 4.5]),
        yaxis=dict(title='vol_z (biến động khối lượng)', showgrid=True, gridcolor='rgba(255,255,255,0.05)', range=[-1, 4.5]),
        height=440,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})
else:
    st.info("Không đủ dữ liệu để hiển thị biểu đồ tương quan.")


# ---- HÀNG 6: Bảng dữ liệu ----
st.markdown("---")
with st.expander("📋 Xem dữ liệu chi tiết", expanded=False):
    tab1, tab2, tab3 = st.tabs(["📊 Dữ liệu bất thường", "📈 Dữ liệu gốc", "📦 Cluster"])

    with tab1:
        display_cols = ['Date', 'Close', 'anomaly_score', 'label', 'severity']
        st.dataframe(
            df_filtered[display_cols].sort_values('Date', ascending=False),
            use_container_width=True,
            height=350,
            column_config={
                'Date': st.column_config.DatetimeColumn('Ngày'),
                'Close': st.column_config.NumberColumn('Giá', format='%.2f'),
                'anomaly_score': st.column_config.NumberColumn('Điểm', format='%.4f'),
                'label': st.column_config.TextColumn('Label'),
                'severity': st.column_config.TextColumn('Mức độ')
            }
        )

    with tab2:
        if not rule.empty and not ml.empty:
            merged = df_filtered.merge(rule[['Date', 'rule_score']], on='Date', how='left')
            merged = merged.merge(ml[['Date', 'ml_score']], on='Date', how='left')
            cols = ['Date', 'Close', 'rule_score', 'ml_score', 'anomaly_score', 'label', 'severity']
            st.dataframe(merged[cols].sort_values('Date', ascending=False), use_container_width=True, height=300)

    with tab3:
        if has_clusters and not clusters.empty:
            st.dataframe(
                clusters.sort_values('cluster_id'),
                use_container_width=True,
                height=300,
                column_config={
                    'cluster_id': 'ID',
                    'start_date': st.column_config.DatetimeColumn('Bắt đầu'),
                    'end_date': st.column_config.DatetimeColumn('Kết thúc'),
                    'length_days': 'Số ngày',
                    'n_high_days': 'High',
                    'max_vol_z': 'Max vol_z',
                    'avg_ret_%': st.column_config.NumberColumn('Lợi nhuận TB %', format='%.2f'),
                    'post_3d_%': st.column_config.NumberColumn('Sau 3 ngày %', format='%.2f'),
                    'post_5d_%': st.column_config.NumberColumn('Sau 5 ngày %', format='%.2f'),
                    'pump_dump': 'Pump&Dump',
                }
            )
        else:
            st.info("Không có dữ liệu cluster.")


# ---- HÀNG 7: Kết luận tự động ----
st.markdown("---")
st.markdown("### 🧠 Tổng kết & Nhận định")

# Tính các chỉ số
n_yes = len(df_filtered[df_filtered['label'] == 'Yes'])
n_total = len(df_filtered)
pct_abnormal = n_yes / n_total * 100 if n_total else 0
high_n = len(df_filtered[df_filtered['severity'] == 'High'])
med_n = len(df_filtered[df_filtered['severity'] == 'Medium'])
low_n = len(df_filtered[df_filtered['severity'] == 'Low'])

# Xác định mức độ cảnh báo
if high_n > 25 or pct_abnormal > 25:
    level = "🔴 Nghiêm trọng"
    color = "#e53935"
    conclusion = "Cổ phiếu có dấu hiệu bị thao túng đáng kể. Cần đặc biệt thận trọng."
elif high_n > 15 or pct_abnormal > 15:
    level = "🟠 Cảnh báo"
    color = "#fb8c00"
    conclusion = "Cổ phiếu có thể đang bị thao túng nhẹ hoặc có nhiều biến động bất thường."
elif pct_abnormal > 5 or high_n > 5:
    level = "🟡 Theo dõi"
    color = "#f6d32d"
    conclusion = "Một vài dấu hiệu bất thường nhưng chưa đủ kết luận thao túng."
else:
    level = "🟢 Bình thường"
    color = "#4caf50"
    conclusion = "Chưa phát hiện dấu hiệu bất thường rõ ràng."

# Tạo badge
st.markdown(f"""
<div style="
    background: rgba(255,255,255,0.04);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    border-left: 6px solid {color};
    margin-top: 8px;
">
    <div style="display: flex; align-items: center; gap: 16px; flex-wrap: wrap;">
        <span style="font-size: 2rem;">{level.split()[0]}</span>
        <span style="font-weight: 700; font-size: 1.2rem; color: #fff;">{level}</span>
        <span style="color: #aaa; font-size: 0.95rem;">|</span>
        <span style="color: #ccc; font-size: 0.95rem;">{conclusion}</span>
    </div>
    <div style="margin-top: 12px; color: #888; font-size: 0.9rem;">
        • High: {high_n} ngày &nbsp;·&nbsp; Medium: {med_n} ngày &nbsp;·&nbsp; Low: {low_n} ngày &nbsp;·&nbsp;
        Tỉ lệ bất thường: {pct_abnormal:.1f}%
    </div>
</div>
""", unsafe_allow_html=True)


# ---- FOOTER ----
st.markdown("""
<div class="footer">
    Anomaly Detection Dashboard &bull; Dữ liệu được cập nhật tự động &bull; Built with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)
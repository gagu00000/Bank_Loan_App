import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             roc_curve, classification_report)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
import os

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Universal Bank - Loan Analytics Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# THEME MANAGEMENT
# -----------------------------------------------------------------------------

if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def toggle_theme():
    if st.session_state.theme == 'light':
        st.session_state.theme = 'dark'
    else:
        st.session_state.theme = 'light'

THEMES = {
    'light': {
        'bg_color': '#FFFFFF',
        'secondary_bg': '#F0F2F6',
        'text_color': '#1E1E1E',
        'card_bg': '#FFFFFF',
        'accent_color': '#3B82F6',
        'success_color': '#10B981',
        'warning_color': '#F59E0B',
        'error_color': '#EF4444',
        'border_color': '#E5E7EB',
        'header_color': '#1E3A8A',
        'subheader_color': '#3B82F6',
        'plotly_template': 'plotly_white',
        'chart_bg': 'rgba(255,255,255,1)',
        'grid_color': 'rgba(128,128,128,0.2)',
        'axis_text_color': '#000000',
        'axis_title_color': '#1E1E1E',
        'legend_text_color': '#1E1E1E',
        'title_color': '#1E3A8A'
    },
    'dark': {
        'bg_color': '#0E1117',
        'secondary_bg': '#262730',
        'text_color': '#FAFAFA',
        'card_bg': '#1E1E2E',
        'accent_color': '#60A5FA',
        'success_color': '#34D399',
        'warning_color': '#FBBF24',
        'error_color': '#F87171',
        'border_color': '#374151',
        'header_color': '#60A5FA',
        'subheader_color': '#93C5FD',
        'plotly_template': 'plotly_dark',
        'chart_bg': 'rgba(14,17,23,1)',
        'grid_color': 'rgba(128,128,128,0.3)',
        'axis_text_color': '#FFFFFF',
        'axis_title_color': '#FAFAFA',
        'legend_text_color': '#FAFAFA',
        'title_color': '#60A5FA'
    }
}

current_theme = THEMES[st.session_state.theme]

def apply_theme_css():
    theme = current_theme
    css = f"""
    <style>
    .stApp {{
        background-color: {theme['bg_color']};
    }}
    [data-testid="stSidebar"] {{
        background-color: {theme['secondary_bg']};
    }}
    [data-testid="stSidebar"] .stMarkdown {{
        color: {theme['text_color']};
    }}
    .main-header {{
        font-size: 2.5rem;
        font-weight: bold;
        color: {theme['header_color']};
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(90deg, {theme['secondary_bg']} 0%, {theme['bg_color']} 50%, {theme['secondary_bg']} 100%);
        border-radius: 10px;
    }}
    .sub-header {{
        font-size: 1.5rem;
        font-weight: bold;
        color: {theme['subheader_color']};
        margin-top: 1rem;
    }}
    [data-testid="stMetricValue"] {{
        color: {theme['text_color']};
    }}
    [data-testid="stMetricLabel"] {{
        color: {theme['text_color']};
    }}
    .stMarkdown, .stText {{
        color: {theme['text_color']};
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {theme['text_color']} !important;
    }}
    p, span, label {{
        color: {theme['text_color']};
    }}
    .metric-card {{
        background-color: {theme['card_bg']};
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid {theme['accent_color']};
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2rem;
        background-color: {theme['secondary_bg']};
        border-radius: 10px;
        padding: 0.5rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 3rem;
        padding-left: 1rem;
        padding-right: 1rem;
        color: {theme['text_color']};
        background-color: transparent;
        border-radius: 5px;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: {theme['accent_color']}33;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {theme['accent_color']};
        color: white !important;
    }}
    .stButton > button {{
        background-color: {theme['accent_color']};
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }}
    .stButton > button:hover {{
        background-color: {theme['header_color']};
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }}
    .streamlit-expanderHeader {{
        background-color: {theme['secondary_bg']};
        color: {theme['text_color']};
        border-radius: 5px;
    }}
    .streamlit-expanderContent {{
        background-color: {theme['card_bg']};
        border: 1px solid {theme['border_color']};
    }}
    .stDataFrame {{
        background-color: {theme['card_bg']};
    }}
    .stSelectbox, .stMultiSelect, .stSlider, .stNumberInput {{
        color: {theme['text_color']};
    }}
    .stAlert {{
        background-color: {theme['secondary_bg']};
        color: {theme['text_color']};
    }}
    .stProgress > div > div {{
        background-color: {theme['accent_color']};
    }}
    hr {{
        border-color: {theme['border_color']};
    }}
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: {theme['secondary_bg']};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {theme['accent_color']};
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {theme['header_color']};
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .element-container {{
        animation: fadeIn 0.5s ease-out;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

apply_theme_css()

# -----------------------------------------------------------------------------
# PLOTLY THEME HELPER
# -----------------------------------------------------------------------------

def apply_plotly_theme(fig):
    theme = current_theme
    
    fig.update_layout(
        template=theme['plotly_template'],
        paper_bgcolor=theme['chart_bg'],
        plot_bgcolor=theme['chart_bg'],
        font=dict(color=theme['text_color']),
        title_font_color=theme['title_color'],
        legend_font_color=theme['legend_text_color'],
        xaxis=dict(
            gridcolor=theme['grid_color'],
            zerolinecolor=theme['grid_color'],
            tickfont=dict(color=theme['axis_text_color']),
            title_font=dict(color=theme['axis_title_color'])
        ),
        yaxis=dict(
            gridcolor=theme['grid_color'],
            zerolinecolor=theme['grid_color'],
            tickfont=dict(color=theme['axis_text_color']),
            title_font=dict(color=theme['axis_title_color'])
        )
    )
    
    layout_dict = fig.to_dict()['layout']
    
    for key in layout_dict:
        if key.startswith('xaxis'):
            fig.update_layout(**{
                key: dict(
                    gridcolor=theme['grid_color'],
                    zerolinecolor=theme['grid_color'],
                    tickfont=dict(color=theme['axis_text_color']),
                    title_font=dict(color=theme['axis_title_color'])
                )
            })
        elif key.startswith('yaxis'):
            fig.update_layout(**{
                key: dict(
                    gridcolor=theme['grid_color'],
                    zerolinecolor=theme['grid_color'],
                    tickfont=dict(color=theme['axis_text_color']),
                    title_font=dict(color=theme['axis_title_color'])
                )
            })
    
    if 'annotations' in layout_dict:
        new_annotations = []
        for annotation in fig.layout.annotations:
            annotation_dict = annotation.to_plotly_json()
            annotation_dict['font'] = dict(color=theme['axis_title_color'])
            new_annotations.append(annotation_dict)
        fig.update_layout(annotations=new_annotations)
    
    return fig

# -----------------------------------------------------------------------------
# DATA LOADING FUNCTION
# -----------------------------------------------------------------------------

@st.cache_data
def load_data():
    column_names = ['ID', 'Age', 'Experience', 'Income', 'ZIPCode', 'Family', 
                    'CCAvg', 'Education', 'Mortgage', 'PersonalLoan', 
                    'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard']
    
    df = None
    
    if not os.path.exists('UniversalBank.csv'):
        raise FileNotFoundError("UniversalBank.csv not found in the current directory")
    
    with open('UniversalBank.csv', 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.strip().split('\n')
    
    try:
        df_test = pd.read_csv('UniversalBank.csv', nrows=2)
        if 'Income' in df_test.columns or 'income' in df_test.columns.str.lower():
            df = pd.read_csv('UniversalBank.csv')
            df.columns = df.columns.str.strip()
            column_mapping = {
                'ZIP Code': 'ZIPCode',
                'Personal Loan': 'PersonalLoan',
                'Securities Account': 'SecuritiesAccount',
                'CD Account': 'CDAccount',
                'ZIP.Code': 'ZIPCode',
                'Personal.Loan': 'PersonalLoan',
                'Securities.Account': 'SecuritiesAccount',
                'CD.Account': 'CDAccount'
            }
            df.rename(columns=column_mapping, inplace=True)
    except:
        pass
    
    if df is None or 'Income' not in df.columns:
        try:
            data_rows = []
            for line in lines:
                parts = line.split(',')
                numeric_values = []
                for part in parts:
                    part = part.strip().strip('"').strip("'")
                    if part:
                        try:
                            val = float(part)
                            numeric_values.append(val)
                        except ValueError:
                            continue
                if len(numeric_values) == 14:
                    data_rows.append(numeric_values)
                elif len(numeric_values) > 14:
                    data_rows.append(numeric_values[-14:])
            if len(data_rows) > 0:
                df = pd.DataFrame(data_rows, columns=column_names)
        except:
            pass
    
    if df is None or 'Income' not in df.columns:
        try:
            for skip in range(5):
                df_temp = pd.read_csv('UniversalBank.csv', skiprows=skip, header=None)
                df_temp = df_temp.dropna(axis=1, how='all')
                if len(df_temp.columns) >= 14:
                    try:
                        first_val = float(df_temp.iloc[0, 0])
                        df = df_temp.iloc[:, :14].copy()
                        df.columns = column_names
                        break
                    except:
                        continue
        except:
            pass
    
    if df is None or 'Income' not in df.columns:
        try:
            import re
            data_rows = []
            for line in lines:
                numbers = re.findall(r'-?\d+\.?\d*', line)
                if len(numbers) >= 14:
                    data_rows.append([float(n) for n in numbers[:14]])
            if len(data_rows) > 0:
                df = pd.DataFrame(data_rows, columns=column_names)
        except:
            pass
    
    if df is None:
        raise ValueError("Could not parse the CSV file. Please check the file format.")
    
    if 'Income' not in df.columns:
        raise ValueError(f"'Income' column not found. Available columns: {list(df.columns)}")
    
    for col in column_names:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    critical_cols = ['Income', 'Age', 'PersonalLoan']
    existing_critical = [col for col in critical_cols if col in df.columns]
    df = df.dropna(subset=existing_critical)
    df = df.reset_index(drop=True)
    
    df['IncomeGroup'] = pd.cut(df['Income'], 
                               bins=[0, 50, 100, 150, 200, 250],
                               labels=['<50K', '50-100K', '100-150K', '150-200K', '>200K'])
    
    df['AgeGroup'] = pd.cut(df['Age'], 
                            bins=[0, 30, 40, 50, 60, 70],
                            labels=['<30', '30-40', '40-50', '50-60', '>60'])
    
    df['EducationLevel'] = df['Education'].map({
        1: 'Undergraduate',
        2: 'Graduate', 
        3: 'Advanced/Professional'
    })
    
    df['HasMortgage'] = df['Mortgage'].apply(lambda x: 'Yes' if x > 0 else 'No')
    df['ZIPPrefix'] = df['ZIPCode'].astype(str).str[:3]
    df['LoanStatus'] = df['PersonalLoan'].map({0: 'Not Accepted', 1: 'Accepted'})
    
    return df

# -----------------------------------------------------------------------------
# KPI METRICS
# -----------------------------------------------------------------------------

def create_kpi_metrics(df):
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(label="📊 Total Customers", value=f"{len(df):,}")
    
    with col2:
        loan_rate = (df['PersonalLoan'].sum() / len(df)) * 100
        st.metric(label="💳 Loan Acceptance Rate", value=f"{loan_rate:.1f}%",
                  delta=f"{df['PersonalLoan'].sum()} customers")
    
    with col3:
        avg_income = df['Income'].mean()
        st.metric(label="💰 Average Income", value=f"${avg_income:,.0f}K")
    
    with col4:
        avg_age = df['Age'].mean()
        st.metric(label="👥 Average Age", value=f"{avg_age:.1f} years")
    
    with col5:
        cc_avg = df['CCAvg'].mean()
        st.metric(label="💳 Avg CC Spending", value=f"${cc_avg:,.2f}K/month")

# -----------------------------------------------------------------------------
# GRAPH 1: Histogram - Income/Age Distribution (Separate for Loan Acceptors/Non-Acceptors)
# -----------------------------------------------------------------------------

def create_income_age_histogram(df):
    """Graph 1: Histogram for Income and Age distribution separated by loan status"""
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=(
                            'Income Distribution - Loan Accepted',
                            'Income Distribution - Loan Not Accepted',
                            'Age Distribution - Loan Accepted',
                            'Age Distribution - Loan Not Accepted'
                        ))
    
    # Income - Loan Accepted
    accepted = df[df['PersonalLoan'] == 1]
    not_accepted = df[df['PersonalLoan'] == 0]
    
    fig.add_trace(
        go.Histogram(x=accepted['Income'], nbinsx=30, name='Loan Accepted',
                     marker_color=current_theme['success_color'], opacity=0.7),
        row=1, col=1
    )
    
    # Income - Loan Not Accepted
    fig.add_trace(
        go.Histogram(x=not_accepted['Income'], nbinsx=30, name='Loan Not Accepted',
                     marker_color=current_theme['error_color'], opacity=0.7),
        row=1, col=2
    )
    
    # Age - Loan Accepted
    fig.add_trace(
        go.Histogram(x=accepted['Age'], nbinsx=25, name='Loan Accepted',
                     marker_color=current_theme['success_color'], opacity=0.7,
                     showlegend=False),
        row=2, col=1
    )
    
    # Age - Loan Not Accepted
    fig.add_trace(
        go.Histogram(x=not_accepted['Age'], nbinsx=25, name='Loan Not Accepted',
                     marker_color=current_theme['error_color'], opacity=0.7,
                     showlegend=False),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Income ($K)", row=1, col=1)
    fig.update_xaxes(title_text="Income ($K)", row=1, col=2)
    fig.update_xaxes(title_text="Age (Years)", row=2, col=1)
    fig.update_xaxes(title_text="Age (Years)", row=2, col=2)
    
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    
    fig.update_layout(
        height=600, 
        title_text="📊 Graph 1: Income & Age Distribution by Loan Status",
        showlegend=True
    )
    
    return apply_plotly_theme(fig)

# -----------------------------------------------------------------------------
# GRAPH 2: Scatter Plot - CCAvg vs Income (by Personal Loan Status)
# -----------------------------------------------------------------------------

def create_ccavg_income_scatter(df):
    """Graph 2: Scatter plot of CCAvg vs Income colored by Personal Loan status"""
    fig = go.Figure()
    
    # Not Accepted
    not_accepted = df[df['PersonalLoan'] == 0]
    fig.add_trace(go.Scatter(
        x=not_accepted['Income'],
        y=not_accepted['CCAvg'],
        mode='markers',
        name='Loan Not Accepted',
        marker=dict(
            color=current_theme['accent_color'],
            size=6,
            opacity=0.5
        )
    ))
    
    # Accepted
    accepted = df[df['PersonalLoan'] == 1]
    fig.add_trace(go.Scatter(
        x=accepted['Income'],
        y=accepted['CCAvg'],
        mode='markers',
        name='Loan Accepted',
        marker=dict(
            color=current_theme['error_color'],
            size=8,
            opacity=0.7
        )
    ))
    
    fig.update_layout(
        title='📊 Graph 2: CCAvg vs Income by Personal Loan Status',
        xaxis_title='Income ($K)',
        yaxis_title='Credit Card Average Spending ($K/month)',
        height=500,
        legend=dict(x=0.02, y=0.98)
    )
    
    return apply_plotly_theme(fig)

# -----------------------------------------------------------------------------
# GRAPH 3: Zip Code vs Income vs Personal Loan Status
# -----------------------------------------------------------------------------

def create_zipcode_income_loan(df):
    """Graph 3: Visualization of Zip Code vs Income vs Personal Loan status"""
    # Group by ZIP prefix and calculate metrics
    zip_analysis = df.groupby('ZIPPrefix').agg({
        'Income': 'mean',
        'PersonalLoan': ['sum', 'count', 'mean']
    }).reset_index()
    zip_analysis.columns = ['ZIPPrefix', 'AvgIncome', 'LoanAccepted', 'TotalCustomers', 'AcceptanceRate']
    zip_analysis['AcceptanceRate'] = zip_analysis['AcceptanceRate'] * 100
    
    # Filter to top 20 ZIP prefixes by customer count
    zip_analysis = zip_analysis.nlargest(20, 'TotalCustomers')
    
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Avg Income by ZIP Code Region', 
                                       'Loan Acceptance Rate by ZIP Code Region'),
                        specs=[[{"type": "bar"}, {"type": "bar"}]])
    
    # Average Income by ZIP
    fig.add_trace(
        go.Bar(
            x=zip_analysis['ZIPPrefix'],
            y=zip_analysis['AvgIncome'],
            name='Avg Income',
            marker_color=current_theme['accent_color'],
            text=[f"${x:.0f}K" for x in zip_analysis['AvgIncome']],
            textposition='outside',
            textfont=dict(color=current_theme['axis_text_color'], size=8)
        ),
        row=1, col=1
    )
    
    # Acceptance Rate by ZIP
    colors = [current_theme['success_color'] if x > zip_analysis['AcceptanceRate'].mean() 
              else current_theme['warning_color'] for x in zip_analysis['AcceptanceRate']]
    fig.add_trace(
        go.Bar(
            x=zip_analysis['ZIPPrefix'],
            y=zip_analysis['AcceptanceRate'],
            name='Acceptance Rate',
            marker_color=colors,
            text=[f"{x:.1f}%" for x in zip_analysis['AcceptanceRate']],
            textposition='outside',
            textfont=dict(color=current_theme['axis_text_color'], size=8)
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="ZIP Code Prefix", row=1, col=1, tickangle=45)
    fig.update_xaxes(title_text="ZIP Code Prefix", row=1, col=2, tickangle=45)
    fig.update_yaxes(title_text="Average Income ($K)", row=1, col=1)
    fig.update_yaxes(title_text="Acceptance Rate (%)", row=1, col=2)
    
    fig.update_layout(
        height=500,
        title_text="📊 Graph 3: ZIP Code vs Income vs Personal Loan Status",
        showlegend=False
    )
    
    return apply_plotly_theme(fig)

# -----------------------------------------------------------------------------
# GRAPH 4: Correlation Heatmap
# -----------------------------------------------------------------------------

def create_correlation_heatmap(df):
    """Graph 4: Correlation heatmap for all columns"""
    numeric_cols = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 
                    'Education', 'Mortgage', 'PersonalLoan', 
                    'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard']
    
    existing_cols = [col for col in numeric_cols if col in df.columns]
    corr_matrix = df[existing_cols].corr()
    
    heatmap_text_color = '#000000' if st.session_state.theme == 'light' else '#FFFFFF'
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 9, "color": heatmap_text_color},
        hoverongaps=False,
        colorbar=dict(
            tickfont=dict(color=current_theme['axis_text_color']),
            title=dict(text='Correlation', font=dict(color=current_theme['axis_title_color']))
        )
    ))
    
    fig.update_layout(
        title='📊 Graph 4: Feature Correlation Matrix',
        height=600,
        xaxis_tickangle=-45
    )
    
    return apply_plotly_theme(fig)

# -----------------------------------------------------------------------------
# GRAPH 5: Family Size vs Income vs Mortgage/CCAvg (by Loan Status)
# -----------------------------------------------------------------------------

def create_family_income_mortgage_ccavg(df):
    """Graph 5: Family Size vs Income vs Mortgage/CCAvg with respect to Personal Loan status"""
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=(
                            'Family Size vs Income (Loan Accepted)',
                            'Family Size vs Income (Loan Not Accepted)',
                            'Family Size vs CCAvg by Loan Status',
                            'Family Size vs Mortgage by Loan Status'
                        ))
    
    accepted = df[df['PersonalLoan'] == 1]
    not_accepted = df[df['PersonalLoan'] == 0]
    
    # Family vs Income - Accepted
    family_income_acc = accepted.groupby('Family')['Income'].mean().reset_index()
    fig.add_trace(
        go.Bar(x=family_income_acc['Family'], y=family_income_acc['Income'],
               name='Loan Accepted', marker_color=current_theme['success_color'],
               text=[f"${x:.0f}K" for x in family_income_acc['Income']],
               textposition='outside'),
        row=1, col=1
    )
    
    # Family vs Income - Not Accepted
    family_income_not = not_accepted.groupby('Family')['Income'].mean().reset_index()
    fig.add_trace(
        go.Bar(x=family_income_not['Family'], y=family_income_not['Income'],
               name='Loan Not Accepted', marker_color=current_theme['error_color'],
               text=[f"${x:.0f}K" for x in family_income_not['Income']],
               textposition='outside'),
        row=1, col=2
    )
    
    # Family vs CCAvg by Loan Status
    for status, color, name in [(1, current_theme['success_color'], 'Accepted'),
                                  (0, current_theme['accent_color'], 'Not Accepted')]:
        subset = df[df['PersonalLoan'] == status]
        family_ccavg = subset.groupby('Family')['CCAvg'].mean().reset_index()
        fig.add_trace(
            go.Scatter(x=family_ccavg['Family'], y=family_ccavg['CCAvg'],
                      mode='lines+markers', name=f'CCAvg - {name}',
                      marker=dict(color=color, size=10),
                      line=dict(color=color, width=2)),
            row=2, col=1
        )
    
    # Family vs Mortgage by Loan Status
    for status, color, name in [(1, current_theme['success_color'], 'Accepted'),
                                  (0, current_theme['accent_color'], 'Not Accepted')]:
        subset = df[df['PersonalLoan'] == status]
        family_mortgage = subset.groupby('Family')['Mortgage'].mean().reset_index()
        fig.add_trace(
            go.Scatter(x=family_mortgage['Family'], y=family_mortgage['Mortgage'],
                      mode='lines+markers', name=f'Mortgage - {name}',
                      marker=dict(color=color, size=10),
                      line=dict(color=color, width=2)),
            row=2, col=2
        )
    
    fig.update_xaxes(title_text="Family Size", row=1, col=1)
    fig.update_xaxes(title_text="Family Size", row=1, col=2)
    fig.update_xaxes(title_text="Family Size", row=2, col=1)
    fig.update_xaxes(title_text="Family Size", row=2, col=2)
    
    fig.update_yaxes(title_text="Avg Income ($K)", row=1, col=1)
    fig.update_yaxes(title_text="Avg Income ($K)", row=1, col=2)
    fig.update_yaxes(title_text="Avg CCAvg ($K)", row=2, col=1)
    fig.update_yaxes(title_text="Avg Mortgage ($K)", row=2, col=2)
    
    fig.update_layout(
        height=700,
        title_text="📊 Graph 5: Family Size vs Income vs Mortgage/CCAvg by Loan Status",
        showlegend=True
    )
    
    return apply_plotly_theme(fig)

# -----------------------------------------------------------------------------
# GRAPH 6: Securities vs CD vs Credit Cards (by Loan Status) - FIXED
# -----------------------------------------------------------------------------

def create_securities_cd_creditcard(df):
    """Graph 6: Securities vs Cash Deposit vs Credit Cards with respect to Personal Loan status"""
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=(
                            'Service Combinations by Loan Status',
                            'Loan Acceptance Rate by Service Type',
                            'Securities + CD Account Holders',
                            'Credit Card Holders by Other Services'
                        ),
                        specs=[[{"type": "bar"}, {"type": "bar"}],
                               [{"type": "pie"}, {"type": "pie"}]])
    
    # Create a working copy
    df_work = df.copy()
    
    # Ensure we have numeric values for the service columns
    df_work['SecuritiesAccount'] = pd.to_numeric(df_work['SecuritiesAccount'], errors='coerce').fillna(0).astype(int)
    df_work['CDAccount'] = pd.to_numeric(df_work['CDAccount'], errors='coerce').fillna(0).astype(int)
    df_work['CreditCard'] = pd.to_numeric(df_work['CreditCard'], errors='coerce').fillna(0).astype(int)
    df_work['PersonalLoan'] = pd.to_numeric(df_work['PersonalLoan'], errors='coerce').fillna(0).astype(int)
    
    # Create service combination labels
    combo_labels = {
        '000': 'No Services',
        '001': 'CC Only',
        '010': 'CD Only',
        '011': 'CD + CC',
        '100': 'Securities Only',
        '101': 'Securities + CC',
        '110': 'Securities + CD',
        '111': 'All Services'
    }
    
    # Create ServiceCombo column
    df_work['ServiceCombo'] = (df_work['SecuritiesAccount'].astype(str) + 
                               df_work['CDAccount'].astype(str) + 
                               df_work['CreditCard'].astype(str))
    
    df_work['ServiceLabel'] = df_work['ServiceCombo'].map(combo_labels)
    
    # Calculate counts for each combination by loan status
    accepted_counts = []
    not_accepted_counts = []
    labels_order = list(combo_labels.values())
    
    for combo, label in combo_labels.items():
        subset = df_work[df_work['ServiceCombo'] == combo]
        accepted = len(subset[subset['PersonalLoan'] == 1])
        not_accepted = len(subset[subset['PersonalLoan'] == 0])
        accepted_counts.append(accepted)
        not_accepted_counts.append(not_accepted)
    
    # Service Combinations by Loan Status - Bar Chart
    fig.add_trace(
        go.Bar(x=labels_order, 
               y=accepted_counts,
               name='Loan Accepted', 
               marker_color=current_theme['success_color']),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=labels_order,
               y=not_accepted_counts,
               name='Loan Not Accepted', 
               marker_color=current_theme['error_color']),
        row=1, col=1
    )
    
    # Loan Acceptance Rate by Service Type
    services = ['SecuritiesAccount', 'CDAccount', 'CreditCard']
    service_names = ['Securities', 'CD Account', 'Credit Card']
    rates = []
    for service in services:
        service_holders = df_work[df_work[service] == 1]
        if len(service_holders) > 0:
            rate = service_holders['PersonalLoan'].mean() * 100
        else:
            rate = 0
        rates.append(rate)
    
    fig.add_trace(
        go.Bar(x=service_names, y=rates,
               marker_color=[current_theme['accent_color'], 
                            current_theme['success_color'], 
                            current_theme['warning_color']],
               text=[f"{r:.1f}%" for r in rates],
               textposition='outside',
               textfont=dict(color=current_theme['axis_text_color']),
               showlegend=False),
        row=1, col=2
    )
    
    # Pie chart - Securities + CD Account Holders by Loan Status
    sec_cd = df_work[(df_work['SecuritiesAccount'] == 1) | (df_work['CDAccount'] == 1)]
    if len(sec_cd) > 0:
        sec_cd_accepted = len(sec_cd[sec_cd['PersonalLoan'] == 1])
        sec_cd_not_accepted = len(sec_cd[sec_cd['PersonalLoan'] == 0])
    else:
        sec_cd_accepted = 0
        sec_cd_not_accepted = 0
    
    fig.add_trace(
        go.Pie(labels=['Not Accepted', 'Accepted'],
               values=[sec_cd_not_accepted, sec_cd_accepted],
               marker_colors=[current_theme['error_color'], current_theme['success_color']],
               hole=0.4,
               textinfo='label+percent',
               textfont=dict(color=current_theme['text_color'])),
        row=2, col=1
    )
    
    # Pie chart - Credit Card Holders by Other Services
    cc_holders = df_work[df_work['CreditCard'] == 1]
    if len(cc_holders) > 0:
        cc_with_other = len(cc_holders[(cc_holders['SecuritiesAccount'] == 1) | (cc_holders['CDAccount'] == 1)])
        cc_without_other = len(cc_holders[(cc_holders['SecuritiesAccount'] == 0) & (cc_holders['CDAccount'] == 0)])
    else:
        cc_with_other = 0
        cc_without_other = 0
    
    fig.add_trace(
        go.Pie(labels=['CC + Other Services', 'CC Only'],
               values=[cc_with_other, cc_without_other],
               marker_colors=[current_theme['accent_color'], current_theme['warning_color']],
               hole=0.4,
               textinfo='label+percent',
               textfont=dict(color=current_theme['text_color'])),
        row=2, col=2
    )
    
    fig.update_xaxes(tickangle=45, row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Acceptance Rate (%)", row=1, col=2)
    
    fig.update_layout(
        height=700,
        title_text="📊 Graph 6: Securities vs CD vs Credit Cards by Loan Status",
        barmode='group'
    )
    
    return apply_plotly_theme(fig)

# -----------------------------------------------------------------------------
# GRAPH 7: Box and Whiskers Plot for Credit Card, CCAvg, and Income
# -----------------------------------------------------------------------------

def create_box_whiskers(df):
    """Graph 7: Box and Whiskers Plot for Credit Card count, CCAvg, and Income"""
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=(
                            'Income Distribution by Loan Status',
                            'CCAvg Distribution by Loan Status',
                            'Income by Credit Card Ownership'
                        ))
    
    # Box plot for Income by Loan Status
    for status, color in [('Accepted', current_theme['success_color']), 
                          ('Not Accepted', current_theme['error_color'])]:
        fig.add_trace(
            go.Box(y=df[df['LoanStatus'] == status]['Income'],
                   name=f'Income - {status}',
                   marker_color=color,
                   boxmean=True),
            row=1, col=1
        )
    
    # Box plot for CCAvg by Loan Status
    for status, color in [('Accepted', current_theme['success_color']), 
                          ('Not Accepted', current_theme['error_color'])]:
        fig.add_trace(
            go.Box(y=df[df['LoanStatus'] == status]['CCAvg'],
                   name=f'CCAvg - {status}',
                   marker_color=color,
                   boxmean=True),
            row=1, col=2
        )
    
    # Box plot for Income by Credit Card Ownership
    for cc_status, color, name in [(1, current_theme['accent_color'], 'Has CC'),
                                    (0, current_theme['warning_color'], 'No CC')]:
        fig.add_trace(
            go.Box(y=df[df['CreditCard'] == cc_status]['Income'],
                   name=f'Income - {name}',
                   marker_color=color,
                   boxmean=True),
            row=1, col=3
        )
    
    fig.update_yaxes(title_text="Income ($K)", row=1, col=1)
    fig.update_yaxes(title_text="CCAvg ($K/month)", row=1, col=2)
    fig.update_yaxes(title_text="Income ($K)", row=1, col=3)
    
    fig.update_layout(
        height=500,
        title_text="📊 Graph 7: Box & Whiskers - Income, CCAvg, and Credit Card",
        showlegend=True
    )
    
    return apply_plotly_theme(fig)

# -----------------------------------------------------------------------------
# GRAPH 8: Education vs Income vs Personal Loan Status
# -----------------------------------------------------------------------------

def create_education_income_loan(df):
    """Graph 8: Education vs Income vs Personal Loan status"""
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=(
                            'Income Distribution by Education Level',
                            'Loan Acceptance Rate by Education',
                            'Education Level Distribution',
                            'Avg Income by Education & Loan Status'
                        ),
                        specs=[[{"type": "box"}, {"type": "bar"}],
                               [{"type": "pie"}, {"type": "bar"}]])
    
    edu_order = ['Undergraduate', 'Graduate', 'Advanced/Professional']
    colors = [current_theme['accent_color'], current_theme['success_color'], current_theme['warning_color']]
    
    # Box plot - Income by Education Level
    for i, edu in enumerate(edu_order):
        subset = df[df['EducationLevel'] == edu]
        fig.add_trace(
            go.Box(y=subset['Income'], name=edu, marker_color=colors[i], boxmean=True),
            row=1, col=1
        )
    
    # Bar chart - Loan Acceptance Rate by Education
    edu_loan = df.groupby('EducationLevel')['PersonalLoan'].mean().reindex(edu_order) * 100
    fig.add_trace(
        go.Bar(x=edu_order, y=edu_loan.values,
               marker_color=colors,
               text=[f"{r:.1f}%" for r in edu_loan.values],
               textposition='outside',
               textfont=dict(color=current_theme['axis_text_color']),
               showlegend=False),
        row=1, col=2
    )
    
    # Pie chart - Education Level Distribution
    edu_counts = df['EducationLevel'].value_counts().reindex(edu_order)
    fig.add_trace(
        go.Pie(labels=edu_order, values=edu_counts.values,
               marker_colors=colors, hole=0.4,
               textfont=dict(color=current_theme['text_color'])),
        row=2, col=1
    )
    
    # Grouped bar - Avg Income by Education & Loan Status
    edu_loan_income = df.groupby(['EducationLevel', 'LoanStatus'])['Income'].mean().unstack()
    
    for status, color in [('Not Accepted', current_theme['error_color']),
                          ('Accepted', current_theme['success_color'])]:
        if status in edu_loan_income.columns:
            values = [edu_loan_income.loc[edu, status] if edu in edu_loan_income.index else 0 
                     for edu in edu_order]
            fig.add_trace(
                go.Bar(x=edu_order, y=values, name=status, marker_color=color),
                row=2, col=2
            )
    
    fig.update_yaxes(title_text="Income ($K)", row=1, col=1)
    fig.update_yaxes(title_text="Acceptance Rate (%)", row=1, col=2)
    fig.update_yaxes(title_text="Avg Income ($K)", row=2, col=2)
    
    fig.update_layout(
        height=700,
        title_text="📊 Graph 8: Education vs Income vs Personal Loan Status",
        barmode='group'
    )
    
    return apply_plotly_theme(fig)

# -----------------------------------------------------------------------------
# GRAPH 9: Mortgage vs Income vs Family Size vs Personal Loan Status
# -----------------------------------------------------------------------------

def create_mortgage_income_family_loan(df):
    """Graph 9: Mortgage vs Income vs Family Size vs Personal Loan status"""
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=(
                            'Mortgage vs Income (colored by Loan Status)',
                            'Avg Mortgage by Family Size & Loan Status',
                            'Income vs Family Size (colored by Loan Status)',
                            'Mortgage Distribution by Family Size'
                        ))
    
    # Scatter plot - Mortgage vs Income by Loan Status
    for status, color, name in [(0, current_theme['accent_color'], 'Not Accepted'),
                                 (1, current_theme['error_color'], 'Accepted')]:
        subset = df[df['PersonalLoan'] == status]
        fig.add_trace(
            go.Scatter(x=subset['Income'], y=subset['Mortgage'],
                      mode='markers', name=name,
                      marker=dict(color=color, size=5, opacity=0.5)),
            row=1, col=1
        )
    
    # Bar chart - Avg Mortgage by Family Size & Loan Status
    mortgage_family = df.groupby(['Family', 'LoanStatus'])['Mortgage'].mean().unstack()
    
    for status, color in [('Not Accepted', current_theme['accent_color']),
                          ('Accepted', current_theme['success_color'])]:
        if status in mortgage_family.columns:
            fig.add_trace(
                go.Bar(x=mortgage_family.index, y=mortgage_family[status],
                      name=f'Mortgage - {status}', marker_color=color),
                row=1, col=2
            )
    
    # Scatter plot - Income vs Family Size by Loan Status
    for status, color, name in [(0, current_theme['accent_color'], 'Not Accepted'),
                                 (1, current_theme['success_color'], 'Accepted')]:
        subset = df[df['PersonalLoan'] == status]
        # Add jitter to Family for better visualization
        jittered_family = subset['Family'] + np.random.uniform(-0.2, 0.2, len(subset))
        fig.add_trace(
            go.Scatter(x=jittered_family, y=subset['Income'],
                      mode='markers', name=f'Income - {name}',
                      marker=dict(color=color, size=5, opacity=0.4),
                      showlegend=False),
            row=2, col=1
        )
    
    # Box plot - Mortgage Distribution by Family Size
    for family in sorted(df['Family'].unique()):
        subset = df[df['Family'] == family]
        fig.add_trace(
            go.Box(y=subset[subset['Mortgage'] > 0]['Mortgage'],
                   name=f'Family {int(family)}',
                   marker_color=current_theme['warning_color'],
                   showlegend=False),
            row=2, col=2
        )
    
    fig.update_xaxes(title_text="Income ($K)", row=1, col=1)
    fig.update_xaxes(title_text="Family Size", row=1, col=2)
    fig.update_xaxes(title_text="Family Size", row=2, col=1)
    fig.update_xaxes(title_text="Family Size", row=2, col=2)
    
    fig.update_yaxes(title_text="Mortgage ($K)", row=1, col=1)
    fig.update_yaxes(title_text="Avg Mortgage ($K)", row=1, col=2)
    fig.update_yaxes(title_text="Income ($K)", row=2, col=1)
    fig.update_yaxes(title_text="Mortgage ($K)", row=2, col=2)
    
    fig.update_layout(
        height=700,
        title_text="📊 Graph 9: Mortgage vs Income vs Family Size vs Loan Status",
        barmode='group'
    )
    
    return apply_plotly_theme(fig)

# -----------------------------------------------------------------------------
# MACHINE LEARNING FUNCTIONS
# -----------------------------------------------------------------------------

@st.cache_resource
def train_models(df):
    feature_cols = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 
                    'Education', 'Mortgage', 'SecuritiesAccount', 
                    'CDAccount', 'Online', 'CreditCard']
    
    existing_features = [col for col in feature_cols if col in df.columns]
    
    X = df[existing_features].copy()
    y = df['PersonalLoan'].copy()
    X = X.fillna(X.median())
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        if name in ['K-Nearest Neighbors', 'Logistic Regression']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        trained_models[name] = model
    
    feature_importance = pd.DataFrame({
        'Feature': existing_features,
        'Importance': trained_models['Random Forest'].feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return results, trained_models, feature_importance, scaler, X_test, y_test, existing_features

def create_model_comparison(results):
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [r['accuracy'] for r in results.values()],
        'Precision': [r['precision'] for r in results.values()],
        'Recall': [r['recall'] for r in results.values()],
        'F1 Score': [r['f1'] for r in results.values()],
        'ROC AUC': [r['roc_auc'] for r in results.values()]
    })
    
    fig = go.Figure()
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    colors = [current_theme['accent_color'], current_theme['success_color'], 
              current_theme['warning_color'], current_theme['error_color'], '#8B5CF6']
    
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric,
            x=metrics_df['Model'],
            y=metrics_df[metric],
            marker_color=colors[i],
            text=[f'{v:.3f}' for v in metrics_df[metric]],
            textposition='outside',
            textfont=dict(color=current_theme['axis_text_color'])
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        barmode='group',
        height=500,
        yaxis_range=[0, 1.1],
        legend=dict(orientation='h', y=1.1, x=0.3, font=dict(color=current_theme['legend_text_color']))
    )
    
    return apply_plotly_theme(fig), metrics_df

def create_roc_curves(results):
    fig = go.Figure()
    colors = {'Logistic Regression': current_theme['accent_color'], 
              'Random Forest': current_theme['success_color'],
              'Gradient Boosting': current_theme['warning_color'], 
              'K-Nearest Neighbors': current_theme['error_color']}
    
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(result['y_test'], result['y_prob'])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"{name} (AUC={result['roc_auc']:.3f})",
            mode='lines',
            line=dict(color=colors.get(name, '#888888'), width=2)
        ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        legend=dict(x=0.6, y=0.1, font=dict(color=current_theme['legend_text_color']))
    )
    
    return apply_plotly_theme(fig)

def create_confusion_matrices(results):
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=list(results.keys()),
                        vertical_spacing=0.15,
                        horizontal_spacing=0.1)
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    cm_text_color = '#000000' if st.session_state.theme == 'light' else '#FFFFFF'
    
    for (name, result), (row, col) in zip(results.items(), positions):
        cm = result['confusion_matrix']
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=['Predicted 0', 'Predicted 1'],
                y=['Actual 0', 'Actual 1'],
                colorscale='Blues',
                showscale=False,
                text=[[cm[0,0], cm[0,1]], [cm[1,0], cm[1,1]]],
                texttemplate='%{text}',
                textfont={"size": 14, "color": cm_text_color}
            ),
            row=row, col=col
        )
    
    fig.update_layout(height=600, title_text="Confusion Matrices")
    return apply_plotly_theme(fig)

def create_feature_importance(feature_importance):
    fig = go.Figure(go.Bar(
        x=feature_importance['Importance'],
        y=feature_importance['Feature'],
        orientation='h',
        marker_color=current_theme['accent_color'],
        text=[f'{v:.3f}' for v in feature_importance['Importance']],
        textposition='outside',
        textfont=dict(color=current_theme['axis_text_color'])
    ))
    
    fig.update_layout(
        title='Feature Importance (Random Forest)',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return apply_plotly_theme(fig)

# -----------------------------------------------------------------------------
# CUSTOMER SEGMENTATION
# -----------------------------------------------------------------------------

def perform_clustering(df):
    features_for_clustering = ['Age', 'Income', 'CCAvg', 'Mortgage']
    existing_features = [f for f in features_for_clustering if f in df.columns]
    
    X_cluster = df[existing_features].copy()
    X_cluster = X_cluster.fillna(X_cluster.median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]
    
    return df, kmeans

def create_cluster_visualization(df):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Customer Segments (PCA)', 'Cluster Characteristics'))
    
    colors = [current_theme['accent_color'], current_theme['success_color'], 
              current_theme['warning_color'], current_theme['error_color']]
    
    for i in df['Cluster'].unique():
        subset = df[df['Cluster'] == i]
        fig.add_trace(
            go.Scatter(
                x=subset['PCA1'],
                y=subset['PCA2'],
                mode='markers',
                name=f'Cluster {i}',
                marker=dict(color=colors[i % len(colors)], size=5, opacity=0.6)
            ),
            row=1, col=1
        )
    
    cluster_stats = df.groupby('Cluster').agg({
        'Income': 'mean',
        'Age': 'mean',
        'CCAvg': 'mean',
        'PersonalLoan': 'mean'
    }).reset_index()
    
    fig.add_trace(
        go.Bar(x=cluster_stats['Cluster'], y=cluster_stats['Income'],
               name='Avg Income', marker_color=current_theme['accent_color']),
        row=1, col=2
    )
    
    fig.update_layout(height=400, title_text="Customer Segmentation Analysis")
    
    return apply_plotly_theme(fig), df.groupby('Cluster').agg({
        'Income': 'mean',
        'Age': 'mean',
        'CCAvg': 'mean',
        'PersonalLoan': ['mean', 'sum', 'count']
    })

# -----------------------------------------------------------------------------
# MAIN APPLICATION
# -----------------------------------------------------------------------------

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        theme_icon = "🌙" if st.session_state.theme == 'light' else "☀️"
        theme_text = "Switch to Dark Mode" if st.session_state.theme == 'light' else "Switch to Light Mode"
        
        if st.button(f"{theme_icon} {theme_text}", key="theme_toggle", use_container_width=True):
            toggle_theme()
            st.rerun()
        
        current_theme_display = "🌞 Light Mode" if st.session_state.theme == 'light' else "🌙 Dark Mode"
        st.markdown(f"**Current Theme:** {current_theme_display}")
        
        st.markdown("---")
        
        if st.checkbox("🔧 Show Debug Info", value=False):
            st.write("**Current Directory:**", os.getcwd())
            if os.path.exists('UniversalBank.csv'):
                st.success("✅ UniversalBank.csv found!")
            else:
                st.error("❌ UniversalBank.csv NOT found!")
        
        st.markdown("---")
    
    # Header
    st.markdown('<h1 class="main-header">🏦 Universal Bank - Loan Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    try:
        df = load_data()
        st.sidebar.success(f"✅ Data loaded! ({len(df)} records)")
    except Exception as e:
        st.error(f"⚠️ **Error loading data:** {str(e)}")
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("🎯 Filters")
    
    income_range = st.sidebar.slider(
        "Income Range ($K)",
        int(df['Income'].min()),
        int(df['Income'].max()),
        (int(df['Income'].min()), int(df['Income'].max()))
    )
    
    age_range = st.sidebar.slider(
        "Age Range",
        int(df['Age'].min()),
        int(df['Age'].max()),
        (int(df['Age'].min()), int(df['Age'].max()))
    )
    
    education_options = df['EducationLevel'].dropna().unique().tolist()
    selected_education = st.sidebar.multiselect(
        "Education Level",
        options=education_options,
        default=education_options
    )
    
    family_options = sorted(df['Family'].unique().tolist())
    selected_family = st.sidebar.multiselect(
        "Family Size",
        options=family_options,
        default=family_options
    )
    
    # Apply filters
    filtered_df = df[
        (df['Income'] >= income_range[0]) &
        (df['Income'] <= income_range[1]) &
        (df['Age'] >= age_range[0]) &
        (df['Age'] <= age_range[1]) &
        (df['EducationLevel'].isin(selected_education)) &
        (df['Family'].isin(selected_family))
    ]
    
    st.sidebar.markdown("---")
    st.sidebar.metric("Filtered Customers", f"{len(filtered_df):,}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview & Key Graphs", 
        "📈 Detailed Analysis", 
        "🤖 ML Models", 
        "👥 Customer Segments",
        "🔮 Predictions"
    ])
    
    # TAB 1: Overview & Key Graphs (Graphs 1-4)
    with tab1:
        st.markdown("### 📊 Key Performance Indicators")
        create_kpi_metrics(filtered_df)
        
        st.markdown("---")
        
        # Graph 1: Income & Age Histograms
        st.markdown("### 📊 Graph 1: Income & Age Distribution by Loan Status")
        fig1 = create_income_age_histogram(filtered_df)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Graph 2: CCAvg vs Income Scatter
            st.markdown("### 📊 Graph 2: CCAvg vs Income")
            fig2 = create_ccavg_income_scatter(filtered_df)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Graph 3: ZIP Code Analysis
            st.markdown("### 📊 Graph 3: ZIP Code Analysis")
            fig3 = create_zipcode_income_loan(filtered_df)
            st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("---")
        
        # Graph 4: Correlation Heatmap
        st.markdown("### 📊 Graph 4: Correlation Matrix")
        fig4 = create_correlation_heatmap(filtered_df)
        st.plotly_chart(fig4, use_container_width=True)
    
    # TAB 2: Detailed Analysis (Graphs 5-9)
    with tab2:
        st.markdown("### 📈 Detailed Feature Analysis")
        
        # Graph 5: Family Size Analysis
        st.markdown("### 📊 Graph 5: Family Size vs Income vs Mortgage/CCAvg")
        fig5 = create_family_income_mortgage_ccavg(filtered_df)
        st.plotly_chart(fig5, use_container_width=True)
        
        st.markdown("---")
        
        # Graph 6: Securities vs CD vs Credit Cards
        st.markdown("### 📊 Graph 6: Securities vs CD vs Credit Cards")
        fig6 = create_securities_cd_creditcard(filtered_df)
        st.plotly_chart(fig6, use_container_width=True)
        
        st.markdown("---")
        
        # Graph 7: Box and Whiskers
        st.markdown("### 📊 Graph 7: Box & Whiskers Plot")
        fig7 = create_box_whiskers(filtered_df)
        st.plotly_chart(fig7, use_container_width=True)
        
        st.markdown("---")
        
        # Graph 8: Education Analysis
        st.markdown("### 📊 Graph 8: Education vs Income vs Loan Status")
        fig8 = create_education_income_loan(filtered_df)
        st.plotly_chart(fig8, use_container_width=True)
        
        st.markdown("---")
        
        # Graph 9: Mortgage Analysis
        st.markdown("### 📊 Graph 9: Mortgage vs Income vs Family Size")
        fig9 = create_mortgage_income_family_loan(filtered_df)
        st.plotly_chart(fig9, use_container_width=True)
        
        st.markdown("---")
        
        # Data Preview
        st.markdown("#### 📋 Data Preview")
        display_cols = ['ID', 'Age', 'Income', 'Family', 'CCAvg', 'Education', 
                        'Mortgage', 'PersonalLoan', 'EducationLevel', 'LoanStatus']
        existing_display_cols = [c for c in display_cols if c in filtered_df.columns]
        st.dataframe(filtered_df[existing_display_cols].head(100), use_container_width=True)
    
    # TAB 3: ML Models
    with tab3:
        st.markdown("### 🤖 Machine Learning Model Performance")
        
        with st.spinner("Training models..."):
            results, trained_models, feature_importance, scaler, X_test, y_test, feature_names = train_models(df)
        
        fig_comparison, metrics_df = create_model_comparison(results)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        st.markdown("#### 📊 Performance Metrics Table")
        st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']),
                     use_container_width=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ROC Curves")
            fig_roc = create_roc_curves(results)
            st.plotly_chart(fig_roc, use_container_width=True)
        
        with col2:
            st.markdown("#### Feature Importance")
            fig_fi = create_feature_importance(feature_importance)
            st.plotly_chart(fig_fi, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("#### Confusion Matrices")
        fig_cm = create_confusion_matrices(results)
        st.plotly_chart(fig_cm, use_container_width=True)
    
    # TAB 4: Customer Segments
    with tab4:
        st.markdown("### 👥 Customer Segmentation Analysis")
        
        with st.spinner("Performing clustering..."):
            clustered_df, kmeans = perform_clustering(filtered_df.copy())
        
        fig_cluster, cluster_stats = create_cluster_visualization(clustered_df)
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        st.markdown("#### Cluster Statistics")
        st.dataframe(cluster_stats, use_container_width=True)
        
        st.markdown("#### 📝 Cluster Insights")
        
        cluster_summary = clustered_df.groupby('Cluster').agg({
            'Income': 'mean',
            'Age': 'mean',
            'CCAvg': 'mean',
            'Mortgage': 'mean',
            'PersonalLoan': 'mean'
        }).round(2)
        
        for i in range(4):
            if i in cluster_summary.index:
                with st.expander(f"Cluster {i} Details"):
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Avg Income", f"${cluster_summary.loc[i, 'Income']:.0f}K")
                    col2.metric("Avg Age", f"{cluster_summary.loc[i, 'Age']:.1f}")
                    col3.metric("Avg CC Spending", f"${cluster_summary.loc[i, 'CCAvg']:.2f}K")
                    col4.metric("Avg Mortgage", f"${cluster_summary.loc[i, 'Mortgage']:.0f}K")
                    col5.metric("Loan Accept Rate", f"{cluster_summary.loc[i, 'PersonalLoan']*100:.1f}%")
    
    # TAB 5: Predictions
    with tab5:
        st.markdown("### 🔮 Predict Loan Acceptance")
        st.markdown("Enter customer details to predict loan acceptance probability.")
        
        with st.spinner("Loading models..."):
            results, trained_models, feature_importance, scaler, X_test, y_test, feature_names = train_models(df)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pred_age = st.number_input("Age", min_value=18, max_value=70, value=35)
            pred_experience = st.number_input("Experience (years)", min_value=-3, max_value=50, value=10)
            pred_income = st.number_input("Income ($K)", min_value=0, max_value=250, value=50)
            pred_family = st.selectbox("Family Size", [1, 2, 3, 4])
        
        with col2:
            pred_ccavg = st.number_input("CC Avg Spending ($K/month)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
            pred_education = st.selectbox("Education", [1, 2, 3], format_func=lambda x: {1: "Undergraduate", 2: "Graduate", 3: "Advanced"}[x])
            pred_mortgage = st.number_input("Mortgage ($K)", min_value=0, max_value=700, value=0)
        
        with col3:
            pred_securities = st.selectbox("Securities Account", [0, 1], format_func=lambda x: "Yes" if x else "No")
            pred_cd = st.selectbox("CD Account", [0, 1], format_func=lambda x: "Yes" if x else "No")
            pred_online = st.selectbox("Online Banking", [0, 1], format_func=lambda x: "Yes" if x else "No")
            pred_cc = st.selectbox("Credit Card", [0, 1], format_func=lambda x: "Yes" if x else "No")
        
        if st.button("🔮 Predict Loan Acceptance", type="primary"):
            input_data = pd.DataFrame({
                'Age': [pred_age],
                'Experience': [pred_experience],
                'Income': [pred_income],
                'Family': [pred_family],
                'CCAvg': [pred_ccavg],
                'Education': [pred_education],
                'Mortgage': [pred_mortgage],
                'SecuritiesAccount': [pred_securities],
                'CDAccount': [pred_cd],
                'Online': [pred_online],
                'CreditCard': [pred_cc]
            })
            
            input_data = input_data[[f for f in feature_names if f in input_data.columns]]
            
            st.markdown("---")
            st.markdown("#### Prediction Results")
            
            results_cols = st.columns(len(trained_models))
            
            for idx, (name, model) in enumerate(trained_models.items()):
                with results_cols[idx]:
                    if name in ['K-Nearest Neighbors', 'Logistic Regression']:
                        input_scaled = scaler.transform(input_data)
                        prob = model.predict_proba(input_scaled)[0][1]
                    else:
                        prob = model.predict_proba(input_data)[0][1]
                    
                    prediction = "✅ ACCEPT" if prob >= 0.5 else "❌ REJECT"
                    color = "green" if prob >= 0.5 else "red"
                    
                    st.markdown(f"**{name}**")
                    st.markdown(f"<h3 style='color: {color}'>{prediction}</h3>", unsafe_allow_html=True)
                    st.progress(float(prob))
                    st.write(f"Probability: {prob:.2%}")

if __name__ == "__main__":
    main()

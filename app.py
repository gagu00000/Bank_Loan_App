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

# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def toggle_theme():
    """Toggle between light and dark theme"""
    if st.session_state.theme == 'light':
        st.session_state.theme = 'dark'
    else:
        st.session_state.theme = 'light'

# Theme colors - FIXED: Added axis_text_color for proper readability
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
        'axis_text_color': '#000000',  # BLACK for light mode
        'axis_title_color': '#1E1E1E',  # Dark gray for axis titles
        'legend_text_color': '#1E1E1E',  # Dark text for legend
        'title_color': '#1E3A8A'  # Dark blue for chart titles
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
        'axis_text_color': '#FFFFFF',  # WHITE for dark mode
        'axis_title_color': '#FAFAFA',  # Light color for axis titles
        'legend_text_color': '#FAFAFA',  # Light text for legend
        'title_color': '#60A5FA'  # Light blue for chart titles
    }
}

# Get current theme
current_theme = THEMES[st.session_state.theme]

# Apply custom CSS based on theme
def apply_theme_css():
    theme = current_theme
    
    css = f"""
    <style>
    /* Main app background */
    .stApp {{
        background-color: {theme['bg_color']};
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {theme['secondary_bg']};
    }}
    
    [data-testid="stSidebar"] .stMarkdown {{
        color: {theme['text_color']};
    }}
    
    /* Headers */
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
    
    /* Metric cards */
    [data-testid="stMetricValue"] {{
        color: {theme['text_color']};
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {theme['text_color']};
    }}
    
    /* Text elements */
    .stMarkdown, .stText {{
        color: {theme['text_color']};
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: {theme['text_color']} !important;
    }}
    
    p, span, label {{
        color: {theme['text_color']};
    }}
    
    /* Cards and containers */
    .metric-card {{
        background-color: {theme['card_bg']};
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid {theme['accent_color']};
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    /* Tabs */
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
    
    /* Buttons */
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
    
    /* Expanders */
    .streamlit-expanderHeader {{
        background-color: {theme['secondary_bg']};
        color: {theme['text_color']};
        border-radius: 5px;
    }}
    
    .streamlit-expanderContent {{
        background-color: {theme['card_bg']};
        border: 1px solid {theme['border_color']};
    }}
    
    /* DataFrames */
    .stDataFrame {{
        background-color: {theme['card_bg']};
    }}
    
    /* Select boxes and inputs */
    .stSelectbox, .stMultiSelect, .stSlider, .stNumberInput {{
        color: {theme['text_color']};
    }}
    
    /* Theme toggle button */
    .theme-toggle {{
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, {theme['accent_color']}, {theme['header_color']});
        color: white;
        border-radius: 25px;
        font-weight: bold;
        margin: 1rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }}
    
    .theme-toggle:hover {{
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }}
    
    /* Info boxes */
    .stAlert {{
        background-color: {theme['secondary_bg']};
        color: {theme['text_color']};
    }}
    
    /* Progress bar */
    .stProgress > div > div {{
        background-color: {theme['accent_color']};
    }}
    
    /* Divider */
    hr {{
        border-color: {theme['border_color']};
    }}
    
    /* Custom scrollbar */
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
    
    /* Animation for cards */
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

# Apply theme CSS
apply_theme_css()

# -----------------------------------------------------------------------------
# PLOTLY THEME HELPER - FIXED FOR AXIS TEXT COLORS
# -----------------------------------------------------------------------------

def get_plotly_layout():
    """Get Plotly layout settings based on current theme"""
    theme = current_theme
    return {
        'template': theme['plotly_template'],
        'paper_bgcolor': theme['chart_bg'],
        'plot_bgcolor': theme['chart_bg'],
        'font': {'color': theme['text_color']},
        'title': {'font': {'color': theme['title_color']}},
        'xaxis': {
            'gridcolor': theme['grid_color'],
            'zerolinecolor': theme['grid_color'],
            'tickfont': {'color': theme['axis_text_color']},
            'titlefont': {'color': theme['axis_title_color']}
        },
        'yaxis': {
            'gridcolor': theme['grid_color'],
            'zerolinecolor': theme['grid_color'],
            'tickfont': {'color': theme['axis_text_color']},
            'titlefont': {'color': theme['axis_title_color']}
        },
        'legend': {
            'font': {'color': theme['legend_text_color']}
        }
    }

def apply_plotly_theme(fig):
    """Apply current theme to a Plotly figure - FIXED for axis text colors"""
    theme = current_theme
    
    # Update main layout
    fig.update_layout(
        template=theme['plotly_template'],
        paper_bgcolor=theme['chart_bg'],
        plot_bgcolor=theme['chart_bg'],
        font=dict(color=theme['text_color']),
        title=dict(font=dict(color=theme['title_color'])),
        legend=dict(font=dict(color=theme['legend_text_color']))
    )
    
    # Update all x-axes (handles subplots too)
    fig.update_xaxes(
        gridcolor=theme['grid_color'],
        zerolinecolor=theme['grid_color'],
        tickfont=dict(color=theme['axis_text_color']),
        titlefont=dict(color=theme['axis_title_color']),
        linecolor=theme['axis_text_color']
    )
    
    # Update all y-axes (handles subplots too)
    fig.update_yaxes(
        gridcolor=theme['grid_color'],
        zerolinecolor=theme['grid_color'],
        tickfont=dict(color=theme['axis_text_color']),
        titlefont=dict(color=theme['axis_title_color']),
        linecolor=theme['axis_text_color']
    )
    
    # Update colorbar if present (for heatmaps)
    fig.update_coloraxes(
        colorbar=dict(
            tickfont=dict(color=theme['axis_text_color']),
            titlefont=dict(color=theme['axis_title_color'])
        )
    )
    
    return fig

# -----------------------------------------------------------------------------
# DATA LOADING FUNCTION - FIXED VERSION
# -----------------------------------------------------------------------------

@st.cache_data
def load_data():
    """Load and preprocess the data with multiple fallback approaches"""
    
    # Define the correct column names
    column_names = ['ID', 'Age', 'Experience', 'Income', 'ZIPCode', 'Family', 
                    'CCAvg', 'Education', 'Mortgage', 'PersonalLoan', 
                    'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard']
    
    df = None
    
    # Check if file exists
    if not os.path.exists('UniversalBank.csv'):
        raise FileNotFoundError("UniversalBank.csv not found in the current directory")
    
    # Read the file content to understand its structure
    with open('UniversalBank.csv', 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.strip().split('\n')
    
    # Approach 1: Check if it's a standard CSV format
    try:
        df_test = pd.read_csv('UniversalBank.csv', nrows=2)
        if 'Income' in df_test.columns or 'income' in df_test.columns.str.lower():
            # Standard format detected
            df = pd.read_csv('UniversalBank.csv')
            df.columns = df.columns.str.strip()
            
            # Rename columns for consistency
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
    
    # Approach 2: Handle the problematic format with "Universal Bank Customer Profiles"
    if df is None or 'Income' not in df.columns:
        try:
            # Parse lines manually
            data_rows = []
            
            for line in lines:
                # Split by comma
                parts = line.split(',')
                
                # Find numeric values (the actual data)
                numeric_values = []
                for part in parts:
                    part = part.strip().strip('"').strip("'")
                    if part:
                        try:
                            # Try to convert to float
                            val = float(part)
                            numeric_values.append(val)
                        except ValueError:
                            # Skip non-numeric values like "Universal Bank Customer Profiles"
                            continue
                
                # If we have exactly 14 numeric values, it's a data row
                if len(numeric_values) == 14:
                    data_rows.append(numeric_values)
                elif len(numeric_values) > 14:
                    # Take the last 14 values (data usually at the end)
                    data_rows.append(numeric_values[-14:])
            
            if len(data_rows) > 0:
                df = pd.DataFrame(data_rows, columns=column_names)
        except Exception as e:
            pass
    
    # Approach 3: Try reading with skiprows
    if df is None or 'Income' not in df.columns:
        try:
            for skip in range(5):  # Try skipping 0-4 rows
                df_temp = pd.read_csv('UniversalBank.csv', skiprows=skip, header=None)
                df_temp = df_temp.dropna(axis=1, how='all')
                
                if len(df_temp.columns) >= 14:
                    # Check if first row looks like data (numeric)
                    try:
                        first_val = float(df_temp.iloc[0, 0])
                        # Looks like data, use it
                        df = df_temp.iloc[:, :14].copy()
                        df.columns = column_names
                        break
                    except:
                        # First row might be header, try next skip value
                        continue
        except:
            pass
    
    # Approach 4: Extract data using regex pattern matching
    if df is None or 'Income' not in df.columns:
        try:
            import re
            data_rows = []
            
            # Pattern to match rows of numbers separated by commas
            for line in lines:
                # Remove any text that's not numbers, commas, dots, or minus signs
                numbers = re.findall(r'-?\d+\.?\d*', line)
                if len(numbers) >= 14:
                    data_rows.append([float(n) for n in numbers[:14]])
            
            if len(data_rows) > 0:
                df = pd.DataFrame(data_rows, columns=column_names)
        except:
            pass
    
    # Verify we have valid data
    if df is None:
        raise ValueError("Could not parse the CSV file. Please check the file format.")
    
    if 'Income' not in df.columns:
        raise ValueError(f"'Income' column not found. Available columns: {list(df.columns)}")
    
    # Ensure numeric types for all columns
    for col in column_names:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN in critical columns
    critical_cols = ['Income', 'Age', 'PersonalLoan']
    existing_critical = [col for col in critical_cols if col in df.columns]
    df = df.dropna(subset=existing_critical)
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Create derived columns
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
# VISUALIZATION FUNCTIONS - ALL UPDATED WITH THEME SUPPORT
# -----------------------------------------------------------------------------

def create_kpi_metrics(df):
    """Create KPI metrics cards"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="📊 Total Customers",
            value=f"{len(df):,}",
            delta=None
        )
    
    with col2:
        loan_rate = (df['PersonalLoan'].sum() / len(df)) * 100
        st.metric(
            label="💳 Loan Acceptance Rate",
            value=f"{loan_rate:.1f}%",
            delta=f"{df['PersonalLoan'].sum()} customers"
        )
    
    with col3:
        avg_income = df['Income'].mean()
        st.metric(
            label="💰 Average Income",
            value=f"${avg_income:,.0f}K",
            delta=None
        )
    
    with col4:
        avg_age = df['Age'].mean()
        st.metric(
            label="👥 Average Age",
            value=f"{avg_age:.1f} years",
            delta=None
        )
    
    with col5:
        cc_avg = df['CCAvg'].mean()
        st.metric(
            label="💳 Avg CC Spending",
            value=f"${cc_avg:,.2f}K/month",
            delta=None
        )

def create_income_distribution(df):
    """Create income distribution visualization"""
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=('Income Distribution', 'Income by Loan Status'),
                        specs=[[{"type": "histogram"}, {"type": "box"}]])
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=df['Income'], nbinsx=50, name='Income Distribution',
                     marker_color=current_theme['accent_color'], opacity=0.7),
        row=1, col=1
    )
    
    # Box plot by loan status
    colors = [current_theme['accent_color'], current_theme['success_color']]
    for i, status in enumerate(df['LoanStatus'].unique()):
        fig.add_trace(
            go.Box(y=df[df['LoanStatus']==status]['Income'], 
                   name=status, boxmean=True,
                   marker_color=colors[i % len(colors)]),
            row=1, col=2
        )
    
    fig.update_layout(height=400, showlegend=True, title_text="Income Analysis")
    
    # Update subplot title colors
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color=current_theme['axis_title_color'])
    
    return apply_plotly_theme(fig)

def create_age_analysis(df):
    """Create age distribution analysis"""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Age Distribution', 'Age vs Income by Loan Status'))
    
    # Age histogram
    fig.add_trace(
        go.Histogram(x=df['Age'], nbinsx=30, name='Age',
                     marker_color=current_theme['success_color'], opacity=0.7),
        row=1, col=1
    )
    
    # Scatter plot
    colors = {'Accepted': current_theme['error_color'], 'Not Accepted': current_theme['accent_color']}
    for status in df['LoanStatus'].unique():
        subset = df[df['LoanStatus']==status]
        fig.add_trace(
            go.Scatter(x=subset['Age'], y=subset['Income'],
                       mode='markers', name=status,
                       marker=dict(color=colors.get(status, '#888888'), 
                                   opacity=0.5, size=5)),
            row=1, col=2
        )
    
    fig.update_layout(height=400, title_text="Age Analysis")
    
    # Update subplot title colors
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color=current_theme['axis_title_color'])
    
    return apply_plotly_theme(fig)

def create_education_analysis(df):
    """Create education level analysis"""
    edu_loan = df.groupby('EducationLevel')['PersonalLoan'].agg(['sum', 'count']).reset_index()
    edu_loan['rate'] = (edu_loan['sum'] / edu_loan['count']) * 100
    
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Customers by Education', 'Loan Acceptance by Education'),
                        specs=[[{"type": "pie"}, {"type": "bar"}]])
    
    # Pie chart
    fig.add_trace(
        go.Pie(labels=edu_loan['EducationLevel'], values=edu_loan['count'],
               hole=0.4, marker_colors=[current_theme['accent_color'], 
                                        current_theme['success_color'], 
                                        current_theme['warning_color']],
               textfont=dict(color=current_theme['axis_text_color'])),
        row=1, col=1
    )
    
    # Bar chart
    fig.add_trace(
        go.Bar(x=edu_loan['EducationLevel'], y=edu_loan['rate'],
               marker_color=[current_theme['accent_color'], 
                            current_theme['success_color'], 
                            current_theme['warning_color']],
               text=[f"{r:.1f}%" for r in edu_loan['rate']],
               textposition='outside',
               textfont=dict(color=current_theme['axis_text_color'])),
        row=1, col=2
    )
    
    fig.update_layout(height=400, title_text="Education Analysis")
    
    # Update subplot title colors
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color=current_theme['axis_title_color'])
    
    return apply_plotly_theme(fig)

def create_family_analysis(df):
    """Create family size analysis"""
    family_loan = df.groupby('Family')['PersonalLoan'].agg(['sum', 'count']).reset_index()
    family_loan['rate'] = (family_loan['sum'] / family_loan['count']) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=family_loan['Family'],
        y=family_loan['count'],
        name='Total Customers',
        marker_color=current_theme['accent_color'],
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=family_loan['Family'],
        y=family_loan['rate'],
        name='Loan Acceptance Rate (%)',
        marker_color=current_theme['error_color'],
        yaxis='y2',
        mode='lines+markers',
        line=dict(width=3)
    ))
    
    fig.update_layout(
        title='Family Size Analysis',
        yaxis=dict(
            title='Number of Customers', 
            side='left',
            tickfont=dict(color=current_theme['axis_text_color']),
            titlefont=dict(color=current_theme['axis_title_color'])
        ),
        yaxis2=dict(
            title='Loan Acceptance Rate (%)', 
            side='right', 
            overlaying='y',
            tickfont=dict(color=current_theme['axis_text_color']),
            titlefont=dict(color=current_theme['axis_title_color'])
        ),
        height=400,
        legend=dict(x=0.1, y=1.1, orientation='h', font=dict(color=current_theme['legend_text_color']))
    )
    
    return apply_plotly_theme(fig)

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    numeric_cols = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 
                    'Education', 'Mortgage', 'PersonalLoan', 
                    'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard']
    
    existing_cols = [col for col in numeric_cols if col in df.columns]
    corr_matrix = df[existing_cols].corr()
    
    # Choose text color based on theme for better visibility on heatmap
    heatmap_text_color = '#000000' if st.session_state.theme == 'light' else '#FFFFFF'
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10, "color": heatmap_text_color},
        hoverongaps=False,
        colorbar=dict(
            tickfont=dict(color=current_theme['axis_text_color']),
            titlefont=dict(color=current_theme['axis_title_color'])
        )
    ))
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        height=600,
        xaxis_tickangle=-45
    )
    
    return apply_plotly_theme(fig)

def create_mortgage_analysis(df):
    """Create mortgage analysis visualization"""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Mortgage Distribution', 'Mortgage vs Income by Loan Status'))
    
    # Mortgage histogram (excluding zeros for better visualization)
    mortgage_data = df[df['Mortgage'] > 0]['Mortgage']
    fig.add_trace(
        go.Histogram(x=mortgage_data, nbinsx=30, name='Mortgage',
                     marker_color='#8B5CF6', opacity=0.7),
        row=1, col=1
    )
    
    # Scatter plot
    colors = {'Accepted': current_theme['error_color'], 'Not Accepted': current_theme['accent_color']}
    for status in df['LoanStatus'].unique():
        subset = df[df['LoanStatus']==status]
        fig.add_trace(
            go.Scatter(x=subset['Income'], y=subset['Mortgage'],
                       mode='markers', name=status,
                       marker=dict(color=colors.get(status, '#888888'), 
                                   opacity=0.5, size=5)),
            row=1, col=2
        )
    
    fig.update_layout(height=400, title_text="Mortgage Analysis")
    
    # Update subplot title colors
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color=current_theme['axis_title_color'])
    
    return apply_plotly_theme(fig)

def create_cc_spending_analysis(df):
    """Create credit card spending analysis"""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('CC Spending Distribution', 'CC Spending by Loan Status'))
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=df['CCAvg'], nbinsx=40, name='CC Spending',
                     marker_color=current_theme['warning_color'], opacity=0.7),
        row=1, col=1
    )
    
    # Violin plot
    colors = [current_theme['accent_color'], current_theme['success_color']]
    for i, status in enumerate(df['LoanStatus'].unique()):
        fig.add_trace(
            go.Violin(y=df[df['LoanStatus']==status]['CCAvg'],
                      name=status, box_visible=True, meanline_visible=True,
                      marker_color=colors[i % len(colors)]),
            row=1, col=2
        )
    
    fig.update_layout(height=400, title_text="Credit Card Spending Analysis")
    
    # Update subplot title colors
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color=current_theme['axis_title_color'])
    
    return apply_plotly_theme(fig)

def create_services_analysis(df):
    """Create banking services analysis"""
    services = ['SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard']
    existing_services = [s for s in services if s in df.columns]
    
    service_data = []
    for service in existing_services:
        total = df[service].sum()
        loan_accepters = df[df['PersonalLoan']==1][service].sum()
        rate = (loan_accepters / total * 100) if total > 0 else 0
        service_data.append({
            'Service': service,
            'Total Users': total,
            'Loan Accepters': loan_accepters,
            'Acceptance Rate': rate
        })
    
    service_df = pd.DataFrame(service_data)
    
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Service Usage', 'Loan Acceptance by Service'))
    
    # Bar chart for service usage
    fig.add_trace(
        go.Bar(x=service_df['Service'], y=service_df['Total Users'],
               name='Total Users', marker_color=current_theme['accent_color']),
        row=1, col=1
    )
    
    # Bar chart for loan acceptance rate
    fig.add_trace(
        go.Bar(x=service_df['Service'], y=service_df['Acceptance Rate'],
               name='Loan Acceptance Rate (%)', marker_color=current_theme['success_color'],
               text=[f"{r:.1f}%" for r in service_df['Acceptance Rate']],
               textposition='outside',
               textfont=dict(color=current_theme['axis_text_color'])),
        row=1, col=2
    )
    
    fig.update_layout(height=400, title_text="Banking Services Analysis")
    
    # Update subplot title colors
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color=current_theme['axis_title_color'])
    
    return apply_plotly_theme(fig)

# -----------------------------------------------------------------------------
# MACHINE LEARNING FUNCTIONS
# -----------------------------------------------------------------------------

@st.cache_resource
def train_models(df):
    """Train multiple ML models"""
    # Prepare features
    feature_cols = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 
                    'Education', 'Mortgage', 'SecuritiesAccount', 
                    'CDAccount', 'Online', 'CreditCard']
    
    existing_features = [col for col in feature_cols if col in df.columns]
    
    X = df[existing_features].copy()
    y = df['PersonalLoan'].copy()
    
    # Handle any remaining NaN values
    X = X.fillna(X.median())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        # Use scaled data for KNN and Logistic Regression
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
    
    # Get feature importance from Random Forest
    feature_importance = pd.DataFrame({
        'Feature': existing_features,
        'Importance': trained_models['Random Forest'].feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return results, trained_models, feature_importance, scaler, X_test, y_test, existing_features

def create_model_comparison(results):
    """Create model comparison visualization"""
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
    """Create ROC curves for all models"""
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
    
    # Add diagonal line
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
    """Create confusion matrices visualization"""
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=list(results.keys()),
                        vertical_spacing=0.15,
                        horizontal_spacing=0.1)
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    # Choose text color for confusion matrix cells
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
    
    # Update subplot title colors
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color=current_theme['axis_title_color'])
    
    return apply_plotly_theme(fig)

def create_feature_importance(feature_importance):
    """Create feature importance visualization"""
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
    """Perform K-Means clustering"""
    features_for_clustering = ['Age', 'Income', 'CCAvg', 'Mortgage']
    existing_features = [f for f in features_for_clustering if f in df.columns]
    
    X_cluster = df[existing_features].copy()
    X_cluster = X_cluster.fillna(X_cluster.median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Perform K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]
    
    return df, kmeans

def create_cluster_visualization(df):
    """Create cluster visualization"""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Customer Segments (PCA)', 'Cluster Characteristics'))
    
    # PCA scatter plot
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
    
    # Cluster characteristics - bar chart
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
    
    # Update subplot title colors
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color=current_theme['axis_title_color'])
    
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
    # Sidebar - Theme Toggle at the TOP
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Theme toggle button with icon
        theme_icon = "🌙" if st.session_state.theme == 'light' else "☀️"
        theme_text = "Switch to Dark Mode" if st.session_state.theme == 'light' else "Switch to Light Mode"
        
        if st.button(f"{theme_icon} {theme_text}", key="theme_toggle", use_container_width=True):
            toggle_theme()
            st.rerun()
        
        # Display current theme
        current_theme_display = "🌞 Light Mode" if st.session_state.theme == 'light' else "🌙 Dark Mode"
        st.markdown(f"**Current Theme:** {current_theme_display}")
        
        st.markdown("---")
        
        # Debug info toggle
        if st.checkbox("🔧 Show Debug Info", value=False):
            st.write("**Current Directory:**", os.getcwd())
            st.write("**Files in Directory:**")
            try:
                files = os.listdir('.')
                for f in files:
                    st.write(f"  - {f}")
            except Exception as e:
                st.error(f"Error listing files: {e}")
            
            if os.path.exists('UniversalBank.csv'):
                st.success("✅ UniversalBank.csv found!")
                try:
                    with open('UniversalBank.csv', 'r') as f:
                        first_line = f.readline()[:200]
                    st.write("**First line preview:**")
                    st.code(first_line)
                except Exception as e:
                    st.error(f"Error reading file: {e}")
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
        st.sidebar.success(f"✅ Data loaded successfully! ({len(df)} records)")
    except FileNotFoundError:
        st.error("""
        ⚠️ **File Not Found Error**
        
        The file `UniversalBank.csv` was not found in the current directory.
        
        Please ensure the file is in the same folder as `app.py`.
        """)
        st.stop()
    except Exception as e:
        st.error(f"""
        ⚠️ **Error loading data:** {str(e)}
        
        Please ensure 'UniversalBank.csv' is in the same directory as the app and is properly formatted.
        
        **Troubleshooting Tips:**
        1. Check if the file exists in the same folder as app.py
        2. Enable "Show Debug Info" in the sidebar
        3. Verify the CSV file format
        """)
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("🎯 Filters")
    
    # Income filter
    income_range = st.sidebar.slider(
        "Income Range ($K)",
        int(df['Income'].min()),
        int(df['Income'].max()),
        (int(df['Income'].min()), int(df['Income'].max()))
    )
    
    # Age filter
    age_range = st.sidebar.slider(
        "Age Range",
        int(df['Age'].min()),
        int(df['Age'].max()),
        (int(df['Age'].min()), int(df['Age'].max()))
    )
    
    # Education filter
    education_options = df['EducationLevel'].dropna().unique().tolist()
    selected_education = st.sidebar.multiselect(
        "Education Level",
        options=education_options,
        default=education_options
    )
    
    # Family size filter
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
        "📊 Overview", 
        "📈 Detailed Analysis", 
        "🤖 ML Models", 
        "👥 Customer Segments",
        "🔮 Predictions"
    ])
    
    # TAB 1: Overview
    with tab1:
        st.markdown("### 📊 Key Performance Indicators")
        create_kpi_metrics(filtered_df)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Income Distribution")
            fig_income = create_income_distribution(filtered_df)
            st.plotly_chart(fig_income, use_container_width=True)
        
        with col2:
            st.markdown("### Age Analysis")
            fig_age = create_age_analysis(filtered_df)
            st.plotly_chart(fig_age, use_container_width=True)
        
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### Education Analysis")
            fig_edu = create_education_analysis(filtered_df)
            st.plotly_chart(fig_edu, use_container_width=True)
        
        with col4:
            st.markdown("### Family Size Impact")
            fig_family = create_family_analysis(filtered_df)
            st.plotly_chart(fig_family, use_container_width=True)
    
    # TAB 2: Detailed Analysis
    with tab2:
        st.markdown("### 📈 Detailed Feature Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Mortgage Analysis")
            fig_mortgage = create_mortgage_analysis(filtered_df)
            st.plotly_chart(fig_mortgage, use_container_width=True)
        
        with col2:
            st.markdown("#### Credit Card Spending")
            fig_cc = create_cc_spending_analysis(filtered_df)
            st.plotly_chart(fig_cc, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("#### Banking Services Usage")
        fig_services = create_services_analysis(filtered_df)
        st.plotly_chart(fig_services, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("#### Correlation Matrix")
        fig_corr = create_correlation_heatmap(filtered_df)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Data preview
        st.markdown("---")
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
        
        # Model comparison
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
        
        # Cluster descriptions
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
        
        # Make sure models are trained
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
            # Create input data
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
            
            # Ensure columns match training features
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

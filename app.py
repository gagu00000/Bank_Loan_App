import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="UniversalBank Analytics Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# THEME CONFIGURATION
# ============================================
def get_theme_colors(theme):
    """Return color scheme based on selected theme"""
    if theme == "Dark":
        return {
            'bg_color': '#0E1117',
            'secondary_bg': '#262730',
            'text_color': '#FAFAFA',
            'primary_color': '#00D4FF',
            'secondary_color': '#FF6B6B',
            'accent_color': '#4ECDC4',
            'grid_color': '#333333',
            'card_bg': '#1E1E1E',
            'plot_bg': '#0E1117',
            'paper_bg': '#0E1117',
            'colorscale': 'Viridis',
            'positive_color': '#00D4FF',
            'negative_color': '#FF6B6B',
            'neutral_color': '#888888'
        }
    else:  # Light theme
        return {
            'bg_color': '#FFFFFF',
            'secondary_bg': '#F0F2F6',
            'text_color': '#1E1E1E',
            'primary_color': '#1E88E5',
            'secondary_color': '#E53935',
            'accent_color': '#43A047',
            'grid_color': '#E0E0E0',
            'card_bg': '#FFFFFF',
            'plot_bg': '#FFFFFF',
            'paper_bg': '#FFFFFF',
            'colorscale': 'Blues',
            'positive_color': '#1E88E5',
            'negative_color': '#E53935',
            'neutral_color': '#757575'
        }

def apply_theme_to_fig(fig, colors):
    """Apply theme colors to a plotly figure"""
    fig.update_layout(
        plot_bgcolor=colors['plot_bg'],
        paper_bgcolor=colors['paper_bg'],
        font=dict(color=colors['text_color'], size=12),
        title_font=dict(color=colors['text_color'], size=16),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color=colors['text_color'])
        ),
        xaxis=dict(
            gridcolor=colors['grid_color'],
            linecolor=colors['grid_color'],
            tickfont=dict(color=colors['text_color']),
            title_font=dict(color=colors['text_color'])
        ),
        yaxis=dict(
            gridcolor=colors['grid_color'],
            linecolor=colors['grid_color'],
            tickfont=dict(color=colors['text_color']),
            title_font=dict(color=colors['text_color'])
        )
    )
    return fig

# ============================================
# CUSTOM CSS FOR THEMES
# ============================================
def load_css(theme):
    colors = get_theme_colors(theme)
    css = f"""
    <style>
        /* Main container */
        .stApp {{
            background-color: {colors['bg_color']};
        }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {colors['secondary_bg']};
        }}
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            color: {colors['text_color']} !important;
        }}
        
        /* Text */
        p, span, label {{
            color: {colors['text_color']};
        }}
        
        /* Metrics */
        [data-testid="stMetricValue"] {{
            color: {colors['primary_color']} !important;
            font-size: 2rem !important;
        }}
        
        [data-testid="stMetricLabel"] {{
            color: {colors['text_color']} !important;
        }}
        
        /* Cards */
        .metric-card {{
            background-color: {colors['card_bg']};
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border: 1px solid {colors['grid_color']};
        }}
        
        /* Insight boxes */
        .insight-box {{
            background-color: {colors['secondary_bg']};
            border-left: 4px solid {colors['primary_color']};
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 8px 8px 0;
            color: {colors['text_color']};
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background-color: {colors['secondary_bg']};
            border-radius: 8px;
            color: {colors['text_color']};
            padding: 10px 20px;
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: {colors['primary_color']};
            color: white;
        }}
        
        /* Expander */
        .streamlit-expanderHeader {{
            background-color: {colors['secondary_bg']};
            color: {colors['text_color']} !important;
        }}
        
        /* Selectbox */
        .stSelectbox > div > div {{
            background-color: {colors['secondary_bg']};
            color: {colors['text_color']};
        }}
        
        /* Dataframe */
        .dataframe {{
            color: {colors['text_color']} !important;
        }}
        
        /* Main title */
        .main-title {{
            font-size: 2.5rem;
            font-weight: 700;
            color: {colors['primary_color']};
            text-align: center;
            padding: 1rem;
            margin-bottom: 1rem;
        }}
        
        /* Subtitle */
        .subtitle {{
            font-size: 1.2rem;
            color: {colors['neutral_color']};
            text-align: center;
            margin-bottom: 2rem;
        }}
        
        /* KPI Container */
        .kpi-container {{
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 20px;
            margin: 20px 0;
        }}
        
        /* Section header */
        .section-header {{
            font-size: 1.5rem;
            font-weight: 600;
            color: {colors['text_color']};
            border-bottom: 2px solid {colors['primary_color']};
            padding-bottom: 10px;
            margin: 20px 0;
        }}
    </style>
    """
    return css

# ============================================
# DATA LOADING
# ============================================
@st.cache_data
def load_data():
    """Load and preprocess the data"""
    df = pd.read_csv('UniversalBank.csv')
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Rename columns for consistency
    column_mapping = {
        'ZIP Code': 'ZIPCode',
        'Personal Loan': 'PersonalLoan',
        'Securities Account': 'SecuritiesAccount',
        'CD Account': 'CDAccount',
        'CreditCard': 'CreditCard'
    }
    df.rename(columns=column_mapping, inplace=True)
    
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

# ============================================
# VISUALIZATION FUNCTIONS
# ============================================

def create_income_histogram(df, colors):
    """Visualization 1a: Income distribution histogram by loan status"""
    fig = px.histogram(
        df,
        x='Income',
        color='LoanStatus',
        barmode='overlay',
        nbins=40,
        color_discrete_map={
            'Not Accepted': colors['primary_color'],
            'Accepted': colors['secondary_color']
        },
        labels={'Income': 'Annual Income ($000)', 'count': 'Number of Customers'},
        title='📊 Income Distribution by Personal Loan Status',
        opacity=0.7
    )
    
    fig.update_layout(
        xaxis_title='Annual Income ($000)',
        yaxis_title='Number of Customers',
        legend_title='Loan Status',
        bargap=0.1,
        hovermode='x unified'
    )
    
    # Add animation
    fig.update_traces(
        hovertemplate='<b>Income:</b> $%{x}K<br><b>Count:</b> %{y}<extra></extra>'
    )
    
    return apply_theme_to_fig(fig, colors)


def create_age_histogram(df, colors):
    """Visualization 1b: Age distribution histogram by loan status"""
    fig = px.histogram(
        df,
        x='Age',
        color='LoanStatus',
        barmode='overlay',
        nbins=30,
        color_discrete_map={
            'Not Accepted': colors['primary_color'],
            'Accepted': colors['secondary_color']
        },
        labels={'Age': 'Customer Age (Years)', 'count': 'Number of Customers'},
        title='📊 Age Distribution by Personal Loan Status',
        opacity=0.7
    )
    
    fig.update_layout(
        xaxis_title='Customer Age (Years)',
        yaxis_title='Number of Customers',
        legend_title='Loan Status',
        bargap=0.1
    )
    
    return apply_theme_to_fig(fig, colors)


def create_ccavg_income_scatter(df, colors):
    """Visualization 2: CCAvg vs Income scatter plot"""
    fig = px.scatter(
        df,
        x='Income',
        y='CCAvg',
        color='LoanStatus',
        color_discrete_map={
            'Not Accepted': colors['primary_color'],
            'Accepted': colors['secondary_color']
        },
        trendline='ols',
        hover_data=['Age', 'EducationLevel', 'Family'],
        labels={
            'Income': 'Annual Income ($000)',
            'CCAvg': 'Avg. Monthly CC Spending ($000)'
        },
        title='💳 Credit Card Spending vs Income by Loan Status',
        opacity=0.6
    )
    
    fig.update_traces(
        marker=dict(size=8, line=dict(width=1, color='white')),
        hovertemplate='<b>Income:</b> $%{x}K<br><b>CC Avg:</b> $%{y}K<br><b>Age:</b> %{customdata[0]}<extra></extra>'
    )
    
    fig.update_layout(
        xaxis_title='Annual Income ($000)',
        yaxis_title='Monthly CC Spending ($000)',
        legend_title='Loan Status'
    )
    
    return apply_theme_to_fig(fig, colors)


def create_zipcode_analysis(df, colors):
    """Visualization 3: ZIP Code vs Income vs Personal Loan"""
    # Group by ZIP prefix
    zip_analysis = df.groupby('ZIPPrefix').agg({
        'Income': 'mean',
        'PersonalLoan': ['sum', 'count', 'mean'],
        'ID': 'count'
    }).reset_index()
    
    zip_analysis.columns = ['ZIPPrefix', 'AvgIncome', 'LoansAccepted', 
                            'TotalCustomers', 'ConvRate', 'Count']
    
    zip_analysis['ConvRatePercent'] = zip_analysis['ConvRate'] * 100
    
    # Filter for ZIP codes with sufficient data
    zip_analysis = zip_analysis[zip_analysis['TotalCustomers'] >= 10]
    
    fig = px.scatter(
        zip_analysis,
        x='ZIPPrefix',
        y='AvgIncome',
        size='TotalCustomers',
        color='ConvRatePercent',
        color_continuous_scale='RdYlGn',
        hover_data=['LoansAccepted', 'TotalCustomers', 'ConvRatePercent'],
        labels={
            'ZIPPrefix': 'ZIP Code Prefix',
            'AvgIncome': 'Average Income ($000)',
            'ConvRatePercent': 'Conversion Rate (%)'
        },
        title='📍 ZIP Code Analysis: Income vs Conversion Rate'
    )
    
    fig.update_traces(
        hovertemplate='<b>ZIP:</b> %{x}<br><b>Avg Income:</b> $%{y:.1f}K<br>' +
                      '<b>Customers:</b> %{customdata[1]}<br><b>Conv Rate:</b> %{customdata[2]:.1f}%<extra></extra>'
    )
    
    fig.update_layout(
        xaxis_title='ZIP Code Prefix (3-digit)',
        yaxis_title='Average Income ($000)',
        coloraxis_colorbar_title='Conv Rate (%)'
    )
    
    return apply_theme_to_fig(fig, colors)


def create_correlation_heatmap(df, colors):
    """Visualization 4: Correlation heatmap for all numerical columns"""
    numerical_cols = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 
                      'Education', 'Mortgage', 'PersonalLoan', 
                      'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard']
    
    corr_matrix = df[numerical_cols].corr()
    
    # Create custom colorscale based on theme
    if colors['bg_color'] == '#0E1117':  # Dark theme
        colorscale = [
            [0, '#FF6B6B'],
            [0.5, '#1E1E1E'],
            [1, '#00D4FF']
        ]
    else:  # Light theme
        colorscale = [
            [0, '#E53935'],
            [0.5, '#FFFFFF'],
            [1, '#1E88E5']
        ]
    
    fig = px.imshow(
        corr_matrix,
        x=numerical_cols,
        y=numerical_cols,
        color_continuous_scale=colorscale,
        zmin=-1,
        zmax=1,
        text_auto='.2f',
        labels=dict(color='Correlation'),
        title='🔥 Correlation Heatmap - All Variables'
    )
    
    fig.update_layout(
        xaxis_title='',
        yaxis_title='',
        width=800,
        height=700
    )
    
    fig.update_traces(
        hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
    )
    
    return apply_theme_to_fig(fig, colors)


def create_family_income_analysis(df, colors):
    """Visualization 5: Family Size vs Income vs Mortgage/CCAvg vs Personal Loan"""
    fig = px.scatter(
        df,
        x='Family',
        y='Income',
        size='CCAvg',
        color='LoanStatus',
        color_discrete_map={
            'Not Accepted': colors['primary_color'],
            'Accepted': colors['secondary_color']
        },
        hover_data=['Mortgage', 'CCAvg', 'EducationLevel'],
        labels={
            'Family': 'Family Size',
            'Income': 'Annual Income ($000)',
            'CCAvg': 'CC Spending'
        },
        title='👨‍👩‍👧‍👦 Family Size vs Income (Bubble Size = CC Spending)',
        animation_frame='EducationLevel' if len(df['EducationLevel'].unique()) > 1 else None,
        opacity=0.7
    )
    
    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        xaxis_title='Family Size',
        yaxis_title='Annual Income ($000)',
        legend_title='Loan Status'
    )
    
    return apply_theme_to_fig(fig, colors)


def create_family_mortgage_heatmap(df, colors):
    """Visualization 5b: Heatmap for Family vs Income conversion rates"""
    df_temp = df.copy()
    df_temp['IncomeRange'] = pd.cut(
        df_temp['Income'],
        bins=[0, 50, 100, 150, 200, 250],
        labels=['<$50K', '$50-100K', '$100-150K', '$150-200K', '>$200K']
    )
    
    heatmap_data = df_temp.pivot_table(
        values='PersonalLoan',
        index='Family',
        columns='IncomeRange',
        aggfunc='mean'
    ) * 100
    
    fig = px.imshow(
        heatmap_data,
        text_auto='.1f',
        color_continuous_scale='RdYlGn',
        labels=dict(x='Income Range', y='Family Size', color='Conv Rate (%)'),
        title='📈 Conversion Rate by Family Size & Income'
    )
    
    fig.update_traces(
        hovertemplate='<b>Family Size:</b> %{y}<br><b>Income:</b> %{x}<br><b>Conv Rate:</b> %{z:.1f}%<extra></extra>'
    )
    
    return apply_theme_to_fig(fig, colors)


def create_product_analysis(df, colors):
    """Visualization 6: Securities vs CD vs Credit Card vs Personal Loan"""
    products = {
        'Securities Account': 'SecuritiesAccount',
        'CD Account': 'CDAccount',
        'Credit Card': 'CreditCard',
        'Online Banking': 'Online'
    }
    
    results = []
    for product_name, col_name in products.items():
        for val in [0, 1]:
            subset = df[df[col_name] == val]
            conv_rate = subset['PersonalLoan'].mean() * 100
            results.append({
                'Product': product_name,
                'HasProduct': 'Yes' if val == 1 else 'No',
                'ConversionRate': conv_rate,
                'CustomerCount': len(subset)
            })
    
    result_df = pd.DataFrame(results)
    
    fig = px.bar(
        result_df,
        x='Product',
        y='ConversionRate',
        color='HasProduct',
        barmode='group',
        text='ConversionRate',
        color_discrete_map={
            'No': colors['primary_color'],
            'Yes': colors['secondary_color']
        },
        labels={
            'ConversionRate': 'Conversion Rate (%)',
            'HasProduct': 'Has Product'
        },
        title='🏦 Conversion Rate by Product Ownership'
    )
    
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Conv Rate: %{y:.1f}%<extra></extra>'
    )
    
    fig.update_layout(
        xaxis_title='Bank Product',
        yaxis_title='Conversion Rate (%)',
        legend_title='Has Product'
    )
    
    return apply_theme_to_fig(fig, colors)


def create_product_combination_chart(df, colors):
    """Visualization 6b: Product combination analysis"""
    # Calculate conversion for key combinations
    combinations = []
    
    # CD Account holders
    cd_holders = df[df['CDAccount'] == 1]['PersonalLoan'].mean() * 100
    combinations.append({'Combination': 'CD Account = Yes', 'ConvRate': cd_holders, 'Count': len(df[df['CDAccount'] == 1])})
    
    # CD + Securities
    cd_sec = df[(df['CDAccount'] == 1) & (df['SecuritiesAccount'] == 1)]['PersonalLoan'].mean() * 100
    combinations.append({'Combination': 'CD + Securities', 'ConvRate': cd_sec, 'Count': len(df[(df['CDAccount'] == 1) & (df['SecuritiesAccount'] == 1)])})
    
    # Securities only
    sec_only = df[df['SecuritiesAccount'] == 1]['PersonalLoan'].mean() * 100
    combinations.append({'Combination': 'Securities = Yes', 'ConvRate': sec_only, 'Count': len(df[df['SecuritiesAccount'] == 1])})
    
    # Credit Card only
    cc_only = df[df['CreditCard'] == 1]['PersonalLoan'].mean() * 100
    combinations.append({'Combination': 'Credit Card = Yes', 'ConvRate': cc_only, 'Count': len(df[df['CreditCard'] == 1])})
    
    # Online only
    online_only = df[df['Online'] == 1]['PersonalLoan'].mean() * 100
    combinations.append({'Combination': 'Online = Yes', 'ConvRate': online_only, 'Count': len(df[df['Online'] == 1])})
    
    # No products
    no_products = df[(df['CDAccount'] == 0) & (df['SecuritiesAccount'] == 0) & 
                     (df['CreditCard'] == 0) & (df['Online'] == 0)]['PersonalLoan'].mean() * 100
    combinations.append({'Combination': 'No Products', 'ConvRate': no_products, 
                        'Count': len(df[(df['CDAccount'] == 0) & (df['SecuritiesAccount'] == 0) & 
                                       (df['CreditCard'] == 0) & (df['Online'] == 0)])})
    
    combo_df = pd.DataFrame(combinations)
    combo_df = combo_df.sort_values('ConvRate', ascending=True)
    
    fig = px.bar(
        combo_df,
        y='Combination',
        x='ConvRate',
        orientation='h',
        text='ConvRate',
        color='ConvRate',
        color_continuous_scale='RdYlGn',
        labels={'ConvRate': 'Conversion Rate (%)', 'Combination': 'Product Combination'},
        title='🎯 Conversion Rate by Product Combination'
    )
    
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside'
    )
    
    # Add baseline reference line
    baseline = df['PersonalLoan'].mean() * 100
    fig.add_vline(x=baseline, line_dash="dash", line_color=colors['text_color'],
                  annotation_text=f"Baseline: {baseline:.1f}%")
    
    return apply_theme_to_fig(fig, colors)


def create_box_plots(df, colors):
    """Visualization 7: Box and whisker plots for Income, CCAvg"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Income Distribution', 'CC Spending Distribution', 'Income by CC Ownership')
    )
    
    # Income Box Plot
    for status in ['Not Accepted', 'Accepted']:
        subset = df[df['LoanStatus'] == status]
        fig.add_trace(
            go.Box(
                y=subset['Income'],
                name=status,
                marker_color=colors['primary_color'] if status == 'Not Accepted' else colors['secondary_color'],
                boxmean=True
            ),
            row=1, col=1
        )
    
    # CCAvg Box Plot
    for status in ['Not Accepted', 'Accepted']:
        subset = df[df['LoanStatus'] == status]
        fig.add_trace(
            go.Box(
                y=subset['CCAvg'],
                name=status,
                marker_color=colors['primary_color'] if status == 'Not Accepted' else colors['secondary_color'],
                boxmean=True,
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Income by Credit Card
    for cc in [0, 1]:
        for status in ['Not Accepted', 'Accepted']:
            subset = df[(df['CreditCard'] == cc) & (df['LoanStatus'] == status)]
            cc_label = 'Has CC' if cc == 1 else 'No CC'
            fig.add_trace(
                go.Box(
                    y=subset['Income'],
                    name=f'{cc_label} - {status}',
                    marker_color=colors['primary_color'] if status == 'Not Accepted' else colors['secondary_color'],
                    showlegend=False
                ),
                row=1, col=3
            )
    
    fig.update_layout(
        title_text='📦 Distribution Analysis: Income & Credit Card Spending',
        height=500,
        showlegend=True,
        legend_title='Loan Status'
    )
    
    fig.update_yaxes(title_text='Income ($000)', row=1, col=1)
    fig.update_yaxes(title_text='CC Avg ($000)', row=1, col=2)
    fig.update_yaxes(title_text='Income ($000)', row=1, col=3)
    
    return apply_theme_to_fig(fig, colors)


def create_education_analysis(df, colors):
    """Visualization 8: Education vs Income vs Personal Loan"""
    fig = px.violin(
        df,
        x='EducationLevel',
        y='Income',
        color='LoanStatus',
        color_discrete_map={
            'Not Accepted': colors['primary_color'],
            'Accepted': colors['secondary_color']
        },
        box=True,
        points='outliers',
        labels={
            'EducationLevel': 'Education Level',
            'Income': 'Annual Income ($000)'
        },
        title='🎓 Income Distribution by Education Level & Loan Status'
    )
    
    fig.update_layout(
        xaxis_title='Education Level',
        yaxis_title='Annual Income ($000)',
        legend_title='Loan Status',
        violinmode='group'
    )
    
    return apply_theme_to_fig(fig, colors)


def create_education_conversion_chart(df, colors):
    """Visualization 8b: Education conversion rates"""
    edu_stats = df.groupby('EducationLevel').agg({
        'PersonalLoan': ['sum', 'count', 'mean'],
        'Income': 'mean'
    }).reset_index()
    
    edu_stats.columns = ['Education', 'Converted', 'Total', 'ConvRate', 'AvgIncome']
    edu_stats['ConvRatePercent'] = edu_stats['ConvRate'] * 100
    
    fig = px.bar(
        edu_stats,
        x='Education',
        y='ConvRatePercent',
        text='ConvRatePercent',
        color='AvgIncome',
        color_continuous_scale='Blues',
        labels={
            'ConvRatePercent': 'Conversion Rate (%)',
            'AvgIncome': 'Avg Income ($K)'
        },
        title='📚 Conversion Rate by Education Level'
    )
    
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside'
    )
    
    return apply_theme_to_fig(fig, colors)


def create_mortgage_analysis_3d(df, colors):
    """Visualization 9: Mortgage vs Income vs Family Size vs Personal Loan"""
    # Sample data for better 3D visualization
    df_sample = df.sample(min(1000, len(df)), random_state=42)
    
    fig = px.scatter_3d(
        df_sample,
        x='Income',
        y='Mortgage',
        z='Family',
        color='LoanStatus',
        size='CCAvg',
        color_discrete_map={
            'Not Accepted': colors['primary_color'],
            'Accepted': colors['secondary_color']
        },
        hover_data=['Age', 'EducationLevel'],
        labels={
            'Income': 'Income ($K)',
            'Mortgage': 'Mortgage ($K)',
            'Family': 'Family Size'
        },
        title='🏠 3D View: Income × Mortgage × Family Size'
    )
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor=colors['plot_bg'], gridcolor=colors['grid_color']),
            yaxis=dict(backgroundcolor=colors['plot_bg'], gridcolor=colors['grid_color']),
            zaxis=dict(backgroundcolor=colors['plot_bg'], gridcolor=colors['grid_color']),
            bgcolor=colors['plot_bg']
        ),
        height=600
    )
    
    return apply_theme_to_fig(fig, colors)


def create_mortgage_heatmap(df, colors):
    """Visualization 9b: Mortgage holders conversion analysis"""
    df_mortgage = df[df['Mortgage'] > 0].copy()
    df_mortgage['IncomeRange'] = pd.cut(
        df_mortgage['Income'],
        bins=[0, 75, 100, 150, 250],
        labels=['<$75K', '$75-100K', '$100-150K', '>$150K']
    )
    
    if len(df_mortgage) > 0:
        heatmap_data = df_mortgage.pivot_table(
            values='PersonalLoan',
            index='Family',
            columns='IncomeRange',
            aggfunc='mean'
        ) * 100
        
        fig = px.imshow(
            heatmap_data,
            text_auto='.1f',
            color_continuous_scale='RdYlGn',
            labels=dict(x='Income Range', y='Family Size', color='Conv Rate (%)'),
            title='🏠 Conversion Rate: Mortgage Holders by Family & Income'
        )
        
        return apply_theme_to_fig(fig, colors)
    
    return None


# ============================================
# KPI CALCULATION FUNCTIONS
# ============================================
def calculate_kpis(df):
    """Calculate key performance indicators"""
    total_customers = len(df)
    converted = df['PersonalLoan'].sum()
    conversion_rate = (converted / total_customers) * 100
    avg_income_converted = df[df['PersonalLoan'] == 1]['Income'].mean()
    avg_income_not_converted = df[df['PersonalLoan'] == 0]['Income'].mean()
    cd_conversion = df[df['CDAccount'] == 1]['PersonalLoan'].mean() * 100
    
    return {
        'total_customers': total_customers,
        'converted': converted,
        'conversion_rate': conversion_rate,
        'avg_income_converted': avg_income_converted,
        'avg_income_not_converted': avg_income_not_converted,
        'cd_conversion': cd_conversion
    }


# ============================================
# MAIN APPLICATION
# ============================================
def main():
    # Initialize session state for theme
    if 'theme' not in st.session_state:
        st.session_state.theme = 'Dark'
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Dashboard Settings")
        
        # Theme Toggle
        st.markdown("### 🎨 Theme")
        theme = st.toggle('Light Mode', value=(st.session_state.theme == 'Light'))
        st.session_state.theme = 'Light' if theme else 'Dark'
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 📑 Navigation")
        page = st.radio(
            "Select Page:",
            [
                "🏠 Executive Overview",
                "📊 Distribution Analysis",
                "🔗 Correlation & Relationships",
                "👨‍👩‍👧‍👦 Demographic Analysis",
                "🏦 Product Analysis",
                "📋 Data Explorer"
            ]
        )
        
        st.markdown("---")
        st.markdown("### 📌 Quick Stats")
    
    # Get theme colors
    colors = get_theme_colors(st.session_state.theme)
    
    # Apply CSS
    st.markdown(load_css(st.session_state.theme), unsafe_allow_html=True)
    
    # Load data
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please ensure 'UniversalBank.csv' is in the same directory as the app.")
        return
    
    # Calculate KPIs
    kpis = calculate_kpis(df)
    
    # Update sidebar with quick stats
    with st.sidebar:
        st.metric("Total Customers", f"{kpis['total_customers']:,}")
        st.metric("Conversion Rate", f"{kpis['conversion_rate']:.1f}%")
        st.metric("CD Holder Conv.", f"{kpis['cd_conversion']:.1f}%")
    
    # ============================================
    # PAGE: EXECUTIVE OVERVIEW
    # ============================================
    if page == "🏠 Executive Overview":
        st.markdown('<h1 class="main-title">🏦 UniversalBank Analytics Dashboard</h1>', 
                    unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Personal Loan Campaign Performance Analysis</p>', 
                    unsafe_allow_html=True)
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Customers",
                value=f"{kpis['total_customers']:,}",
                delta="Database Size"
            )
        
        with col2:
            st.metric(
                label="Conversion Rate",
                value=f"{kpis['conversion_rate']:.1f}%",
                delta=f"{kpis['converted']:,} Converted"
            )
        
        with col3:
            st.metric(
                label="Avg Income (Converted)",
                value=f"${kpis['avg_income_converted']:.0f}K",
                delta=f"+${kpis['avg_income_converted'] - kpis['avg_income_not_converted']:.0f}K vs Others"
            )
        
        with col4:
            st.metric(
                label="CD Holders Conv. Rate",
                value=f"{kpis['cd_conversion']:.1f}%",
                delta=f"+{kpis['cd_conversion'] - kpis['conversion_rate']:.1f}% vs Baseline"
            )
        
        st.markdown("---")
        
        # Key Insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header">📈 Key Insights</div>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="insight-box">
                <strong>💰 Income Impact:</strong> Customers who accepted loans have 
                {((kpis['avg_income_converted'] / kpis['avg_income_not_converted']) - 1) * 100:.0f}% higher average income
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="insight-box">
                <strong>🏦 CD Account Effect:</strong> CD account holders are 
                {kpis['cd_conversion'] / kpis['conversion_rate']:.1f}x more likely to accept a loan
            </div>
            """, unsafe_allow_html=True)
            
            # Education insight
            edu_conv = df.groupby('EducationLevel')['PersonalLoan'].mean() * 100
            highest_edu = edu_conv.idxmax()
            st.markdown(f"""
            <div class="insight-box">
                <strong>🎓 Education:</strong> {highest_edu} education shows highest conversion at {edu_conv.max():.1f}%
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="section-header">🎯 Conversion by Segment</div>', unsafe_allow_html=True)
            
            # Quick segment analysis
            segment_data = df.groupby('EducationLevel')['PersonalLoan'].agg(['sum', 'count', 'mean']).reset_index()
            segment_data.columns = ['Education', 'Converted', 'Total', 'Rate']
            segment_data['Rate'] = segment_data['Rate'] * 100
            
            fig_segment = px.bar(
                segment_data,
                x='Education',
                y='Rate',
                text='Rate',
                color='Rate',
                color_continuous_scale='Blues'
            )
            fig_segment.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_segment.update_layout(
                showlegend=False,
                height=300,
                margin=dict(t=30, b=30)
            )
            st.plotly_chart(apply_theme_to_fig(fig_segment, colors), use_container_width=True)
        
        st.markdown("---")
        
        # Overview Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_income_histogram(df, colors), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_product_analysis(df, colors), use_container_width=True)
    
    # ============================================
    # PAGE: DISTRIBUTION ANALYSIS
    # ============================================
    elif page == "📊 Distribution Analysis":
        st.markdown('<h1 class="main-title">📊 Distribution Analysis</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Analyze income, age, and spending distributions</p>', 
                    unsafe_allow_html=True)
        
        # Tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📈 Income & Age Histograms", "📦 Box Plots", "💳 CC Analysis"])
        
        with tab1:
            st.markdown("### Income Distribution by Loan Status")
            st.markdown("""
            > **Visualization 1a:** This histogram shows the distribution of annual income, 
            separated by whether customers accepted the personal loan offer.
            """)
            st.plotly_chart(create_income_histogram(df, colors), use_container_width=True)
            
            st.markdown("---")
            
            st.markdown("### Age Distribution by Loan Status")
            st.markdown("""
            > **Visualization 1b:** This histogram compares age distributions between 
            customers who accepted and those who didn't accept the loan.
            """)
            st.plotly_chart(create_age_histogram(df, colors), use_container_width=True)
            
            # Statistical Summary
            with st.expander("📊 Statistical Summary"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Income Statistics:**")
                    income_stats = df.groupby('LoanStatus')['Income'].describe()
                    st.dataframe(income_stats.style.format("{:.2f}"))
                with col2:
                    st.markdown("**Age Statistics:**")
                    age_stats = df.groupby('LoanStatus')['Age'].describe()
                    st.dataframe(age_stats.style.format("{:.2f}"))
        
        with tab2:
            st.markdown("### Box and Whisker Plots")
            st.markdown("""
            > **Visualization 7:** Box plots showing the distribution of Income and Credit Card 
            spending, with median, quartiles, and outliers clearly visible.
            """)
            st.plotly_chart(create_box_plots(df, colors), use_container_width=True)
            
            # Key observations
            st.markdown("""
            <div class="insight-box">
                <strong>📌 Key Observations:</strong><br>
                • Loan acceptors have significantly higher median income (~$144K vs ~$64K)<br>
                • Credit card spending also notably higher for loan acceptors<br>
                • Clear separation in income distributions indicates strong predictive power
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### Credit Card Spending vs Income")
            st.markdown("""
            > **Visualization 2:** Scatter plot showing the relationship between monthly 
            credit card spending and annual income, colored by loan acceptance status.
            """)
            st.plotly_chart(create_ccavg_income_scatter(df, colors), use_container_width=True)
            
            # Correlation info
            corr_accepted = df[df['PersonalLoan'] == 1][['Income', 'CCAvg']].corr().iloc[0, 1]
            corr_not_accepted = df[df['PersonalLoan'] == 0][['Income', 'CCAvg']].corr().iloc[0, 1]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Correlation (Accepted)", f"{corr_accepted:.3f}")
            with col2:
                st.metric("Correlation (Not Accepted)", f"{corr_not_accepted:.3f}")
    
    # ============================================
    # PAGE: CORRELATION & RELATIONSHIPS
    # ============================================
    elif page == "🔗 Correlation & Relationships":
        st.markdown('<h1 class="main-title">🔗 Correlation & Relationships</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Explore variable relationships and geographic patterns</p>', 
                    unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔥 Correlation Heatmap", "📍 Geographic Analysis"])
        
        with tab1:
            st.markdown("### Full Correlation Matrix")
            st.markdown("""
            > **Visualization 4:** Heatmap showing correlations between all numerical variables. 
            Values range from -1 (strong negative) to +1 (strong positive correlation).
            """)
            
            st.plotly_chart(create_correlation_heatmap(df, colors), use_container_width=True)
            
            # Key correlations with Personal Loan
            st.markdown("### 🔑 Key Correlations with Personal Loan")
            
            numerical_cols = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 
                              'Education', 'Mortgage', 'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard']
            
            correlations = df[numerical_cols + ['PersonalLoan']].corr()['PersonalLoan'].drop('PersonalLoan').sort_values(ascending=False)
            
            corr_df = pd.DataFrame({
                'Variable': correlations.index,
                'Correlation': correlations.values
            })
            
            fig_corr = px.bar(
                corr_df,
                x='Correlation',
                y='Variable',
                orientation='h',
                color='Correlation',
                color_continuous_scale='RdBu_r',
                title='Correlation with Personal Loan Acceptance'
            )
            fig_corr.update_layout(height=400)
            st.plotly_chart(apply_theme_to_fig(fig_corr, colors), use_container_width=True)
        
        with tab2:
            st.markdown("### ZIP Code Analysis")
            st.markdown("""
            > **Visualization 3:** Analysis of conversion rates across different ZIP code regions.
            Bubble size represents customer count, color represents conversion rate.
            
            ⚠️ **Note:** ZIP code should not be used for prediction models as per data guidelines.
            """)
            
            st.plotly_chart(create_zipcode_analysis(df, colors), use_container_width=True)
            
            # Top ZIP codes table
            with st.expander("📋 Top ZIP Code Regions by Conversion"):
                zip_stats = df.groupby('ZIPPrefix').agg({
                    'Income': 'mean',
                    'PersonalLoan': ['sum', 'count', 'mean']
                }).reset_index()
                zip_stats.columns = ['ZIP Prefix', 'Avg Income', 'Converted', 'Total', 'Conv Rate']
                zip_stats = zip_stats[zip_stats['Total'] >= 10]
                zip_stats = zip_stats.sort_values('Conv Rate', ascending=False).head(10)
                zip_stats['Conv Rate'] = (zip_stats['Conv Rate'] * 100).round(1).astype(str) + '%'
                zip_stats['Avg Income'] = '$' + zip_stats['Avg Income'].round(0).astype(int).astype(str) + 'K'
                st.dataframe(zip_stats, use_container_width=True)
    
    # ============================================
    # PAGE: DEMOGRAPHIC ANALYSIS
    # ============================================
    elif page == "👨‍👩‍👧‍👦 Demographic Analysis":
        st.markdown('<h1 class="main-title">👨‍👩‍👧‍👦 Demographic Analysis</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Deep dive into family, education, and mortgage factors</p>', 
                    unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["👨‍👩‍👧 Family Analysis", "🎓 Education Impact", "🏠 Mortgage Analysis"])
        
        with tab1:
            st.markdown("### Family Size vs Income Analysis")
            st.markdown("""
            > **Visualization 5:** Explore how family size and income interact with loan acceptance.
            Bubble size represents credit card spending.
            """)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.plotly_chart(create_family_income_analysis(df, colors), use_container_width=True)
            
            with col2:
                st.markdown("#### Conversion by Family Size")
                family_conv = df.groupby('Family')['PersonalLoan'].mean() * 100
                for fam, conv in family_conv.items():
                    st.metric(f"Family Size: {fam}", f"{conv:.1f}%")
            
            st.markdown("---")
            st.markdown("### Conversion Heatmap: Family Size × Income")
            st.plotly_chart(create_family_mortgage_heatmap(df, colors), use_container_width=True)
        
        with tab2:
            st.markdown("### Education Level Impact")
            st.markdown("""
            > **Visualization 8:** Violin plots showing income distribution by education level 
            and loan status, revealing the combined effect of education and income.
            """)
            
            st.plotly_chart(create_education_analysis(df, colors), use_container_width=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_education_conversion_chart(df, colors), use_container_width=True)
            
            with col2:
                st.markdown("#### Education Statistics")
                edu_table = df.groupby('EducationLevel').agg({
                    'PersonalLoan': ['sum', 'count', 'mean'],
                    'Income': 'mean'
                }).reset_index()
                edu_table.columns = ['Education', 'Converted', 'Total', 'Conv Rate', 'Avg Income']
                edu_table['Conv Rate'] = (edu_table['Conv Rate'] * 100).round(1)
                edu_table['Avg Income'] = edu_table['Avg Income'].round(0)
                st.dataframe(edu_table, use_container_width=True)
                
                st.markdown("""
                <div class="insight-box">
                    <strong>💡 Insight:</strong> Advanced/Professional education shows 
                    ~4x higher conversion rate than Undergraduate level.
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### Mortgage Analysis")
            st.markdown("""
            > **Visualization 9:** 3D visualization of Income × Mortgage × Family Size, 
            showing how these factors combine to influence loan acceptance.
            """)
            
            st.plotly_chart(create_mortgage_analysis_3d(df, colors), use_container_width=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Mortgage presence impact
                mortgage_conv = df.groupby('HasMortgage')['PersonalLoan'].mean() * 100
                st.markdown("#### Mortgage Impact on Conversion")
                
                fig_mortgage = px.pie(
                    values=[mortgage_conv.get('Yes', 0), mortgage_conv.get('No', 0)],
                    names=['Has Mortgage', 'No Mortgage'],
                    title='Conversion Rate Comparison',
                    color_discrete_sequence=[colors['secondary_color'], colors['primary_color']]
                )
                st.plotly_chart(apply_theme_to_fig(fig_mortgage, colors), use_container_width=True)
            
            with col2:
                heatmap_fig = create_mortgage_heatmap(df, colors)
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                else:
                    st.info("Insufficient mortgage data for heatmap visualization.")
    
    # ============================================
    # PAGE: PRODUCT ANALYSIS
    # ============================================
    elif page == "🏦 Product Analysis":
        st.markdown('<h1 class="main-title">🏦 Product Analysis & Cross-Selling</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Analyze product ownership patterns and cross-selling opportunities</p>', 
                    unsafe_allow_html=True)
        
        st.markdown("### Product Ownership vs Loan Acceptance")
        st.markdown("""
        > **Visualization 6:** Compare conversion rates between customers who have 
        various bank products vs those who don't.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_product_analysis(df, colors), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_product_combination_chart(df, colors), use_container_width=True)
        
        st.markdown("---")
        
        # Cross-selling opportunities
        st.markdown("### 🎯 Cross-Selling Opportunities")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cd_holders = df[df['CDAccount'] == 1]
            cd_no_loan = cd_holders[cd_holders['PersonalLoan'] == 0]
            st.metric(
                "CD Holders without Loan",
                f"{len(cd_no_loan):,}",
                f"Potential: {(len(cd_no_loan) * 0.29):.0f} conversions"
            )
        
        with col2:
            sec_holders = df[df['SecuritiesAccount'] == 1]
            sec_no_cd = sec_holders[sec_holders['CDAccount'] == 0]
            st.metric(
                "Securities Holders without CD",
                f"{len(sec_no_cd):,}",
                "Cross-sell opportunity"
            )
        
        with col3:
            online_only = df[(df['Online'] == 1) & (df['CreditCard'] == 0)]
            st.metric(
                "Online Users without CC",
                f"{len(online_only):,}",
                "Credit card opportunity"
            )
        
        # Product overlap analysis
        st.markdown("### Product Overlap Analysis")
        
        # Create product overlap matrix
        products = ['CDAccount', 'SecuritiesAccount', 'CreditCard', 'Online']
        overlap_matrix = pd.DataFrame(index=products, columns=products)
        
        for p1 in products:
            for p2 in products:
                if p1 == p2:
                    overlap_matrix.loc[p1, p2] = df[p1].sum()
                else:
                    overlap_matrix.loc[p1, p2] = ((df[p1] == 1) & (df[p2] == 1)).sum()
        
        overlap_matrix = overlap_matrix.astype(float)
        
        fig_overlap = px.imshow(
            overlap_matrix,
            text_auto=True,
            color_continuous_scale='Blues',
            title='Product Ownership Overlap Matrix'
        )
        st.plotly_chart(apply_theme_to_fig(fig_overlap, colors), use_container_width=True)
    
    # ============================================
    # PAGE: DATA EXPLORER
    # ============================================
    elif page == "📋 Data Explorer":
        st.markdown('<h1 class="main-title">📋 Data Explorer</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Explore and filter the raw data</p>', 
                    unsafe_allow_html=True)
        
        # Filters
        st.markdown("### 🔍 Filters")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            income_range = st.slider(
                "Income Range ($K)",
                int(df['Income'].min()),
                int(df['Income'].max()),
                (int(df['Income'].min()), int(df['Income'].max()))
            )
        
        with col2:
            age_range = st.slider(
                "Age Range",
                int(df['Age'].min()),
                int(df['Age'].max()),
                (int(df['Age'].min()), int(df['Age'].max()))
            )
        
        with col3:
            education_filter = st.multiselect(
                "Education Level",
                options=df['EducationLevel'].unique(),
                default=df['EducationLevel'].unique()
            )
        
        with col4:
            loan_filter = st.multiselect(
                "Loan Status",
                options=['Accepted', 'Not Accepted'],
                default=['Accepted', 'Not Accepted']
            )
        
        # Apply filters
        filtered_df = df[
            (df['Income'] >= income_range[0]) &
            (df['Income'] <= income_range[1]) &
            (df['Age'] >= age_range[0]) &
            (df['Age'] <= age_range[1]) &
            (df['EducationLevel'].isin(education_filter)) &
            (df['LoanStatus'].isin(loan_filter))
        ]
        
        st.markdown(f"### Showing {len(filtered_df):,} of {len(df):,} records")
        
        # Display columns selector
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect(
            "Select columns to display",
            options=all_columns,
            default=['ID', 'Age', 'Income', 'Family', 'CCAvg', 'EducationLevel', 'Mortgage', 'LoanStatus']
        )
        
        # Display dataframe
        if selected_columns:
            st.dataframe(
                filtered_df[selected_columns].head(100),
                use_container_width=True,
                height=400
            )
        
        # Download button
        col1, col2 = st.columns([1, 3])
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data",
                data=csv,
                file_name="filtered_bank_data.csv",
                mime="text/csv"
            )
        
        # Summary statistics
        with st.expander("📊 Summary Statistics"):
            st.dataframe(filtered_df.describe(), use_container_width=True)


# ============================================
# RUN APPLICATION
# ============================================
if __name__ == "__main__":
    main()
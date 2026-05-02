import pandas as pd
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from scipy.interpolate import make_interp_spline
import math

# Global Palette
# Crimson Slate for Risk (Exclusion), Deep Teal for Budget (Inclusion)
SHA_PALETTE = ['#1f77b4', '#ff7f0e'] 

def exclusion_line_area(
        df: pd.DataFrame, 
        class_col: str, 
        save_path : None, 
        log : bool = True, 
        tolerance : float = 0.2 ) -> None:
    """
    Plot smooth exclusion risk (% overpredicted) per actual decile, split by a class (e.g., sex).
    Skips smoothing if there are too few points.
    """
    # Compute overprediction
    if log : 
        df['Prediction_Bucket'] = np.where(
            abs(df['log_error']) <= tolerance,
            'Accurate',
            np.where(df['log_error'] > 0, 'Higher', 'Lower')
        )

    else : 
        tolerance = (np.exp(tolerance) - 1) * 100
        df['log_error_pct'] = (np.exp(df['log_error']) - 1) * 100
        df['Prediction_Bucket'] = np.where(
            abs(df['log_error_pct']) <= tolerance,
            'Accurate',
            np.where(df['log_error_pct'] > 0, 'Higher', 'Lower')
        )
    
    df['is_overpredicted'] = df['Prediction_Bucket'] == 'Higher'

    # Seaborn style
    sns.set_style("whitegrid")
    sns.set_context("talk")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    classes = df[class_col].unique()
    palette = sns.color_palette("Set2", n_colors=len(classes))

    for i, cls in enumerate(classes):
        subset = df[df[class_col] == cls]
        audit_data = subset.groupby('Actual_Decile')['is_overpredicted'].mean() * 100

        x = audit_data.index.values
        y = audit_data.values

        # Only smooth if there are at least 4 unique points
        if len(x) >= 4 and len(np.unique(x)) == len(x):
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spline = make_interp_spline(x, y, k=3)
            y_smooth = spline(x_smooth)
        else:
            # Not enough points — just use original values
            x_smooth = x
            y_smooth = y

        # Shaded area
        ax.fill_between(x_smooth, y_smooth, color=palette[i], alpha=0.2)

        # Line
        ax.plot(x_smooth, y_smooth, color=palette[i], linewidth=2.5, label=str(cls))

        # Scatter original points
        ax.scatter(x, y, color=palette[i], s=50, zorder=5)

    # Axes and labels
    ax.set_xticks(range(1, 11))
    ax.set_xlabel('Actual Income Decile (1 = Poorest)', fontsize=12)
    ax.set_ylabel('% of People (Overpredicted)', fontsize=12)
    ax.set_title(f'Overprediction by Actual Decile ({class_col})', fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(title=class_col, frameon=False, fontsize=11)

    plt.tight_layout()
    plt.show()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')

    return fig, ax

def prediction_bar_chart(
        df: pd.DataFrame, 
        log: bool = True, 
        tolerance: float = 0.2, 
        save_path  = None, 
        save_csv = True 
        ) -> None:
    """
    Plot prediction accuracy per decile, allowing both decile tolerance and % tolerance.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns 'Actual_KES', 'Predicted_KES', 'Actual_Decile', 'Predicted_Decile', 'Decile_Diff'.
    tolerance : int
        Number of deciles difference allowed to still be considered accurate.
    pct_tolerance : float
        Relative tolerance of predicted vs actual value (e.g., 0.1 = ±10%).
    """

    if log : 
        df['Prediction_Bucket'] = np.where(
            abs(df['log_error']) <= tolerance,
            'Accurate',
            np.where(df['log_error'] > 0, 'Higher', 'Lower')
        )

    else : 
        tolerance = (np.exp(tolerance) - 1) * 100
        df['log_error_pct'] = (np.exp(df['log_error']) - 1) * 100
        df['Prediction_Bucket'] = np.where(
            abs(df['log_error_pct']) <= tolerance,
            'Accurate',
            np.where(df['log_error_pct'] > 0, 'Higher', 'Lower')
        )

    # --- 2. Aggregate ---
    bucket_counts = df.groupby(['Actual_Decile', 'Prediction_Bucket']).size().unstack(fill_value=0)
    bucket_percent = bucket_counts.div(bucket_counts.sum(axis=1), axis=0) * 100
    plot_data = bucket_percent[['Lower', 'Accurate', 'Higher']].copy()

    # --- 3. Colors & Labels ---
    colors = ['#5D8AA8', '#2A9D8F', '#D32F2F'] 
    labels = ['Under-Predicted (Lower)', 'Accurate', 'Over-Predicted (Higher)']

    fig, ax = plt.subplots(figsize=(15, 8), dpi=150)

    # Plot
    plot_data.plot(kind='bar', stacked=True, color=colors, ax=ax, width=0.85, edgecolor='white', linewidth=0.5)

    # --- 4. Aesthetics ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='grey', zorder=0)
    ax.set_title('Prediction Accuracy (±{}%)'.format(tolerance*100),
                 fontsize=16, fontweight='bold', pad=20, color='#333333')
    ax.set_xlabel('Actual Income Decile (1 = lowest)', fontsize=11, labelpad=10, color='#555555')
    ax.set_ylabel('Percentage of Population (%)', fontsize=11, color='#555555')
    ax.tick_params(axis='x', rotation=0, colors='#555555')
    ax.tick_params(axis='y', colors='#555555')

    # --- 5. Value Labels ---
    for c in ax.containers:
        labels_text = [f'{int(v.get_height())}%' if v.get_height() > 5 else '' for v in c]
        ax.bar_label(c, labels=labels_text, label_type='center', fontsize=10, color='white', fontweight='bold')

    # Legend at bottom
    ax.legend(labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=11)

    if save_path:
        fig.savefig(save_path / 'decile_chart.png', dpi=200, bbox_inches='tight')
    
    if save_csv:
        export_df = (
            plot_data
            .reset_index()  # move Actual_Decile from index to column
            .rename(columns={
                'Actual_Decile': 'Actual Decile',
                'Lower': 'Under-Predicted (%)',
                'Accurate': 'Accurate (%)',
                'Higher': 'Over-Predicted (%)'
            })
        )

        export_df.to_csv(save_path / "prediction_accuracy_by_decile.csv", index=False)

    return fig, ax 

def _apply_style(ax, title):
    """Internal helper to apply consistent styling."""
    ax.set_facecolor('#F8F9F9')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20, color='#1B2631')
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='grey')
    sns.despine(ax=ax, left=True)

def plot_prediction_scatter(
        df,
        x_col='Actual_KES',
        y_col='Predicted_KES',
        hue_col='urban_rural_classification',
        save_path=None,
    ):
    """Standardized Scatterplot for visualizing prediction variance."""

    # Palette matches sha_longread brand colors
    palette = {'Urban': '#00843d', 'Rural': '#DC143C'}

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=200)
    fig.patch.set_facecolor('#f8f1e4')
    ax.set_facecolor('#fdf6ec')

    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette=palette,
        alpha=0.45,
        s=22,
        linewidth=0,
        ax=ax,
    )

    # Perfect prediction line — ink, subtle dash (matches sha_longread band dividers)
    ax.axline((1, 1), slope=1, color='#1a1008', linewidth=1.5,
              linestyle=(0, (3, 5)), alpha=0.35, label='Perfect prediction')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ticks = [1_000, 10_000, 100_000]
    ax.set_xlim(500, 200_000)
    ax.set_ylim(500, 200_000)
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(mticker.FixedLocator(ticks))
        axis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        axis.set_minor_locator(mticker.NullLocator())

    # Spines
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#c8b89a')
    ax.tick_params(colors='#6b5c42', length=0)
    ax.grid(axis='both', color='#e0d4be', linewidth=0.8, alpha=0.9)

    # Typography — mirrors sha_longread font stack
    font_sans  = {'fontfamily': 'Helvetica Neue', 'color': '#1a1008'}
    font_muted = {'fontfamily': 'Helvetica Neue', 'color': '#6b5c42'}

    ax.set_title(
        'Predicted vs. Actual Consumption',
        fontsize=15, fontweight='bold', loc='left', pad=14, **font_sans,
    )
    ax.set_xlabel('Actual (KES)', fontsize=11, labelpad=10, **font_muted)
    ax.set_ylabel('Predicted (KES)', fontsize=11, labelpad=10, **font_muted)
    ax.tick_params(labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color('#6b5c42')

    legend = ax.legend(frameon=True, fontsize=10, framealpha=1,
                       edgecolor='#c8b89a', facecolor='#fdf6ec')
    for text in legend.get_texts():
        text.set_color('#1a1008')

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#f8f1e4')

    return fig, ax

def plot_density_comparison(
        df,
        actual_col='Actual',
        predicted_col='Predicted',
        save_path=None,
    ):
    """Overlapping KDE curves comparing actual vs predicted consumption distributions."""

    # sha_longread brand colors
    color_actual = '#00843d'  # Forest (Truth)
    color_pred   = '#DC143C'  # Crimson (Model)

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=200)
    fig.patch.set_facecolor('#f8f1e4')
    ax.set_facecolor('#fdf6ec')

    # Actual Distribution (Truth)
    sns.kdeplot(
        data=df[actual_col],
        fill=True,
        color=color_actual,
        alpha=0.15,
        linewidth=2.5,
        label='Actual',
        ax=ax,
    )

    # Predicted Distribution (Model)
    sns.kdeplot(
        data=df[predicted_col],
        fill=True,
        color=color_pred,
        alpha=0.15,
        linewidth=2.5,
        linestyle='--',
        label='Predicted',
        ax=ax,
    )

    sns.despine(left=True, bottom=False)
    ax.yaxis.set_visible(False)
    ax.spines['bottom'].set_color('#c8b89a')
    ax.tick_params(colors='#6b5c42', length=0, labelsize=10)
    ax.grid(axis='x', color='#e0d4be', linewidth=0.8, alpha=0.9)

    ax.set_title("Comparing Actual vs. Predicted Consumption Distributions",
                 fontsize=15, fontweight='bold', loc='left', pad=14,
                 color='#1a1008', fontfamily='Helvetica Neue')
    ax.set_xlabel("Log Consumption per Capita", fontsize=11, fontweight=600,
                  labelpad=10, color='#6b5c42', fontfamily='Helvetica Neue')

    legend_elements = [
        mpatches.Patch(facecolor=color_actual, edgecolor=color_actual, alpha=0.4, label='Actual'),
        mpatches.Patch(facecolor=color_pred,   edgecolor=color_pred,   alpha=0.4, label='Predicted'),
    ]
    legend = ax.legend(handles=legend_elements, loc='upper right', frameon=True,
                       fontsize=10, framealpha=1, edgecolor='#c8b89a', facecolor='#fdf6ec')
    for text in legend.get_texts():
        text.set_color('#1a1008')

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#f8f1e4')

    return fig, ax


def plot_residuals_grid(
        dataframes : list, 
        labels : list, 
        cols : int =2, 
        save_path = None 
        ):
    
    # 1. Calculate the number of rows needed
    num_cats = len(dataframes)
    rows = math.ceil(num_cats / cols)
    
    # 2. Create the figure and axes grid
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), dpi=200)
    axes = axes.flatten() # Flatten so we can loop through them easily
    
    # 3. Loop through categories and plot on specific axes
    for i, df in enumerate(dataframes):
        ax = axes[i]

        df['Residuals'] = df['Predicted'] - df['Actual']
        
        # Plotting on the specific subplot 'ax'
        sns.scatterplot(x='Predicted', y='Residuals', data=df, alpha=0.5, color='teal', ax=ax)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_title(f'{labels[i]}: Residual Plot', fontsize=12)
        ax.grid(True, alpha=0.3)
        
    # 4. Hide any empty subplots if the grid is larger than the number of categories
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        
    return fig, axes 

def plot_extreme_targeting_errors(poverty_summary, wealth_summary, save_path = None):
    """
    Creates a grouped bar chart comparing targeting errors for 
    the Bottom 20% and Upper 20% across Urban and Rural segments.
    """
    # 1. Prepare Data (Filtering for Urban/Rural rows only)
    segments = ['URBAN', 'RURAL']
    
    # Poverty Errors (Bottom 20%)
    pov_excl = poverty_summary[poverty_summary['Level'].isin(segments)]['Exclusion (%)'].values
    pov_incl = poverty_summary[poverty_summary['Level'].isin(segments)]['Inclusion (%)'].values
    
    # Wealth Errors (Upper 20%)
    wealth_excl = wealth_summary[wealth_summary['Level'].isin(segments)]['Exclusion (%)'].values
    wealth_incl = wealth_summary[wealth_summary['Level'].isin(segments)]['Inclusion (%)'].values

    # 2. Setup Plot
    x = np.arange(len(segments))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 8))

    # 3. Create Grouped Bars
    # Red tones for Poverty, Blue/Green tones for Wealth
    ax.bar(x - 1.5*width, pov_excl, width, label='Poverty Exclusion', color="#FBA35C")
    ax.bar(x - 0.5*width, pov_incl, width, label='Poverty Inclusion', color="#F1E7DF")
    ax.bar(x + 0.5*width, wealth_excl, width, label='Wealth Exclusion', color='#93C2E8')
    ax.bar(x + 1.5*width, wealth_incl, width, label='Wealth Inclusion', color="#CBE1F8")

    # 4. Branding & Labels
    ax.set_ylabel('Error Rate (%)', fontweight='bold')
    ax.set_title('Who is the model missing or misidentifying: POVERTY vs WEALTH EXTREMES', fontweight='bold', fontsize=16, pad=25)
    ax.set_xticks(x)
    ax.set_xticklabels(segments, fontweight='bold', fontsize=12)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Add percentage labels
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width()/2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')

    return fig, ax



def plot_poverty_line_scatter(
        df,
        quantile=0.40,
        x_col='Actual_KES',
        y_col='Predicted_KES',
        title=None,
        save_path=None,
    ):
    """
    Scatter plot for a single sector (urban or rural). Shows only
    households in the bottom quantile, with axes as percentiles.
    Poverty line is a horizontal line at the quantile cutoff.
    Points above = exclusion errors.
    """

    pov_pctile = quantile * 100

    # Percentile ranks within this sector
    actual_pct = df[x_col].rank(pct=True) * 100
    predicted_pct = df[y_col].rank(pct=True) * 100

    # Filter to bottom quantile
    mask = actual_pct <= pov_pctile
    x_vals = actual_pct[mask]
    y_vals = predicted_pct[mask]

    # Color based on whether dot is above or below the line visually
    above_line = y_vals > pov_pctile
    below_line = ~above_line

    # Exclusion rate based on KES (matches training.py logic)
    poverty_line_kes = df[x_col].quantile(quantile)
    excluded_kes = df.loc[mask, y_col] > poverty_line_kes

    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

    ax.scatter(x_vals[below_line], y_vals[below_line],
               color='#2C3E50', alpha=0.35, s=15, label='Correctly Targeted', zorder=2)
    ax.scatter(x_vals[above_line], y_vals[above_line],
               color='#E74C3C', alpha=0.55, s=20, label='Exclusion Error', zorder=3)

    # Horizontal poverty line
    ax.axhline(y=pov_pctile, color='black', linewidth=1.5, linestyle='-', zorder=4,
               label=f'Poverty Line ({int(pov_pctile)}th percentile)')

    # Stats
    n_total = mask.sum()
    n_excluded = excluded_kes.sum()
    excl_rate = n_excluded / n_total * 100 if n_total > 0 else 0

    stats_text = f'Exclusion Error: {n_excluded}/{n_total} ({excl_rate:.1f}%)'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='grey', alpha=0.8))

    ax.set_xlim(0, pov_pctile + 2)
    ax.set_ylim(0, 100)

    if title is None:
        pct_label = int(quantile * 100)
        title = f'Exclusion Errors — Bottom {pct_label}%'
    ax.set_title(title, fontsize=14, fontweight='bold', loc='left', pad=12)
    ax.set_xlabel('Actual Consumption (percentile)', fontsize=11)
    ax.set_ylabel('Predicted Consumption (percentile)', fontsize=11)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    sns.despine()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')

    return fig, ax


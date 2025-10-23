"""
Standard visualization functions for Myanmar Economic Monitor
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import geopandas as gpd
from wbpyplot import wb_plot
from typing import Optional, Union, Literal


def pick_value_column(df: pd.DataFrame):
    """Helper function to pick a value column from DataFrame"""
    _DEF_EXCLUDE = {"year","month","date","adm0","adm1","adm2","adm3","adm4","name","code","id"}
    prefer = [
        "ntl_sum", "ntl_mean"
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for key in prefer:
        if key in lower_map:
            return lower_map[key]
    num_cols = [c for c in df.select_dtypes("number").columns if c.lower() not in _DEF_EXCLUDE]
    return num_cols[0] if num_cols else None


def plot_line_chart(df: pd.DataFrame, 
                   x_col: str, 
                   value_col: Optional[str] = None,
                   title: str = "Line Chart",
                   xlabel: str = "X",
                   ylabel: str = "Y",
                   source_text: str = "Source: VIIRS Nighttime Lights — Collection 2",
                   date_col: Optional[str] = None,
                   color: str = "#1f77b4",
                   marker: str = "o",
                   linewidth: float = 2,
                   earthquake_marker: Optional[str] = None) -> None:
    """
    Create a standard line chart using wb_plot decorator
    
    Parameters:
    - df: DataFrame containing the data
    - x_col: Column name for x-axis (e.g., 'year')
    - value_col: Column name for values (if None, will auto-detect)
    - title: Chart title
    - xlabel: X-axis label
    - ylabel: Y-axis label
    - source_text: Source note for bottom of chart
    - date_col: Optional date column to derive year from
    - color: Line color
    - marker: Line marker style
    - linewidth: Line width
    - earthquake_marker: Optional date string for earthquake marker (e.g., '2025-03-25')
    """
    
    # Prepare data
    plot_df = df.copy()
    
    # Handle date column if provided
    if date_col and date_col in plot_df.columns:
        plot_df['year'] = pd.to_datetime(plot_df[date_col]).dt.year
        year_col = 'year'
    elif 'date' in plot_df.columns:
        plot_df['year'] = pd.to_datetime(plot_df['date']).dt.year
        year_col = 'year'
    else:
        year_col = x_col
    
    # Auto-detect value column if not provided
    if value_col is None:
        value_col = pick_value_column(plot_df)
    
    assert year_col is not None and value_col is not None, f"Could not determine columns. x={year_col}, value={value_col}"
    
    # Aggregate data
    agg_data = (plot_df[[year_col, value_col]]
                .dropna()
                .groupby(year_col, as_index=False)[value_col].mean()
                .sort_values(year_col))
    
    @wb_plot(title=title, subtitle=None, note=source_text)
    def plot_func(*args):
        # Handle axes input from wb_plot
        axes = args[0] if len(args) == 1 else (args[1] if len(args) >= 2 else None)
        if axes is None:
            raise ValueError("wb_plot did not provide axes")
        
        # Normalize to a single Axes
        if hasattr(axes, 'plot') and hasattr(axes, 'bar'):
            ax = axes
        else:
            try:
                arr = np.asarray(axes).ravel()
                ax = arr[0]
            except Exception:
                ax = axes[0] if isinstance(axes, (list, tuple)) else axes
        
        # Create the plot
        ax.plot(agg_data[year_col], agg_data[value_col], 
                marker=marker, linewidth=linewidth, color=color, label=value_col)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Show every tick if reasonable number
        if len(agg_data) <= 20:
            years = agg_data[year_col].astype(int).tolist()
            ax.set_xticks(years)
            ax.set_xticklabels([str(y) for y in years], rotation=0)
        
        # Add earthquake marker if specified
        if earthquake_marker:
            try:
                if pd.api.types.is_datetime64_any_dtype(agg_data[year_col]):
                    ax.axvline(pd.Timestamp(earthquake_marker), color='red', linestyle='--', 
                              linewidth=2, alpha=0.8, zorder=3, label='Earthquake')
                else:
                    # Try to parse as year if it's not datetime
                    marker_year = pd.to_datetime(earthquake_marker).year
                    ax.axvline(marker_year, color='red', linestyle='--', 
                              linewidth=2, alpha=0.8, zorder=3, label='Earthquake')
            except Exception:
                # If parsing fails, skip the marker
                pass
        
        ax.legend()
    
    plot_func()


def plot_bar_chart(df: pd.DataFrame,
                   x_col: Optional[str] = None,
                   value_col: Optional[str] = None,
                   title: str = "Bar Chart",
                   xlabel: str = "X", 
                   ylabel: str = "Value",
                   source_text: str = "Source: VIIRS Nighttime Lights — Collection 2",
                   date_col: Optional[str] = None,
                   color: Union[str, dict] = "#4e79a7",
                   figsize: tuple = (12, 6),
                   earthquake_marker: Optional[str] = None,
                   bar_width: Union[int, float] = None,
                   is_percentage: bool = False,
                   pos_color: str = "#2ca02c",
                   neg_color: str = "#d62728",
                   zero_line: bool = False) -> None:
    """
    Create a standard bar chart using World Bank styling
    
    Parameters:
    - df: DataFrame containing the data
    - x_col: Column name for x-axis (if None, will try to auto-detect)
    - value_col: Column name for values (if None, will auto-detect)
    - title: Chart title
    - xlabel: X-axis label
    - ylabel: Y-axis label
    - source_text: Source note for bottom of chart
    - date_col: Date column name (if None, will try to construct from year/month for time series)
    - color: Bar color (single color) or dict for custom coloring
    - figsize: Figure size tuple
    - earthquake_marker: Optional date string for earthquake marker (e.g., '2025-03-25')
    - bar_width: Width of bars (auto-calculated if None)
    - is_percentage: If True, uses different colors for positive/negative values
    - pos_color: Color for positive percentage values
    - neg_color: Color for negative percentage values
    - zero_line: If True, adds a horizontal line at y=0
    """
    
    # Prepare data
    plot_df = df.copy()
    
    # Determine x column
    if x_col is None:
        # Try to auto-detect x column
        if date_col and date_col in plot_df.columns:
            x_column = date_col
        elif 'date' in plot_df.columns:
            x_column = 'date'
        elif 'year' in plot_df.columns:
            x_column = 'year'
        else:
            # Look for date-like columns
            year_cand = next((c for c in ["year","Year","YEAR"] if c in plot_df.columns), None)
            month_cand = next((c for c in ["month","Month","MONTH","mon","Mon","MON"] if c in plot_df.columns), None)
            if year_cand and month_cand:
                plot_df['date'] = pd.to_datetime(dict(year=plot_df[year_cand], month=plot_df[month_cand], day=1))
                x_column = 'date'
            elif year_cand:
                x_column = year_cand
            else:
                raise ValueError('Could not determine x column for data')
    else:
        x_column = x_col
        
    # Handle date column conversion if needed
    if date_col and date_col in plot_df.columns and x_column != date_col:
        plot_df[x_column] = pd.to_datetime(plot_df[date_col])
    elif x_column in ['date', 'Date'] and not pd.api.types.is_datetime64_any_dtype(plot_df[x_column]):
        plot_df[x_column] = pd.to_datetime(plot_df[x_column])
    
    # Auto-detect value column
    if value_col is None:
        value_col = pick_value_column(plot_df)
    assert value_col is not None, 'Could not determine value column'
    
    # Prepare final data
    chart_data = (plot_df[[x_column, value_col]]
                  .dropna()
                  .copy())
    chart_data = chart_data.sort_values(x_column)
    
    # Determine if this is a time series (for formatting)
    is_time_series = pd.api.types.is_datetime64_any_dtype(chart_data[x_column])
    
    # Create the plot manually with World Bank styling
    import matplotlib.pyplot as plt
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Apply World Bank styling manually
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set grid
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Determine bar colors
    if is_percentage:
        colors = [pos_color if v >= 0 else neg_color for v in chart_data[value_col]]
        bars = ax.bar(chart_data[x_column], chart_data[value_col], color=colors, width=bar_width)
    else:
        bars = ax.bar(chart_data[x_column], chart_data[value_col], color=color, width=bar_width)
    
    # Add zero line for percentage charts
    if zero_line or is_percentage:
        ax.axhline(0, color='#666666', linewidth=1, alpha=0.8)
    
    # Add earthquake marker if specified
    if earthquake_marker and is_time_series:
        ax.axvline(pd.Timestamp(earthquake_marker), color='red', linestyle='--', 
                  linewidth=2, alpha=0.8, zorder=3, label='Earthquake')
    
    # Format x-axis based on data type
    if is_time_series:
        # Time series formatting - improved for different time spans
        time_span_years = (chart_data[x_column].max() - chart_data[x_column].min()).days / 365.25
        
        if time_span_years <= 2:
            # For short time spans (<=2 years), show months
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Every 3 months
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
            ax.tick_params(axis='x', which='minor', length=2, labelbottom=False)
            ax.tick_params(axis='x', which='major', rotation=45, labelsize=9)
        elif time_span_years <= 5:
            # For medium time spans (2-5 years), show every 6 months  
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
            ax.tick_params(axis='x', which='minor', length=2, labelbottom=False)
            ax.tick_params(axis='x', which='major', rotation=45, labelsize=9)
        else:
            # For long time spans (>5 years), show years with some months
            ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 7]))  # Jan and July
            ax.tick_params(axis='x', which='minor', length=2, labelbottom=False)
            ax.tick_params(axis='x', which='major', rotation=0, labelsize=10)
        
        # Pad x-limits for time series
        if bar_width is None:
            time_padding = pd.Timedelta(days=15)
        else:
            time_padding = pd.Timedelta(days=bar_width//2)
        xmin = chart_data[x_column].min() - time_padding
        xmax = chart_data[x_column].max() + time_padding
        ax.set_xlim(xmin, xmax)
    else:
        # Categorical/discrete x-axis
        if len(chart_data[x_column].unique()) <= 20:
            ax.set_xticks(chart_data[x_column])
            if x_column in ['year', 'Year']:
                ax.set_xticklabels([str(int(x)) for x in chart_data[x_column]], rotation=0)
            else:
                ax.set_xticklabels([str(x) for x in chart_data[x_column]], rotation=45)
    
    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Set y-axis limits to show negative values BEFORE adding labels
    if is_percentage or chart_data[value_col].min() < 0:
        y_min = chart_data[value_col].min()
        y_max = chart_data[value_col].max()
        y_range = y_max - y_min
        # Add 10% padding to both ends
        padding = y_range * 0.1
        ax.set_ylim(y_min - padding, y_max + padding)
    
    # Add value labels on bars if requested for percentage charts
    if is_percentage:
        for bar, value in zip(bars, chart_data[value_col]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., 
                   height + (0.5 if height >= 0 else -0.5),
                   f'{value:.1f}%',
                   ha='center', va='bottom' if height >= 0 else 'top', 
                   fontsize=8)
    
    # Add title with World Bank styling (left-aligned with bold first word)
    title_parts = title.split(' ', 1)
    if len(title_parts) == 2:
        bold_part, normal_part = title_parts
        formatted_title = r'$\mathbf{' + bold_part + '}$ ' + normal_part
    else:
        formatted_title = r'$\mathbf{' + title + '}$'
    
    ax.set_title(formatted_title, fontsize='x-large', fontweight='normal', 
                ha='left', x=0, pad=20)
    
    # Add source note
    fig.text(0.1, 0.02, source_text, fontsize=9, color='gray', 
             ha='left', va='bottom')
    
    plt.tight_layout()
    plt.show()




def plot_subplots_bar_charts(df: pd.DataFrame,
                            group_col: str,
                            x_col: str,
                            value_col: Optional[str] = None,
                            title: str = "Bar Chart Subplots",
                            xlabel: str = "X",
                            ylabel: str = "Y",
                            source_text: str = "Source: VIIRS Nighttime Lights — Collection 2",
                            ncols: int = 4,
                            figsize_per_subplot: tuple = (4.5, 3.5),
                            color: str = "#4e79a7",
                            bar_width: float = 0.7,
                            share_axes: bool = True,
                            earthquake_marker: str = None) -> None:
    """
    Create multiple bar chart subplots for different groups
    
    Parameters:
    - df: DataFrame containing the data
    - group_col: Column to group by (e.g., 'ADM1_EN')
    - x_col: Column for x-axis (e.g., 'year')
    - value_col: Column for values (if None, will auto-detect)
    - title: Overall title for the figure
    - xlabel: X-axis label for subplots
    - ylabel: Y-axis label for subplots
    - source_text: Source note (not used in this version but kept for consistency)
    - ncols: Number of columns in subplot grid
    - figsize_per_subplot: Size multiplier for each subplot
    - color: Bar color
    - bar_width: Width of bars
    - share_axes: Whether to share x and y axes across subplots
    - earthquake_marker: Optional date string for earthquake marker (e.g., '2025-03-25')
    """
    
    # Auto-detect value column
    if value_col is None:
        value_col = pick_value_column(df)
    assert value_col is not None, f"Could not determine value column"
    
    # Ensure required columns exist
    assert group_col in df.columns, f"{group_col} column not found"
    assert x_col in df.columns, f"{x_col} column not found"
    assert value_col in df.columns, f"{value_col} column not found"
    
    # Get unique groups - preserve categorical order if present
    if pd.api.types.is_categorical_dtype(df[group_col]):
        # Use categorical order if the column is categorical
        group_list = [cat for cat in df[group_col].cat.categories if cat in df[group_col].values]
    else:
        # Fall back to sorted order for non-categorical columns
        group_list = sorted(df[group_col].dropna().unique())
    n_groups = len(group_list)
    
    # Calculate grid dimensions
    nrows = int(np.ceil(n_groups / ncols))
    
    # Create subplots with World Bank styling
    fig, axs = plt.subplots(nrows, ncols, 
                           figsize=(figsize_per_subplot[0]*ncols, figsize_per_subplot[1]*nrows),
                           sharex=share_axes, sharey=share_axes)
    
    # Handle case where only one subplot
    if n_groups == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
    
    # Create individual plots
    for i, group in enumerate(group_list):
        sub_data = df[df[group_col] == group].copy()
        
        # Apply World Bank styling to each subplot
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].grid(True, alpha=0.3, linewidth=0.5)
        axs[i].set_axisbelow(True)
        
        # Handle different x-axis data types
        if x_col in ['year', 'Year']:
            x = sub_data[x_col].astype(int)
            y = sub_data[value_col]
            
            axs[i].bar(x, y, color=color, width=bar_width)
            axs[i].set_xticks(x)
            axs[i].set_xticklabels([str(int(xx)) for xx in x], rotation=0)
            
        elif pd.api.types.is_datetime64_any_dtype(sub_data[x_col]):
            # Handle datetime x-axis
            x = sub_data[x_col]
            y = sub_data[value_col]
            
            axs[i].bar(x, y, color=color, width=bar_width)
            
            # Format date axis intelligently
            n_dates = len(x.unique())
            if n_dates <= 12:  # Show all dates if 12 or fewer
                axs[i].set_xticks(x)
                # Check if we're dealing with months - use month names for better readability
                if all(d.day == 1 for d in x):  # Monthly data (day=1)
                    axs[i].set_xticklabels([d.strftime('%b %Y') if n_dates > 6 else d.strftime('%b') for d in x], rotation=45)
                else:
                    axs[i].set_xticklabels([d.strftime('%Y-%m-%d') for d in x], rotation=45)
            else:  # Show subset of dates
                # Show every nth date to avoid overcrowding
                step = max(1, n_dates // 8)  # Show about 8 labels max
                tick_positions = x[::step]
                axs[i].set_xticks(tick_positions)
                if all(d.day == 1 for d in tick_positions):  # Monthly data
                    axs[i].set_xticklabels([d.strftime('%b %Y') for d in tick_positions], rotation=45)
                else:
                    axs[i].set_xticklabels([d.strftime('%Y-%m') for d in tick_positions], rotation=45)
        else:
            # Handle other data types
            x = sub_data[x_col]
            y = sub_data[value_col]
            
            axs[i].bar(x, y, color=color, width=bar_width)
            axs[i].set_xticks(x)
            axs[i].set_xticklabels([str(xx) for xx in x], rotation=45)
        
        # World Bank styling for subplot titles and labels with fancy formatting
        title_words = str(group).split()
        if len(title_words) > 0:
            # Create formatted title with first word bold
            if len(title_words) > 1:
                formatted_title = f"$\\mathbf{{{title_words[0]}}}$ {' '.join(title_words[1:])}"
            else:
                formatted_title = f"$\\mathbf{{{title_words[0]}}}$"
            
            axs[i].text(0.02, 0.95, formatted_title, transform=axs[i].transAxes, 
                       fontsize=12, verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
        
        axs[i].set_xlabel(xlabel, fontsize=9)
        axs[i].set_ylabel(ylabel, fontsize=9)
        
        # Add earthquake marker if specified and x-axis is datetime
        if earthquake_marker and pd.api.types.is_datetime64_any_dtype(sub_data[x_col]):
            axs[i].axvline(pd.Timestamp(earthquake_marker), color='red', linestyle='--', 
                          linewidth=1.5, alpha=0.8, zorder=3)
    
    # Hide unused axes with World Bank styling
    for j in range(n_groups, len(axs)):
        axs[j].axis('off')

    # Set overall title with World Bank formatting (left-aligned, first word bold)
    title_words = title.split()
    if len(title_words) > 0:
        # Create formatted main title with first word bold using LaTeX math mode
        if len(title_words) > 1:
            formatted_main_title = f"$\\mathbf{{{title_words[0]}}}$ {' '.join(title_words[1:])}"
        else:
            formatted_main_title = f"$\\mathbf{{{title_words[0]}}}$"
        
        fig.text(0.02, 0.98, formatted_main_title, fontsize=18, fontweight='normal',
                verticalalignment='top', horizontalalignment='left', transform=fig.transFigure)

    # Add source note
    fig.text(0.02, 0.01, source_text, fontsize=9, color='gray', 
             ha='left', va='bottom')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_comparison_bar_chart(df: pd.DataFrame,
                             group_col: str,
                             x_col: str,
                             value_col: Optional[str] = None,
                             compare_values: list = [2024, 2025],
                             title: str = "Comparison Bar Chart",
                             xlabel: str = "Groups",
                             ylabel: str = "Value",
                             source_text: str = "Source: VIIRS Nighttime Lights — Collection 2",
                             figsize: tuple = (10, 6),
                             colors: list = None,
                             bar_width: float = 0.35,
                             add_trend_lines: bool = True,
                             earthquake_marker: Optional[str] = None) -> None:
    """
    Create a comparison bar chart with optional trend lines
    
    Parameters:
    - df: DataFrame containing the data
    - group_col: Column to group by (e.g., 'ADM1_EN')
    - x_col: Column containing comparison values (e.g., 'year')
    - value_col: Column for values (if None, will auto-detect)
    - compare_values: List of values to compare (e.g., [2024, 2025])
    - title: Chart title
    - xlabel: X-axis label
    - ylabel: Y-axis label
    - source_text: Source note (not used but kept for consistency)
    - figsize: Figure size
    - colors: List of colors for each comparison value
    - bar_width: Width of bars
    - add_trend_lines: Whether to add trend lines connecting the bars
    - earthquake_marker: Optional date string for earthquake marker (e.g., '2025-03-25')
    """
    
    # Auto-detect value column
    if value_col is None:
        value_col = pick_value_column(df)
    assert value_col is not None, f"Could not determine value column"
    
    # Default colors
    if colors is None:
        colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']
    
    # Filter data for comparison values
    comparison_df = df[df[x_col].isin(compare_values)].copy()
    
    # Get unique groups
    group_list = sorted(comparison_df[group_col].dropna().unique())
    n_groups = len(group_list)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    
    # Calculate positions
    x_pos = np.arange(n_groups)
    
    # Create bars for each comparison value
    for i, compare_val in enumerate(compare_values):
        values = []
        for group in group_list:
            group_data = comparison_df[(comparison_df[group_col] == group) & 
                                     (comparison_df[x_col] == compare_val)]
            val = group_data[value_col].mean() if not group_data.empty else 0
            values.append(val)
        
        offset = (i - len(compare_values)/2 + 0.5) * bar_width
        ax.bar(x_pos + offset, values, bar_width, 
               label=str(compare_val), color=colors[i % len(colors)])
        
        # Add trend line if requested
        if add_trend_lines:
            ax.plot(x_pos + offset, values, color=colors[i % len(colors)], 
                   marker='o', linestyle='--' if i == 0 else '-', alpha=0.7)
    
    # Add earthquake marker if specified (for time series data)
    if earthquake_marker:
        try:
            # Try to parse as year for yearly data
            marker_year = pd.to_datetime(earthquake_marker).year
            if marker_year in compare_values:
                # Add a vertical line at the earthquake marker position
                ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8, zorder=3)
                ax.text(0.02, 0.98, f'Earthquake: {earthquake_marker}', 
                       transform=ax.transAxes, fontsize=9, color='red',
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except Exception:
            # If parsing fails, skip the marker
            pass
    
    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(group_list, rotation=45, ha='right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def plot_comparative_lines(df: pd.DataFrame,
                          x_col: str,
                          value_col: Optional[str] = None,
                          group_col: str = None,
                          title: str = "Comparative Line Plot",
                          xlabel: str = "X",
                          ylabel: str = "Y",
                          source_text: str = "Source: VIIRS Nighttime Lights — Collection 2",
                          date_col: Optional[str] = None,
                          figsize: tuple = (12, 8),
                          colors: list = None,
                          markers: list = None,
                          linewidth: float = 2,
                          marker_size: float = 6,
                          alpha: float = 0.8,
                          legend_location: str = 'best',
                          earthquake_marker: Optional[str] = None) -> None:
    """
    Create comparative line plots for different groups/categories
    
    Parameters:
    - df: DataFrame containing the data
    - x_col: Column for x-axis (e.g., 'year', 'date')
    - value_col: Column for values (if None, will auto-detect)
    - group_col: Column to group by for separate lines (e.g., 'ADM1_EN', 'category')
    - title: Chart title
    - xlabel: X-axis label
    - ylabel: Y-axis label
    - source_text: Source note for bottom of chart
    - date_col: Optional date column to derive year from
    - figsize: Figure size
    - colors: List of colors for different lines
    - markers: List of markers for different lines
    - linewidth: Width of lines
    - marker_size: Size of markers
    - alpha: Transparency of lines
    - legend_location: Location of legend
    - earthquake_marker: Optional date string for earthquake marker (e.g., '2025-03-25')
    """
    
    # Auto-detect value column
    if value_col is None:
        value_col = pick_value_column(df)
    assert value_col is not None, f"Could not determine value column"
    
    # Prepare data
    plot_df = df.copy()
    
    # Convert group column to string for consistent handling (especially for years)
    plot_df[group_col] = plot_df[group_col].astype(str)
    
    # Handle date column if provided - but avoid conflicts with group_col
    x_column = x_col
    
    # Only convert date to year if we're not already using 'year' as group_col
    # and if x_col is asking for 'year' but we have dates
    if (date_col and date_col in plot_df.columns and 
        x_col == 'year' and group_col != 'year'):
        plot_df['derived_year'] = pd.to_datetime(plot_df[date_col]).dt.year
        x_column = 'derived_year'
    elif ('date' in plot_df.columns and x_col == 'year' and 
          group_col != 'year' and 'year' not in plot_df.columns):
        plot_df['derived_year'] = pd.to_datetime(plot_df['date']).dt.year
        x_column = 'derived_year'
    
    # Ensure required columns exist
    assert x_column in plot_df.columns, f"{x_column} column not found"
    assert group_col in plot_df.columns, f"{group_col} column not found"
    assert value_col in plot_df.columns, f"{value_col} column not found"
    
    # Aggregate data by group and x
    agg_data = (plot_df[[x_column, group_col, value_col]]
                .dropna()
                .groupby([x_column, group_col], as_index=False)[value_col].mean()
                .sort_values([group_col, x_column]))
    
    # Get unique groups
    groups = sorted(agg_data[group_col].unique())
    n_groups = len(groups)
    
    # Default colors and markers
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    if markers is None:
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Create plot with World Bank styling
    fig, ax = plt.subplots(figsize=figsize)
    
    # Apply World Bank styling manually
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set grid
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Plot each group as a separate line
    for i, group in enumerate(groups):
        group_data = agg_data[agg_data[group_col] == group]
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        ax.plot(group_data[x_column], group_data[value_col], 
               color=color, marker=marker, linewidth=linewidth, 
               markersize=marker_size, alpha=alpha, label=str(group))
    
    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Add title with World Bank styling (left-aligned with bold first word)
    title_parts = title.split(' ', 1)
    if len(title_parts) == 2:
        bold_part, normal_part = title_parts
        formatted_title = r'$\mathbf{' + bold_part + '}$ ' + normal_part
    else:
        formatted_title = r'$\mathbf{' + title + '}$'
    
    ax.set_title(formatted_title, fontsize='x-large', fontweight='normal', 
                ha='left', x=0, pad=20)
    
    # Add earthquake marker if specified
    if earthquake_marker:
        try:
            if pd.api.types.is_datetime64_any_dtype(agg_data[x_column]):
                ax.axvline(pd.Timestamp(earthquake_marker), color='red', linestyle='--', 
                          linewidth=2, alpha=0.8, zorder=3, label='Earthquake')
            else:
                # Try to parse as year if it's not datetime
                marker_year = pd.to_datetime(earthquake_marker).year
                ax.axvline(marker_year, color='red', linestyle='--', 
                          linewidth=2, alpha=0.8, zorder=3, label='Earthquake')
        except Exception:
            # If parsing fails, skip the marker
            pass
    
    # Legend with World Bank styling
    ax.legend(loc=legend_location, frameon=False, fontsize=10)
    
    # Format x-axis if it's years
    if x_column in ['year', 'Year'] and len(agg_data[x_column].unique()) <= 20:
        years = sorted(agg_data[x_column].unique())
        ax.set_xticks(years)
        ax.set_xticklabels([str(int(y)) for y in years], rotation=0)
    
    # Add source note
    fig.text(0.1, 0.02, source_text, fontsize=9, color='gray', 
             ha='left', va='bottom')
    
    plt.tight_layout()
    plt.show()


def plot_single_map(gdf: gpd.GeoDataFrame,
                   value_col: str,
                   title: str = "Myanmar Map",
                   cmap: str = 'RdYlGn',
                   figsize: tuple = (10, 8),
                   legend: bool = True,
                   source_text: str = "Source: VIIRS Nighttime Lights — Collection 2",
                   vmin: float = None,
                   vmax: float = None,
                   legend_kwargs: dict = None) -> None:
    """
    Create a single choropleth map
    
    Parameters:
    - gdf: GeoDataFrame with geometry and data
    - value_col: Column name for values to map
    - title: Map title
    - cmap: Colormap name
    - figsize: Figure size
    - legend: Whether to show legend
    - source_text: Source note for bottom of chart
    - vmin, vmax: Value range for color scaling
    - legend_kwargs: Additional kwargs for legend
    """
    
    if legend_kwargs is None:
        legend_kwargs = {}
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    
    # Plot the map
    gdf.plot(column=value_col, 
             cmap=cmap, 
             legend=legend,
             ax=ax,
             vmin=vmin,
             vmax=vmax,
             legend_kwds=legend_kwargs)
    
    # Clean up the map styling
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add title with consistent font
    _ff = plt.rcParams.get('font.family', ['sans-serif'])
    font_family = _ff[0] if isinstance(_ff, (list, tuple)) else _ff
    ax.set_title(title, fontfamily=font_family, fontsize=14, fontweight='bold')
    
    # Source note
    fig.text(0.5, 0.01, source_text, ha='center', va='bottom', fontsize=9, color='gray')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()


def plot_multiple_maps(gdf: gpd.GeoDataFrame,
                      value_cols: list,
                      titles: list = None,
                      cmap: str = 'RdYlGn',
                      figsize: tuple = None,
                      ncols: int = 2,
                      legend: bool = True,
                      source_text: str = "Source: VIIRS Nighttime Lights — Collection 2",
                      suptitle: str = "Myanmar Maps",
                      vmin: float = None,
                      vmax: float = None,
                      legend_kwargs: dict = None) -> None:
    """
    Create multiple choropleth maps in subplots
    
    Parameters:
    - gdf: GeoDataFrame with geometry and data
    - value_cols: List of column names for values to map
    - titles: List of titles for each subplot (if None, uses column names)
    - cmap: Colormap name
    - figsize: Figure size (auto-calculated if None)
    - ncols: Number of columns in subplot grid
    - legend: Whether to show legends
    - source_text: Source note for bottom of chart
    - suptitle: Overall figure title
    - vmin, vmax: Value range for color scaling (applied to all maps)
    - legend_kwargs: Additional kwargs for legends
    """
    
    if legend_kwargs is None:
        legend_kwargs = {}
    
    n_maps = len(value_cols)
    nrows = int(np.ceil(n_maps / ncols))
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (6 * ncols, 5 * nrows)
    
    # Default titles if not provided
    if titles is None:
        titles = [col.replace('_', ' ').title() for col in value_cols]
    
    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=120)
    
    # Handle single row case
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Plot each map
    for i, (value_col, title) in enumerate(zip(value_cols, titles)):
        ax = axes[i]
        
        # Plot the map
        gdf.plot(column=value_col,
                 cmap=cmap,
                 legend=legend,
                 ax=ax,
                 vmin=vmin,
                 vmax=vmax,
                 legend_kwds=legend_kwargs)
        
        # Clean up the map styling
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Add title
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Hide unused subplots
    for j in range(n_maps, len(axes)):
        axes[j].axis('off')
    
    # Add overall title
    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.98)
    
    # Source note
    fig.text(0.5, 0.01, source_text, ha='center', va='bottom', fontsize=9, color='gray')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


def plot_maps_by_category(gdf: gpd.GeoDataFrame,
                         category_col: str,
                         value_col: str,
                         titles: list = None,
                         cmap: str = 'RdYlGn',
                         figsize: tuple = None,
                         ncols: int = 3,
                         legend: bool = True,
                         source_text: str = "Source: VIIRS Nighttime Lights — Collection 2",
                         suptitle: str = "Myanmar Maps by Category",
                         vmin: float = None,
                         vmax: float = None,
                         legend_kwargs: dict = None,
                         boundary_gdf: gpd.GeoDataFrame = None,
                         boundary_kwargs: dict = None) -> None:
    """
    Create multiple choropleth maps split by unique categories in a specified column
    
    Parameters:
    - gdf: GeoDataFrame with geometry and data
    - category_col: Column name to split categories by (e.g., 'category', 'period')
    - value_col: Column name for values to map
    - titles: List of titles for each subplot (if None, uses category values)
    - cmap: Colormap name
    - figsize: Figure size (auto-calculated if None)
    - ncols: Number of columns in subplot grid
    - legend: Whether to show legends
    - source_text: Source note for bottom of chart
    - suptitle: Overall figure title
    - vmin, vmax: Value range for color scaling (applied to all maps)
    - legend_kwargs: Additional kwargs for legends
    - boundary_gdf: Optional GeoDataFrame to overlay boundaries on all maps
    - boundary_kwargs: Optional styling kwargs for boundary overlay (e.g., {'color': 'black', 'linewidth': 1})
    """
    
    if legend_kwargs is None:
        legend_kwargs = {}
        
    if boundary_kwargs is None:
        boundary_kwargs = {'color': 'black', 'linewidth': 0.8, 'alpha': 0.7}
    
    # Get unique categories (excluding None/NaN)
    categories = [cat for cat in gdf[category_col].dropna().unique() if cat is not None]
    n_maps = len(categories)
    
    if n_maps == 0:
        print(f"No valid categories found in column '{category_col}'")
        return
    
    nrows = int(np.ceil(n_maps / ncols))
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (6 * ncols, 5 * nrows)
    
    # Default titles if not provided
    if titles is None:
        titles = [str(cat) for cat in categories]
    
    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=120)
    
    # Handle single subplot case
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()
    
    # Calculate global vmin/vmax if not provided
    if vmin is None or vmax is None:
        all_values = gdf[value_col].dropna()
        if vmin is None:
            vmin = all_values.min()
        if vmax is None:
            vmax = all_values.max()
    
    # Create a diverging normalization centered at zero
    import matplotlib.colors as mcolors
    
    # Check if we have data crossing zero to determine if we need diverging colormap
    has_negative = vmin < 0
    has_positive = vmax > 0
    
    if has_negative and has_positive:
        # Use diverging normalization centered at zero
        # Make the scale symmetric around zero for balanced visualization
        max_abs = max(abs(vmin), abs(vmax))
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-max_abs, vmax=max_abs)
        # Update vmin/vmax to be symmetric for consistent display
        vmin = -max_abs
        vmax = max_abs
    else:
        # Use regular normalization for data that doesn't cross zero
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    # Plot each category as a separate map
    mappable = None  # Store the mappable for shared colorbar
    
    for i, (category, title) in enumerate(zip(categories, titles)):
        ax = axes[i]
        
        # Filter data for this category
        category_data = gdf[gdf[category_col] == category].copy()
        
        if len(category_data) == 0:
            ax.text(0.5, 0.5, f'No data for\n{category}', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='gray')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
            continue
        
        # Plot the map without individual legend using the diverging normalization
        plot_result = category_data.plot(column=value_col,
                                        cmap=cmap,
                                        legend=False,  # No individual legends
                                        ax=ax,
                                        norm=norm)
        
        # Store the mappable from the first successful plot
        if mappable is None and len(ax.collections) > 0:
            mappable = ax.collections[0]
            
        # Overlay boundary if provided
        if boundary_gdf is not None:
            boundary_gdf.boundary.plot(ax=ax, **boundary_kwargs)
        
        # Clean up the map styling
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Add title
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Hide unused subplots
    for j in range(n_maps, len(axes)):
        axes[j].axis('off')
    
    # Add shared colorbar/legend if requested and we have a mappable
    if legend and mappable is not None:
        # Create shared colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(mappable, cax=cbar_ax, **legend_kwargs)
        cbar.ax.tick_params(labelsize=10)
        
        # Set colorbar label to the value column name
        # Format the label nicely by replacing underscores and capitalizing
        label = value_col.replace('_', ' ').title()
        cbar.set_label(label, rotation=270, labelpad=20, fontsize=11)
    
    # Adjust layout first, then add title and source to avoid tight_layout conflicts
    if legend and mappable is not None:
        plt.subplots_adjust(left=0.05, bottom=0.10, right=0.90, top=0.90, wspace=0.1, hspace=0.1)
        title_x = 0.05
        source_x = 0.05
    else:
        plt.subplots_adjust(left=0.05, bottom=0.10, right=0.98, top=0.90, wspace=0.1, hspace=0.1)
        title_x = 0.05
        source_x = 0.05
    
    # Add overall title (left-aligned) after layout adjustment
    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.98, ha='left', x=title_x)
    
    # Source note (left-aligned) after layout adjustment
    fig.text(source_x, 0.04, source_text, ha='left', va='bottom', fontsize=9, color='gray')
    
    plt.show()


def plot_comparative_lines_subplots(df: pd.DataFrame,
                                   category_col: str,
                                   x_col: str,
                                   value_col: Optional[str] = None,
                                   group_col: str = None,
                                   title: str = "Comparative Line Subplots",
                                   xlabel: str = "X",
                                   ylabel: str = "Y",
                                   source_text: str = "Source: VIIRS Nighttime Lights — Collection 2",
                                   date_col: Optional[str] = None,
                                   ncols: int = 3,
                                   figsize_per_subplot: tuple = (5, 4),
                                   colors: list = None,
                                   markers: list = None,
                                   linewidth: float = 2,
                                   marker_size: float = 6,
                                   alpha: float = 0.8,
                                   legend_location: str = 'best',
                                   share_axes: bool = True,
                                   earthquake_marker: Optional[str] = None) -> None:
    """
    Create comparative line plots as subplots for different categories using World Bank styling
    
    Parameters:
    - df: DataFrame containing the data
    - category_col: Column to create subplots by (e.g., 'ADM1_EN', 'region')
    - x_col: Column for x-axis (e.g., 'year', 'date')
    - value_col: Column for values (if None, will auto-detect)
    - group_col: Column to group by for separate lines within each subplot (e.g., 'year_col', 'category')
    - title: Overall title for the figure
    - xlabel: X-axis label for subplots
    - ylabel: Y-axis label for subplots
    - source_text: Source note for bottom of figure
    - date_col: Optional date column to derive year from
    - ncols: Number of columns in subplot grid
    - figsize_per_subplot: Size multiplier for each subplot
    - colors: List of colors for different lines
    - markers: List of markers for different lines
    - linewidth: Width of lines
    - marker_size: Size of markers
    - alpha: Transparency of lines
    - legend_location: Location of legend in each subplot
    - share_axes: Whether to share x and y axes across subplots
    - earthquake_marker: Optional date string for earthquake marker (e.g., '2025-03-25')
    """
    
    # Auto-detect value column
    if value_col is None:
        value_col = pick_value_column(df)
    assert value_col is not None, f"Could not determine value column"
    
    # Prepare data
    plot_df = df.copy()
    
    # Convert group column to string for consistent handling (especially for years)
    if group_col:
        plot_df[group_col] = plot_df[group_col].astype(str)
    
    # Handle date column if provided - but avoid conflicts with group_col
    x_column = x_col
    
    # Only convert date to year if we're not already using 'year' as group_col
    # and if x_col is asking for 'year' but we have dates
    if (date_col and date_col in plot_df.columns and 
        x_col == 'year' and group_col != 'year'):
        plot_df['derived_year'] = pd.to_datetime(plot_df[date_col]).dt.year
        x_column = 'derived_year'
    elif ('date' in plot_df.columns and x_col == 'year' and 
          group_col != 'year' and 'year' not in plot_df.columns):
        plot_df['derived_year'] = pd.to_datetime(plot_df['date']).dt.year
        x_column = 'derived_year'
    
    # Ensure required columns exist
    assert category_col in plot_df.columns, f"{category_col} column not found"
    assert x_column in plot_df.columns, f"{x_column} column not found"
    assert value_col in plot_df.columns, f"{value_col} column not found"
    if group_col:
        assert group_col in plot_df.columns, f"{group_col} column not found"
    
    # Get unique categories for subplots - preserve categorical order if present
    if pd.api.types.is_categorical_dtype(plot_df[category_col]):
        # Use categorical order if the column is categorical
        categories = [cat for cat in plot_df[category_col].cat.categories if cat in plot_df[category_col].values]
    else:
        # Fall back to sorted order for non-categorical columns
        categories = sorted(plot_df[category_col].dropna().unique())
    n_categories = len(categories)
    
    # Calculate grid dimensions
    nrows = int(np.ceil(n_categories / ncols))
    
    # Create subplots with World Bank styling
    fig, axs = plt.subplots(nrows, ncols, 
                           figsize=(figsize_per_subplot[0]*ncols, figsize_per_subplot[1]*nrows),
                           sharex=share_axes, sharey=share_axes)
    
    # Handle case where only one subplot
    if n_categories == 1:
        axs = [axs]
    else:
        axs = axs.flatten() if nrows > 1 else [axs] if ncols == 1 else axs
    
    # World Bank colors as default
    if colors is None:
        colors = ['#4e79a7', '#f28e2c', '#e15759', '#76b7b2', '#59a14f', 
                 '#edc949', '#af7aa1', '#ff9d9a', '#90ee90', '#17becf']
    if markers is None:
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Create individual plots for each category
    for i, category in enumerate(categories):
        cat_data = plot_df[plot_df[category_col] == category].copy()
        
        if group_col:
            # Aggregate data by group and x within this category
            agg_data = (cat_data[[x_column, group_col, value_col]]
                        .dropna()
                        .groupby([x_column, group_col], as_index=False)[value_col].mean()
                        .sort_values([group_col, x_column]))
            
            # Get unique groups for this category
            groups = sorted(agg_data[group_col].unique())
            
            # Plot each group as a separate line
            for j, group in enumerate(groups):
                group_data = agg_data[agg_data[group_col] == group]
                color = colors[j % len(colors)]
                marker = markers[j % len(markers)] if marker_size > 0 else None
                
                axs[i].plot(group_data[x_column], group_data[value_col], 
                           color=color, marker=marker, linewidth=linewidth, 
                           markersize=marker_size, alpha=alpha, label=str(group))
        else:
            # No grouping - single line per category
            # Aggregate data by x within this category
            agg_data = (cat_data[[x_column, value_col]]
                        .dropna()
                        .groupby(x_column, as_index=False)[value_col].mean()
                        .sort_values(x_column))
            
            color = colors[0]
            marker = markers[0] if marker_size > 0 else None
            
            axs[i].plot(agg_data[x_column], agg_data[value_col], 
                       color=color, marker=marker, linewidth=linewidth, 
                       markersize=marker_size, alpha=alpha)
        
        # Apply World Bank formatting to each subplot
        axs[i].set_xlabel(xlabel, fontweight='normal')
        axs[i].set_ylabel(ylabel, fontweight='normal')
        
        # Set subplot title with World Bank styling (left-aligned, first word bold)
        title_words = str(category).split()
        if len(title_words) > 0:
            # Create formatted title with first word bold
            if len(title_words) > 1:
                formatted_title = f"$\\mathbf{{{title_words[0]}}}$ {' '.join(title_words[1:])}"
            else:
                formatted_title = f"$\\mathbf{{{title_words[0]}}}$"
            
            axs[i].text(0.02, 0.95, formatted_title, transform=axs[i].transAxes, 
                       fontsize=12, verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Remove top and right spines (World Bank style)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['left'].set_color('#cccccc')
        axs[i].spines['bottom'].set_color('#cccccc')
        
        # Grid styling
        axs[i].grid(True, alpha=0.3, color='#cccccc', linewidth=0.5)
        axs[i].set_axisbelow(True)
        
        # Add legend if there are groups
        if group_col and len(cat_data[group_col].unique()) > 1:
            legend = axs[i].legend(loc=legend_location, frameon=True, fancybox=False, 
                                  shadow=False, framealpha=0.9, edgecolor='#cccccc')
            legend.get_frame().set_linewidth(0.5)
        
        # Format x-axis if it's years
        if x_column in ['year', 'Year', 'derived_year']:
            years = sorted(cat_data[x_column].dropna().unique())
            if len(years) <= 20:
                axs[i].set_xticks(years)
                axs[i].set_xticklabels([str(int(y)) for y in years], rotation=45)
        
        # Add earthquake marker if specified
        if earthquake_marker:
            try:
                if pd.api.types.is_datetime64_any_dtype(cat_data[x_column]):
                    axs[i].axvline(pd.Timestamp(earthquake_marker), color='red', linestyle='--', 
                                  linewidth=2, alpha=0.8, zorder=3)
                else:
                    # Try to parse as year if it's not datetime
                    marker_year = pd.to_datetime(earthquake_marker).year
                    axs[i].axvline(marker_year, color='red', linestyle='--', 
                                  linewidth=2, alpha=0.8, zorder=3)
            except Exception:
                # If parsing fails, skip the marker
                pass
    
    # Hide unused axes
    for j in range(n_categories, len(axs)):
        axs[j].axis('off')
    
    # Set overall title with World Bank formatting (left-aligned, first word bold)
    title_words = title.split()
    if len(title_words) > 0:
        # Create formatted main title with first word bold using LaTeX math mode
        if len(title_words) > 1:
            formatted_main_title = f"$\\mathbf{{{title_words[0]}}}$ {' '.join(title_words[1:])}"
        else:
            formatted_main_title = f"$\\mathbf{{{title_words[0]}}}$"
        
        fig.text(0.02, 0.98, formatted_main_title, fontsize=18, fontweight='normal',
                verticalalignment='top', horizontalalignment='left', transform=fig.transFigure)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
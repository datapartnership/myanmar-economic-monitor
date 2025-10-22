import pandas as pd
from typing import Optional, Union, Literal

_DEF_EXCLUDE = {"year","month","date","adm0","adm1","adm2","adm3","adm4","name","code","id"}

def pick_value_column(df: pd.DataFrame):
    prefer = [
        "ntl_sum", "ntl_mean"
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for key in prefer:
        if key in lower_map:
            return lower_map[key]
    num_cols = [c for c in df.select_dtypes("number").columns if c.lower() not in _DEF_EXCLUDE]
    return num_cols[0] if num_cols else None

def annual_ntl_pct_change(df: pd.DataFrame,
                          year_col: str = 'year',
                          value_col: Optional[str] = None,
                          baseline_year: Optional[int] = None,
                          current_year: Optional[int] = None,
                          date_col: Optional[str] = None,
                          mode: Literal['baseline','yoy','baseline_series'] = 'baseline',
                          baseline_years: Optional[Union[tuple, list, range]] = None) -> pd.DataFrame:
    """
    Compute percentage differences for annual NTL data.

    Modes:
    - mode='baseline' (default): percentage change (current - baseline) / baseline * 100
      using either a single baseline_year or a baseline_years range/list (mean over that period),
      with current_year (default: max year). Returns a single-row DataFrame.
    - mode='yoy': year-over-year per-year changes vs previous year for the whole series.
      Returns [year, value, prev_year_value, pct_change].
    - mode='baseline_series': compare every observed year to a baseline mean computed from
      baseline_years (range/list/tuple). Returns [year, value, baseline_period, baseline_value, pct_change].
    """
    if value_col is None:
        value_col = pick_value_column(df)
    if value_col is None:
        raise ValueError('Could not determine value column')

    work = df.copy()

    # Establish a year column
    detected_year_col = None
    if date_col and date_col in work.columns:
        work['year'] = pd.to_datetime(work[date_col]).dt.year
        detected_year_col = 'year'
    elif 'date' in work.columns:
        work['year'] = pd.to_datetime(work['date']).dt.year
        detected_year_col = 'year'
    elif year_col in work.columns:
        detected_year_col = year_col
    else:
        for cand in ['year', 'Year', 'YEAR', 'yr', 'Yr', 'YR']:
            if cand in work.columns:
                detected_year_col = cand
                break
        if detected_year_col is None:
            raise ValueError('Could not determine year column from data; provide date_col or year_col')

    # Aggregate by year to ensure one value per year
    agg = (work[[detected_year_col, value_col]]
           .dropna()
           .groupby(detected_year_col, as_index=False)[value_col].mean()
           .sort_values(detected_year_col))

    # Normalize column name for downstream logic
    agg = agg.rename(columns={detected_year_col: 'year'})

    if mode == 'yoy':
        agg = agg.sort_values('year').reset_index(drop=True)
        agg['prev_year_value'] = agg[value_col].shift(1)
        def safe_pct(curr, prev):
            try:
                return (curr - prev) / prev * 100 if pd.notnull(prev) and prev != 0 else float('nan')
            except Exception:
                return float('nan')
        agg['pct_change'] = agg.apply(lambda r: safe_pct(r[value_col], r['prev_year_value']), axis=1)
        return agg[['year', value_col, 'prev_year_value', 'pct_change']]

    if mode == 'baseline_series':
        if baseline_years is None:
            raise ValueError("baseline_series mode requires baseline_years (tuple/list/range)")
        # Normalize baseline_years into list
        if isinstance(baseline_years, range):
            years_list = list(baseline_years)
        elif isinstance(baseline_years, tuple) and len(baseline_years) == 2:
            start, end = baseline_years
            if start > end:
                start, end = end, start
            years_list = list(range(int(start), int(end) + 1))
        elif isinstance(baseline_years, (list, set)):
            years_list = sorted(int(y) for y in baseline_years)
        else:
            raise ValueError('baseline_years must be a tuple(start,end), list, set, or range')
        base_rows = agg.loc[agg['year'].isin(years_list)]
        if base_rows.empty:
            raise ValueError(f"None of the baseline years {years_list} found in data")
        baseline_val = float(base_rows[value_col].mean())
        baseline_period = f"{years_list[0]}-{years_list[-1]}" if len(years_list) > 1 else str(years_list[0])
        def safe_pct(curr, basev):
            try:
                return (curr - basev) / basev * 100 if pd.notnull(basev) and basev != 0 else float('nan')
            except Exception:
                return float('nan')
        out = agg[['year', value_col]].copy()
        out['baseline_value'] = baseline_val
        out['baseline_period'] = baseline_period
        out['pct_change'] = out.apply(lambda r: safe_pct(r[value_col], baseline_val), axis=1)
        return out[['year', value_col, 'baseline_period', 'baseline_value', 'pct_change']]

    # Default: baseline (single result)
    if current_year is None:
        current_year = int(agg['year'].max())
    curr_row = agg.loc[agg['year'] == current_year]
    if curr_row.empty:
        raise ValueError(f"Current year ({current_year}) not found in data")
    current_val = float(curr_row.iloc[0][value_col])

    baseline_period = None
    baseline_n_years = None

    if baseline_years is not None:
        if isinstance(baseline_years, range):
            years_list = list(baseline_years)
        elif isinstance(baseline_years, tuple) and len(baseline_years) == 2:
            start, end = baseline_years
            if start > end:
                start, end = end, start
            years_list = list(range(int(start), int(end) + 1))
        elif isinstance(baseline_years, (list, set)):
            years_list = sorted(int(y) for y in baseline_years)
        else:
            raise ValueError('baseline_years must be a tuple(start,end), list, set, or range')
        base_rows = agg.loc[agg['year'].isin(years_list)]
        if base_rows.empty:
            raise ValueError(f"None of the baseline years {years_list} found in data")
        baseline_val = float(base_rows[value_col].mean())
        baseline_period = f"{years_list[0]}-{years_list[-1]}" if len(years_list) > 1 else str(years_list[0])
        baseline_n_years = len(base_rows)
        baseline_year_single = years_list[0]
    else:
        if baseline_year is None:
            baseline_year = int(agg['year'].min())
        base_row = agg.loc[agg['year'] == baseline_year]
        if base_row.empty:
            raise ValueError(f"Baseline year ({baseline_year}) not found in data")
        baseline_val = float(base_row.iloc[0][value_col])
        baseline_period = str(int(baseline_year))
        baseline_n_years = 1
        baseline_year_single = int(baseline_year)

    pct_change = (current_val - baseline_val) / baseline_val * 100 if baseline_val != 0 else float('nan')

    out = pd.DataFrame({
        'baseline_year': [baseline_year_single],
        'baseline_period': [baseline_period],
        'baseline_n_years': [baseline_n_years],
        'current_year': [current_year],
        'baseline_value': [baseline_val],
        'current_value': [current_val],
        'pct_change': [pct_change]
    })
    return out


def monthly_ntl_yoy(df: pd.DataFrame,
                    date_col: Optional[str] = None,
                    year_col: str = 'year',
                    month_col: str = 'month',
                    value_col: Optional[str] = None,
                    baseline_year: Optional[int] = None) -> pd.DataFrame:
    """
    Compute percentage difference for monthly NTL data.

    Modes:
    - YoY mode (baseline_year=None): compare each month to the same month in the previous year.
      Output columns include 'prev_year_value' and 'pct_change'.
    - Baseline mode (baseline_year provided): compare each month to the same month in the baseline year.
      Output columns include 'baseline_value' and 'pct_change'.
    """
    if value_col is None:
        value_col = pick_value_column(df)
    if value_col is None:
        raise ValueError('Could not determine value column')

    work = df.copy()

    # Ensure a proper date column exists
    if date_col and date_col in work.columns:
        work['date'] = pd.to_datetime(work[date_col])
        work['year'] = work['date'].dt.year
        work['month'] = work['date'].dt.month
    else:
        if year_col not in work.columns or month_col not in work.columns:
            raise ValueError('Provide date_col or both year_col and month_col')
        work['year'] = work[year_col].astype(int)
        work['month'] = work[month_col].astype(int)
        work['date'] = pd.to_datetime(dict(year=work['year'], month=work['month'], day=1))

    # Aggregate to one value per year-month
    monthly = (work[['year', 'month', 'date', value_col]]
               .dropna()
               .groupby(['year', 'month'], as_index=False)[value_col].mean())

    if baseline_year is None:
        prev = monthly.copy()
        prev['year'] = prev['year'] + 1
        prev = prev.rename(columns={value_col: 'prev_year_value'})
        merged = monthly.merge(prev[['year', 'month', 'prev_year_value']], on=['year', 'month'], how='left')
        merged['date'] = pd.to_datetime(dict(year=merged['year'], month=merged['month'], day=1))
        def safe_pct(curr, prev):
            try:
                return (curr - prev) / prev * 100 if pd.notnull(prev) and prev != 0 else float('nan')
            except Exception:
                return float('nan')
        merged['pct_change'] = merged.apply(lambda r: safe_pct(r[value_col], r['prev_year_value']), axis=1)
        merged = merged[['date', 'year', 'month', value_col, 'prev_year_value', 'pct_change']].sort_values('date')
        return merged
    else:
        base = monthly.loc[monthly['year'] == baseline_year, ['month', value_col]].rename(columns={value_col: 'baseline_value'})
        merged = monthly.merge(base, on='month', how='left')
        merged['date'] = pd.to_datetime(dict(year=merged['year'], month=merged['month'], day=1))
        def safe_pct(curr, basev):
            try:
                return (curr - basev) / basev * 100 if pd.notnull(basev) and basev != 0 else float('nan')
            except Exception:
                return float('nan')
        merged['pct_change'] = merged.apply(lambda r: safe_pct(r[value_col], r['baseline_value']), axis=1)
        merged = merged[['date', 'year', 'month', value_col, 'baseline_value', 'pct_change']].sort_values('date')
        return merged
    
def subnational_pct_change(
    df: pd.DataFrame,
    region_col: str,
    *,
    baseline_year: Optional[int] = None,
    mode: Literal['baseline','yoy'] = 'baseline',
    year_col: str = 'year',
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
    current_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute subnational NTL percentage change by region.

    Parameters:
    - df: tidy table with at least region + year/date + value columns
    - region_col: column indicating region (ADM1/ADM2/ADM3 name or code)
    - baseline_year: baseline year for mode='baseline' (required for baseline mode)
    - mode: 'baseline' (region's current vs baseline year) or 'yoy' (region's year vs previous year)
    - year_col: explicit year column name if present (default 'year')
    - date_col: if provided (or if 'date' exists), derive year from dates
    - value_col: column to use for values; if None, will try pick_value_column(df)
    - current_year: for baseline mode, which year to compare to; default = region's max available year

    Returns:
    - mode='baseline': one row per region with [region, baseline_year, current_year, baseline_value, current_value, pct_change]
    - mode='yoy': per region-year rows with [region, year, value, prev_year_value, pct_change]
    """
    if value_col is None:
        value_col = pick_value_column(df)
    if value_col is None:
        raise ValueError('Could not determine value column')

    work = df.copy()

    # Ensure region exists
    if region_col not in work.columns:
        raise ValueError(f"region_col '{region_col}' not found in DataFrame")

    # Establish a year column
    detected_year_col = None
    if date_col and date_col in work.columns:
        work['year'] = pd.to_datetime(work[date_col]).dt.year
        detected_year_col = 'year'
    elif 'date' in work.columns:
        work['year'] = pd.to_datetime(work['date']).dt.year
        detected_year_col = 'year'
    elif year_col in work.columns:
        detected_year_col = year_col
    else:
        for cand in ['year', 'Year', 'YEAR', 'yr', 'Yr', 'YR']:
            if cand in work.columns:
                detected_year_col = cand
                break
        if detected_year_col is None:
            raise ValueError('Could not determine year column; provide date_col or year_col')

    # Aggregate to one value per region-year
    agg = (work[[region_col, detected_year_col, value_col]]
           .dropna()
           .groupby([region_col, detected_year_col], as_index=False)[value_col].mean()
           .rename(columns={detected_year_col: 'year'}))

    if mode == 'yoy':
        agg = agg.sort_values([region_col, 'year']).reset_index(drop=True)
        agg['prev_year_value'] = agg.groupby(region_col)[value_col].shift(1)
        def safe_pct(curr, prev):
            try:
                return (curr - prev) / prev * 100 if pd.notnull(prev) and prev != 0 else float('nan')
            except Exception:
                return float('nan')
        agg['pct_change'] = agg.apply(lambda r: safe_pct(r[value_col], r['prev_year_value']), axis=1)
        return agg[[region_col, 'year', value_col, 'prev_year_value', 'pct_change']]

    # baseline mode
    if baseline_year is None:
        raise ValueError("baseline_year is required for mode='baseline'")

    # Current year per region (max unless specified)
    if current_year is None:
        curr = (agg.sort_values(['year'])
                  .groupby(region_col, as_index=False)
                  .tail(1)
                  .rename(columns={value_col: 'current_value'}))
    else:
        curr = (agg.loc[agg['year'] == int(current_year), [region_col, 'year', value_col]]
                  .rename(columns={value_col: 'current_value'}))
        curr = curr.assign(current_year=curr['year']).drop(columns=['year'])

    base = (agg.loc[agg['year'] == int(baseline_year), [region_col, value_col]]
              .rename(columns={value_col: 'baseline_value'}))

    out = curr.merge(base, on=region_col, how='left')

    def safe_pct2(curr, basev):
        try:
            return (curr - basev) / basev * 100 if pd.notnull(basev) and basev != 0 else float('nan')
        except Exception:
            return float('nan')

    if 'current_year' not in out.columns:
        out = out.rename(columns={'year': 'current_year'})

    out['baseline_year'] = int(baseline_year)
    out['pct_change'] = out.apply(lambda r: safe_pct2(r['current_value'], r['baseline_value']), axis=1)

    return out[[region_col, 'baseline_year', 'current_year', 'baseline_value', 'current_value', 'pct_change']]


def calculate_percentage_change(df: pd.DataFrame,
                               baseline_filter: dict,
                               group_cols: list,
                               value_col: Optional[str] = None,
                               date_col: Optional[str] = None,
                               comparison_filter: dict = None,
                               mode: Literal['baseline', 'yoy', 'mom'] = 'baseline',
                               suffix: str = '_pct_change') -> pd.DataFrame:
    """
    Flexible function to calculate percentage changes with custom grouping and baseline definition.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    baseline_filter : dict
        Dictionary defining the baseline condition (e.g., {'year': 2022} or {'year': [2020, 2021, 2022]})
    group_cols : list
        List of columns to group by (e.g., ['ADM1_EN'] or ['ADM1_EN', 'month'])
    value_col : str, optional
        Column containing values to calculate percentage change for (auto-detected if None)
    date_col : str, optional
        Date column for time-based operations
    comparison_filter : dict, optional
        Dictionary defining what to compare to baseline (if None, uses all non-baseline data)
    mode : str
        'baseline': Compare to baseline period
        'yoy': Year-over-year comparison
        'mom': Month-over-month comparison
    suffix : str
        Suffix to add to percentage change column name
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with percentage changes calculated
        
    Examples:
    ---------
    # Basic regional comparison: 2024 vs 2022 baseline by region
    pct_change = calculate_percentage_change(
        df=ntl_data,
        baseline_filter={'year': 2022},
        group_cols=['ADM1_EN'],
        comparison_filter={'year': 2024}
    )
    
    # Monthly comparison: Each month vs same month in baseline year
    monthly_pct = calculate_percentage_change(
        df=ntl_monthly,
        baseline_filter={'year': 2022},
        group_cols=['ADM1_EN', 'month'],
        comparison_filter={'year': 2024}
    )
    
    # Multi-year baseline: Compare to average of multiple years
    multi_baseline = calculate_percentage_change(
        df=ntl_data,
        baseline_filter={'year': [2020, 2021, 2022]},
        group_cols=['ADM1_EN']
    )
    """
    
    # Auto-detect value column
    if value_col is None:
        value_col = pick_value_column(df)
        if value_col is None:
            raise ValueError("Could not determine value column")
    
    # Validate inputs
    if not isinstance(baseline_filter, dict):
        raise ValueError("baseline_filter must be a dictionary")
    if not isinstance(group_cols, list):
        raise ValueError("group_cols must be a list")
    
    # Make a copy to avoid modifying original data
    work_df = df.copy()
    
    # Handle date column if provided
    if date_col and date_col in work_df.columns:
        work_df[date_col] = pd.to_datetime(work_df[date_col])
        if 'year' not in work_df.columns:
            work_df['year'] = work_df[date_col].dt.year
        if 'month' not in work_df.columns:
            work_df['month'] = work_df[date_col].dt.month
    
    # Validate required columns exist
    missing_cols = [col for col in group_cols if col not in work_df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataframe: {missing_cols}")
    
    # Handle different modes
    if mode == 'yoy':
        # Year-over-year comparison
        work_df = work_df.sort_values(group_cols + ['year'])
        work_df['prev_year_value'] = work_df.groupby(group_cols)[value_col].shift(1)
        work_df[f'{value_col}{suffix}'] = ((work_df[value_col] - work_df['prev_year_value']) / 
                                          work_df['prev_year_value'] * 100)
        return work_df[group_cols + ['year', value_col, 'prev_year_value', f'{value_col}{suffix}']].dropna()
    
    elif mode == 'mom':
        # Month-over-month comparison
        work_df = work_df.sort_values(group_cols + [date_col])
        work_df['prev_month_value'] = work_df.groupby(group_cols)[value_col].shift(1)
        work_df[f'{value_col}{suffix}'] = ((work_df[value_col] - work_df['prev_month_value']) / 
                                          work_df['prev_month_value'] * 100)
        return work_df[group_cols + [date_col, value_col, 'prev_month_value', f'{value_col}{suffix}']].dropna()
    
    else:  # baseline mode
        # Create baseline data
        baseline_condition = True
        for col, val in baseline_filter.items():
            if isinstance(val, list):
                baseline_condition &= work_df[col].isin(val)
            else:
                baseline_condition &= (work_df[col] == val)
        
        baseline_data = work_df[baseline_condition]
        
        # Calculate baseline values (mean if multiple baseline periods)
        baseline_values = (baseline_data.groupby(group_cols)[value_col]
                          .mean()
                          .reset_index()
                          .rename(columns={value_col: 'baseline_value'}))
        
        # Determine comparison data
        if comparison_filter:
            comparison_condition = True
            for col, val in comparison_filter.items():
                if isinstance(val, list):
                    comparison_condition &= work_df[col].isin(val)
                else:
                    comparison_condition &= (work_df[col] == val)
            comparison_data = work_df[comparison_condition]
        else:
            # Use all non-baseline data
            comparison_data = work_df[~baseline_condition]
        
        # Calculate comparison values
        comparison_values = (comparison_data.groupby(group_cols)[value_col]
                           .mean()
                           .reset_index()
                           .rename(columns={value_col: 'current_value'}))
        
        # Merge and calculate percentage change
        result = comparison_values.merge(baseline_values, on=group_cols, how='inner')
        result[f'{value_col}{suffix}'] = ((result['current_value'] - result['baseline_value']) / 
                                         result['baseline_value'] * 100)
        
        # Add baseline and comparison info
        result['baseline_filter'] = str(baseline_filter)
        if comparison_filter:
            result['comparison_filter'] = str(comparison_filter)
        
        return result


def subnational_monthly_pct_change(
    df: pd.DataFrame,
    region_col: str,
    baseline_year: int,
    *,
    date_col: Optional[str] = None,
    year_col: str = 'year',
    month_col: str = 'month', 
    value_col: Optional[str] = None,
    current_year: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute month-to-month percentage changes from baseline year same month values for subnational regions.
    
    This function compares each month's value for each region to the same month in the baseline year,
    allowing for analysis of seasonal patterns and changes relative to a baseline period.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with monthly data by region
    region_col : str
        Column indicating region (ADM1/ADM2/ADM3 name or code)  
    baseline_year : int
        The baseline year to compare against
    date_col : str, optional
        Date column to extract year/month from (if None, uses year_col and month_col)
    year_col : str, default 'year'
        Column containing year values
    month_col : str, default 'month' 
        Column containing month values
    value_col : str, optional
        Column containing values to analyze (auto-detected if None)
    current_year : int, optional
        Specific year to analyze (if None, uses all available years except baseline)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
        - region_col: Region identifier
        - year: Year of comparison data
        - month: Month (1-12)
        - date: Date (first day of month)
        - value_col: Current month value
        - baseline_value: Same month value from baseline year
        - pct_change: Percentage change from baseline
        
    Examples:
    ---------
    # Compare 2025 monthly values to same months in 2024
    monthly_changes = subnational_monthly_pct_change(
        df=ntl_monthly_adm1,
        region_col='ADM1_EN',
        baseline_year=2024,
        current_year=2025,
        date_col='date'
    )
    
    # Compare all years to baseline 2022
    all_monthly_changes = subnational_monthly_pct_change(
        df=ntl_monthly_data,
        region_col='ADM1_PCODE', 
        baseline_year=2022,
        date_col='date'
    )
    """
    
    # Auto-detect value column
    if value_col is None:
        value_col = pick_value_column(df)
        if value_col is None:
            raise ValueError('Could not determine value column')
    
    # Validate region column exists
    if region_col not in df.columns:
        raise ValueError(f"region_col '{region_col}' not found in DataFrame")
        
    work = df.copy()
    
    # Handle date extraction
    if date_col and date_col in work.columns:
        work['date'] = pd.to_datetime(work[date_col])
        work['year'] = work['date'].dt.year
        work['month'] = work['date'].dt.month
    else:
        # Use existing year and month columns
        if year_col not in work.columns or month_col not in work.columns:
            raise ValueError('Must provide date_col or have both year_col and month_col in data')
        work['year'] = work[year_col].astype(int)
        work['month'] = work[month_col].astype(int)
        work['date'] = pd.to_datetime(dict(year=work['year'], month=work['month'], day=1))
    
    # Aggregate to one value per region-year-month (in case of duplicates)
    agg = (work[[region_col, 'year', 'month', 'date', value_col]]
           .dropna()
           .groupby([region_col, 'year', 'month'], as_index=False)
           .agg({
               'date': 'first',
               value_col: 'mean'
           }))
    
    # Filter to specific year if requested
    if current_year is not None:
        comparison_data = agg[agg['year'] == current_year].copy()
    else:
        # Use all years except baseline
        comparison_data = agg[agg['year'] != baseline_year].copy()
    
    # Get baseline data (same regions and months from baseline year)
    baseline_data = agg[agg['year'] == baseline_year][[region_col, 'month', value_col]].copy()
    baseline_data = baseline_data.rename(columns={value_col: 'baseline_value'})
    
    # Check if baseline year exists
    if baseline_data.empty:
        raise ValueError(f"No data found for baseline year {baseline_year}")
    
    # Merge comparison data with baseline values by region and month
    result = comparison_data.merge(
        baseline_data, 
        on=[region_col, 'month'], 
        how='left'
    )
    
    # Calculate percentage change
    def safe_pct_change(current, baseline):
        try:
            if pd.isnull(current) or pd.isnull(baseline) or baseline == 0:
                return float('nan')
            return (current - baseline) / baseline * 100
        except Exception:
            return float('nan')
    
    result['pct_change'] = result.apply(
        lambda row: safe_pct_change(row[value_col], row['baseline_value']), 
        axis=1
    )
    
    # Add baseline year info
    result['baseline_year'] = baseline_year
    
    # Sort by region, year, month
    result = result.sort_values([region_col, 'year', 'month']).reset_index(drop=True)
    
    # Return with clean column order
    return result[[region_col, 'year', 'month', 'date', value_col, 'baseline_value', 'baseline_year', 'pct_change']]
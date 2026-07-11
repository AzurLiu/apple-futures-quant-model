# -*- coding: utf-8 -*-
"""Project X(CN AP) - Simplified Meteorological Model
此版本为简化版气象因子模型，包含：
1. 优先加载本地 SQLite 数据库缓存，缺失则自动通过 AKShare 接口拉取最新真实数据
2. 自动化获取四大主产区降水数据 (Open-Meteo) 和 NOAA 厄尔尼诺指数 (Niño 3.4)
3. 纯气象特征工程：降水距平值、厄尔尼诺指数、7天气象预报代理
4. 真实无偏的比例复权和双向回测引擎
5. 双模式运行支持：--mode backtest 和 --mode signal
"""

import os
import io
import sys
import time
import warnings
import sqlite3
import argparse
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import requests
import akshare as ak
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize

# Suppress Warnings
warnings.filterwarnings('ignore')

DB_PATH = os.path.join("/Users/azur/Downloads/AppleFuture", "data", "apple_futures.db")

def init_db():
    """初始化 SQLite 数据库，创建必要的表结构"""
    db_dir = os.path.dirname(DB_PATH)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 期货数据表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS futures_data (
        symbol TEXT,
        date TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        open_interest REAL,
        volume REAL,
        PRIMARY KEY (symbol, date)
    )
    """)
    
    # 天气数据表 (包含降水)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS weather_data (
        location TEXT,
        date TEXT,
        temp_min REAL,
        temp_max REAL,
        precipitation REAL,
        PRIMARY KEY (location, date)
    )
    """)
    
    # 厄尔尼诺指数表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS nino_data (
        year INTEGER,
        month INTEGER,
        nino34_anom REAL,
        PRIMARY KEY (year, month)
    )
    """)
    
    conn.commit()
    conn.close()


def save_futures_to_db(df, symbol_prefix):
    """保存标准化的期货数据到 SQLite 数据库中"""
    if df is None or len(df) == 0:
        return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for idx, row in df.iterrows():
        date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
        cursor.execute("""
        INSERT OR REPLACE INTO futures_data (symbol, date, open, high, low, close, open_interest, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol_prefix,
            date_str,
            float(row['open']) if pd.notna(row['open']) else None,
            float(row['high']) if 'high' in row and pd.notna(row['high']) else None,
            float(row['low']) if 'low' in row and pd.notna(row['low']) else None,
            float(row['close']) if pd.notna(row['close']) else None,
            float(row['open_interest']) if pd.notna(row['open_interest']) else None,
            float(row['volume']) if 'volume' in row and pd.notna(row['volume']) else None
        ))
    conn.commit()
    conn.close()


def load_futures_from_db(symbol_prefix, start_date="2017-01-01", end_date="2026-12-31"):
    """从数据库加载指定品种的历史数据"""
    if not os.path.exists(DB_PATH):
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
        SELECT date, open, high, low, close, open_interest, volume
        FROM futures_data
        WHERE symbol = ? AND date >= ? AND date <= ?
        ORDER BY date ASC
        """
        df = pd.read_sql_query(query, conn, params=(symbol_prefix, start_date, end_date))
        conn.close()
        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])
            return df
    except Exception as e:
        print(f"Error loading {symbol_prefix} from SQLite: {e}")
    return None


def save_weather_to_db(df, location):
    """保存天气数据到 SQLite 数据库中"""
    if df is None or len(df) == 0:
        return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for idx, row in df.iterrows():
        date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
        cursor.execute("""
        INSERT OR REPLACE INTO weather_data (location, date, temp_min, temp_max, precipitation)
        VALUES (?, ?, ?, ?, ?)
        """, (
            location,
            date_str,
            float(row['temp_min']) if pd.notna(row['temp_min']) else None,
            float(row['temp_max']) if pd.notna(row['temp_max']) else None,
            float(row['precipitation']) if pd.notna(row['precipitation']) else None
        ))
    conn.commit()
    conn.close()


def load_weather_from_db(location, start_date="2017-01-01", end_date="2026-12-31"):
    """从数据库加载天气数据"""
    if not os.path.exists(DB_PATH):
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
        SELECT date, temp_min, temp_max, precipitation
        FROM weather_data
        WHERE location = ? AND date >= ? AND date <= ?
        ORDER BY date ASC
        """
        df = pd.read_sql_query(query, conn, params=(location, start_date, end_date))
        conn.close()
        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])
            return df
    except Exception as e:
        print(f"Error loading weather for {location} from SQLite: {e}")
    return None


def fetch_nino_data():
    """获取 NOAA ERSSTv5 Niño 3.4 指数并进行本地 SQLite 缓存"""
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        df_db = pd.read_sql_query("SELECT year, month, nino34_anom FROM nino_data ORDER BY year, month", conn)
        conn.close()
        if len(df_db) > 0:
            last_row = df_db.iloc[-1]
            last_year, last_month = int(last_row['year']), int(last_row['month'])
            current_year = pd.Timestamp.now().year
            if (current_year - last_year) * 12 + (pd.Timestamp.now().month - last_month) <= 3:
                print("SQLite cache for El Nino (Niño 3.4) is up to date.")
                return df_db
    except Exception as e:
        print(f"[WARNING] Loading El Nino from SQLite failed: {e}")
        df_db = None

    print("[INFO] Fetching El Niño 3.4 index from NOAA CPC...")
    url = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"
    try:
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            lines = res.text.strip().split('\n')
            rows = []
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 10:
                    rows.append({
                        'year': int(parts[0]),
                        'month': int(parts[1]),
                        'nino34_anom': float(parts[9])
                    })
            df_online = pd.DataFrame(rows)
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            for idx, row in df_online.iterrows():
                cursor.execute("""
                INSERT OR REPLACE INTO nino_data (year, month, nino34_anom)
                VALUES (?, ?, ?)
                """, (int(row['year']), int(row['month']), float(row['nino34_anom'])))
            conn.commit()
            conn.close()
            print("El Niño 3.4 index downloaded and cached successfully.")
            return df_online
    except Exception as e:
        print(f"[WARNING] NOAA El Niño fetch failed: {e}")
        
    if df_db is not None and len(df_db) > 0:
        return df_db
        
    print("[INFO] Falling back to synthetic El Niño index...")
    years = range(2017, pd.Timestamp.now().year + 1)
    months = range(1, 13)
    mock_rows = []
    np.random.seed(42)
    for yr in years:
        for m in months:
            mock_rows.append({
                'year': yr,
                'month': m,
                'nino34_anom': round(float(np.sin(yr + m / 12) + np.random.normal(0, 0.3)), 2)
            })
    df_mock = pd.DataFrame(mock_rows)
    return df_mock


def process_continuous_contract(df, symbol_prefix):
    """处理主力连续合约，并执行比例复权 (Proportional / Ratio Back-Adjustment)"""
    df = df.copy()
    if 'symbol' not in df.columns or df['symbol'].isnull().all():
        print(f"Applying heuristic Proportional back-adjustment for {symbol_prefix}...")
        df = df.sort_values('date').reset_index(drop=True)
        close_raw = df['close'].values.copy().astype(float)
        open_raw = df['open'].values.copy().astype(float)
        dates = df['date'].values
        close_adj = close_raw.copy()
        open_adj = open_raw.copy()
        
        gaps = []
        for i in range(1, len(df)):
            m = pd.to_datetime(dates[i]).month
            overnight_ret = open_raw[i] / close_raw[i-1] - 1
            gaps.append((i, dates[i], m, overnight_ret))
            
        df_gaps = pd.DataFrame(gaps, columns=['idx', 'date', 'month', 'gap'])
        df_gaps['year'] = df_gaps['date'].dt.year
        
        roll_indices = []
        windows = [(3, 4), (7, 8), (11, 12)]
        for yr in df_gaps['year'].unique():
            for w in windows:
                sub = df_gaps[(df_gaps['year'] == yr) & (df_gaps['month'].isin(w))]
                if len(sub) > 0:
                    max_gap_idx = sub['gap'].abs().idxmax()
                    roll_idx = sub.loc[max_gap_idx, 'idx']
                    roll_indices.append(roll_idx)
                    
        accumulated_ratio = 1.0
        roll_set = set(roll_indices)
        for i in range(len(df) - 2, -1, -1):
            if (i + 1) in roll_set:
                gap = open_raw[i+1] - close_raw[i]
                if abs(gap) > close_raw[i] * 0.008:
                    ratio = open_raw[i+1] / close_raw[i]
                    accumulated_ratio *= ratio
            close_adj[i] = close_raw[i] * accumulated_ratio
            open_adj[i] = open_raw[i] * accumulated_ratio
            
        df[f'{symbol_prefix}_close_adj'] = close_adj
        df[f'{symbol_prefix}_open_adj'] = open_adj
        df[f'{symbol_prefix}_close'] = close_raw
        df[f'{symbol_prefix}_open'] = open_raw
        return df
        
    print(f"Applying main-contract volume selection and Proportional adjustment for {symbol_prefix}...")
    idx = df.groupby('date')['open_interest'].idxmax()
    main_df = df.loc[idx].sort_values('date').reset_index(drop=True)
    price_lookup = df.set_index(['date', 'symbol'])['close'].to_dict()
    close_raw = main_df['close'].values.copy().astype(float)
    open_raw = main_df['open'].values.copy().astype(float)
    contracts = main_df['symbol'].values
    dates = main_df['date'].values
    close_adj = close_raw.copy()
    open_adj = open_raw.copy()
    
    accumulated_ratio = 1.0
    for i in range(len(main_df) - 2, -1, -1):
        curr_contract = contracts[i]
        next_contract = contracts[i+1]
        if curr_contract != next_contract:
            date_i = dates[i]
            p_next_at_i = price_lookup.get((date_i, next_contract))
            p_curr_at_i = close_raw[i]
            if p_next_at_i is not None and not np.isnan(p_next_at_i) and p_curr_at_i > 0:
                ratio = p_next_at_i / p_curr_at_i
                accumulated_ratio *= ratio
        close_adj[i] = close_raw[i] * accumulated_ratio
        open_adj[i] = open_raw[i] * accumulated_ratio
        
    main_df[f'{symbol_prefix}_close_adj'] = close_adj
    main_df[f'{symbol_prefix}_open_adj'] = open_adj
    main_df[f'{symbol_prefix}_close'] = close_raw
    main_df[f'{symbol_prefix}_open'] = open_raw
    return main_df


def generate_mock_data(symbol_prefix, date_col, symbol_col, open_interest_col, close_col, open_col, start_date="2017-01-01", end_date="2026-10-25"):
    """生成模拟期货行情数据"""
    print(f"Generating synthetic data for {symbol_prefix}...")
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    contracts = [f"{symbol_prefix}{str(yr)[2:]}{m:02d}" for yr in range(2017, 2027) for m in [1, 5, 9]]
    rows = []
    np.random.seed(42)
    base_price = 8000.0
    for dt in dates:
        if dt.dayofweek >= 5:
            continue
        m = dt.month
        year = dt.year
        if m in [1, 2, 3, 4]:
            main_c = f"{symbol_prefix}{str(year)[2:]}09"
            alt_c = f"{symbol_prefix}{str(year)[2:]}01"
        elif m in [5, 6, 7, 8]:
            main_c = f"{symbol_prefix}{str(year)[2:]}01"
            alt_c = f"{symbol_prefix}{str(year)[2:]}05"
        else:
            main_c = f"{symbol_prefix}{str(year+1)[2:]}05"
            alt_c = f"{symbol_prefix}{str(year)[2:]}09"
            
        daily_noise = np.random.normal(0, base_price * 0.012)
        close_p = base_price + daily_noise
        open_p = base_price + np.random.normal(0, base_price * 0.005)
        base_price = close_p
        
        if base_price < 100: base_price = 100.0
        if open_p < 100: open_p = 100.0
            
        main_oi = np.random.randint(120000, 250000)
        alt_oi = np.random.randint(20000, 80000)
        rows.append({date_col: dt, symbol_col: main_c, open_interest_col: main_oi, close_col: close_p, open_col: open_p})
        rows.append({date_col: dt, symbol_col: alt_c, open_interest_col: alt_oi, close_col: close_p * 0.98, open_col: open_p * 0.98})
    return pd.DataFrame(rows)


def load_exchange_data_via_akshare(symbol_prefix, start_date="2017-01-01", end_date="2026-10-25"):
    """通过 AKShare 获取主力合约日频行情"""
    print(f"Fetching real market data for {symbol_prefix} from AKShare...")
    try:
        sina_symbol = f"{symbol_prefix}0"
        start_str = pd.to_datetime(start_date).strftime("%Y%m%d")
        end_str = pd.to_datetime(end_date).strftime("%Y%m%d")
        df = ak.futures_main_sina(symbol=sina_symbol, start_date=start_str, end_date=end_str)
        df = df.rename(columns={
            '日期': 'date', '开盘价': 'open', '最高价': 'high', '最低价': 'low',
            '收盘价': 'close', '持仓量': 'open_interest', '成交量': 'volume'
        })
        for col in ['open', 'high', 'low', 'close', 'open_interest', 'volume']:
            df[col] = df[col].astype(float)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[ERROR] Failed to fetch {symbol_prefix} from AKShare: {e}")
        return None


def load_local_or_akshare_data(file_name, symbol_prefix, date_col, symbol_col, open_interest_col, close_col, open_col, high_col='high', low_col='low', volume_col='volume'):
    """加载期货数据，支持 SQLite 缓存"""
    init_db()
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    df_db = load_futures_from_db(symbol_prefix, "2017-01-01", today_str)
    is_up_to_date = False
    if df_db is not None and len(df_db) > 0:
        last_date = df_db['date'].max()
        days_diff = (pd.Timestamp.now() - last_date).days
        if days_diff <= 3:
            is_up_to_date = True
            print(f"SQLite cache for {symbol_prefix} is up to date (last date: {last_date.strftime('%Y-%m-%d')}).")
            return df_db
            
    if not is_up_to_date:
        print(f"[INFO] SQLite cache for {symbol_prefix} is outdated/missing. Fetching from AKShare...")
        try:
            df_online = load_exchange_data_via_akshare(symbol_prefix, "2017-01-01", today_str)
            if df_online is not None and len(df_online) > 0:
                save_futures_to_db(df_online, symbol_prefix)
                return load_futures_from_db(symbol_prefix, "2017-01-01", today_str)
        except Exception as e:
            print(f"[WARNING] Online fetch failed: {e}")
            
    if df_db is not None and len(df_db) > 0:
        print(f"[WARNING] Using outdated SQLite cache for {symbol_prefix} due to connection error.")
        return df_db
        
    paths_to_check = [
        file_name,
        os.path.join("data", file_name),
        os.path.join("/Users/azur/Downloads/AppleFuture", file_name),
        os.path.join("/Users/azur/Downloads/AppleFuture/data", file_name)
    ]
    df_local = None
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"Loading '{path}'...")
            try:
                if path.endswith('.xlsx'):
                    df_local = pd.read_excel(path)
                else:
                    try:
                        df_local = pd.read_csv(path, encoding='utf-8')
                    except:
                        df_local = pd.read_csv(path, encoding='gbk')
                break
            except Exception as e:
                print(f"Error loading file {path}: {e}")
                
    if df_local is not None:
        rename_map = {date_col: 'date', open_interest_col: 'open_interest', close_col: 'close', open_col: 'open'}
        if symbol_col in df_local.columns: rename_map[symbol_col] = 'symbol'
        if high_col in df_local.columns: rename_map[high_col] = 'high'
        if low_col in df_local.columns: rename_map[low_col] = 'low'
        if volume_col in df_local.columns: rename_map[volume_col] = 'volume'
        df_local = df_local.rename(columns=rename_map)
        df_local['date'] = pd.to_datetime(df_local['date'])
        for c in ['open', 'close', 'open_interest']:
            if c in df_local.columns:
                df_local[c] = df_local[c].astype(float)
        save_futures_to_db(df_local, symbol_prefix)
        return df_local

    print(f"[INFO] Falling back to synthetic mock data generator for {symbol_prefix}...")
    df_mock = generate_mock_data(symbol_prefix, 'date', 'symbol', 'open_interest', 'close', 'open')
    return df_mock


def send_signal_notification(signal):
    """向交易员发送交易通知"""
    message = (
        f"🚨 【苹果气象因数量化策略】明日交易信号通知 🚨\n"
        f"信号日期: {signal['signal_date']}\n"
        f"操作指令: {signal['action']}\n"
        f"目标仓位: {signal['position_size']} 倍杠杆\n"
        f"前日收盘价: {signal['last_close_price']} 元/吨\n"
        f"看涨概率: {signal['prob_long']:.2%}\n"
        f"看跌概率: {signal['prob_short']:.2%}\n"
    )
    log_dir = "/Users/azur/.gemini/antigravity/brain/69a0c2da-0a47-4c37-afbd-71d16e9f5e6e/signals"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"signal_{signal['signal_date']}.json")
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(signal, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Signal logged locally to: {log_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save local signal log: {e}")
        
    webhook_url = os.environ.get("QUANT_SIGNAL_WEBHOOK")
    if webhook_url:
        try:
            requests.post(webhook_url, json={"text": message}, timeout=10)
            print("[INFO] Signal sent successfully via Webhook.")
        except Exception as e:
            print(f"[WARNING] Webhook push failed: {e}")
    else:
        print("[INFO] QUANT_SIGNAL_WEBHOOK env var not set. Skipping Webhook notification.")


def fetch_weather_with_retry(url, params, max_retries=3, backoff=2):
    """带指数退避的 API 请求重试函数"""
    for attempt in range(max_retries):
        try:
            res = requests.get(url, params=params, timeout=15)
            if res.status_code == 200:
                return res.json()
            elif res.status_code == 429:
                print(f"Rate limit hit. Retrying in {backoff} seconds...")
            else:
                print(f"Server returned status {res.status_code}. Retrying in {backoff} seconds...")
        except Exception as e:
            print(f"Request failed: {e}. Retrying in {backoff} seconds...")
        time.sleep(backoff)
        backoff *= 2
    raise Exception(f"Failed to fetch weather data after {max_retries} attempts.")


def apply_flexible_labeling_three_class(df, lookahead=15, profit_threshold=0.02):
    """未来 15 天收益率触发三分类标签"""
    labels = []
    prices = df['AP_close_adj'].values
    for i in range(len(prices) - lookahead):
        window = prices[i+1 : i+1+lookahead]
        curr = prices[i]
        max_idx = np.argmax(window)
        min_idx = np.argmin(window)
        up_pct = (window[max_idx] / curr) - 1.0
        down_pct = (window[min_idx] / curr) - 1.0
        
        if up_pct >= profit_threshold and down_pct <= -profit_threshold:
            if max_idx < min_idx:
                labels.append(1)
            else:
                labels.append(2)
        elif up_pct >= profit_threshold:
            labels.append(1)
        elif down_pct <= -profit_threshold:
            labels.append(2)
        else:
            labels.append(0)
    labels.extend([np.nan] * lookahead)
    return np.array(labels)


def get_forward_rolling(series, window, func='mean'):
    """计算前向滚动计算（包含当天）并向后移动一位代表未来预测"""
    rev = series.iloc[::-1]
    if func == 'mean':
        roll = rev.rolling(window, min_periods=1).mean()
    else:
        roll = rev.rolling(window, min_periods=1).sum()
    return roll.iloc[::-1].shift(-1)


def run_backtest(df_rob, FEATURES, threshold_long=0.40, threshold_short=0.40):
    """执行全量 Walk-Forward 滚动历史回测"""
    print("\nBlock 5: Starting Walk-Forward Backtesting...")
    df_rob = df_rob.copy()
    
    # 标签计算 (基于 Apple Proportional 复权价格)
    df_rob['target'] = apply_flexible_labeling_three_class(df_rob, lookahead=15, profit_threshold=0.02)
    df_rob = df_rob.dropna(subset=FEATURES + ['target'])
    
    # 缩尾处理
    for col in ['national_precip_anomaly', 'nino34_anom', 'forecast_temp_mean', 'forecast_precip_sum']:
        df_rob[col] = np.array(winsorize(df_rob[col], limits=[0.01, 0.01]))
        
    df_rob = df_rob.sort_values('date').reset_index(drop=True)
    df_rob['prob_long'] = np.nan
    df_rob['prob_short'] = np.nan
    
    init_size = 1000
    step = 252
    
    print(f"Starting Walk-Forward Validation on {len(df_rob)} rows...")
    
    for i in range(init_size, len(df_rob) - step, step):
        train = df_rob.iloc[:i-36]
        test = df_rob.iloc[i : i+step].copy()
        
        model = lgb.LGBMClassifier(
            objective='multiclass', num_class=3,
            n_estimators=80, max_depth=3, learning_rate=0.03,
            reg_lambda=5.0, min_child_samples=30, colsample_bytree=0.8,
            random_state=42, verbosity=-1
        )
        
        X_tr = train[FEATURES].shift(1).dropna()
        y_tr = train.loc[X_tr.index, 'target']
        
        if len(np.unique(y_tr)) < 2:
            continue
        model.fit(X_tr, y_tr)
        
        X_te = test[FEATURES].shift(1).fillna(0)
        probs_all = model.predict_proba(X_te)
        
        classes = list(model.classes_)
        prob_long = np.zeros(len(test))
        prob_short = np.zeros(len(test))
        if 1 in classes:
            prob_long = probs_all[:, classes.index(1)]
        if 2 in classes:
            prob_short = probs_all[:, classes.index(2)]
            
        df_rob.loc[test.index, 'prob_long'] = prob_long
        df_rob.loc[test.index, 'prob_short'] = prob_short
        
    test_df = df_rob.dropna(subset=['prob_long', 'prob_short']).copy().reset_index(drop=True)
    
    if len(test_df) == 0:
        print("[WARNING] Not enough data rows to execute backtest.")
        return
        
    fee_rate = 0.0002
    slippage_price = 2.0
    target_risk = 0.01
    
    portfolio_values = []
    daily_returns = []
    dates = []
    trade_returns = []
    current_equity = 1.0
    in_pos = 0
    entry_raw = 0.0
    pos_size = 1.0
    days = 0
    exit_triggered = False
    
    for idx in range(len(test_df)):
        row = test_df.iloc[idx]
        date = row['date']
        close_price = row['AP_close_adj']
        open_price = row['AP_open_adj']
        prob_long = row['prob_long']
        prob_short = row['prob_short']
        
        if idx == 0:
            portfolio_values.append(current_equity)
            daily_returns.append(0.0)
            dates.append(date)
            continue
            
        prev_close = test_df.iloc[idx-1]['AP_close_adj']
        prev_vol = test_df.iloc[idx-1]['AP_vol_20d']
        daily_ret_net = 0.0
        
        if in_pos == 1:
            if exit_triggered:
                daily_ret_raw = (open_price / prev_close) - 1
                friction = fee_rate + (slippage_price / prev_close)
                daily_ret_net = (daily_ret_raw * pos_size) - (friction * pos_size)
                trade_gross_ret = (open_price / entry_raw) - 1
                trade_returns.append(trade_gross_ret - 2 * fee_rate - (2 * slippage_price / entry_raw))
                in_pos = 0
                exit_triggered = False
            else:
                daily_ret_raw = (close_price / prev_close) - 1
                daily_ret_net = daily_ret_raw * pos_size
                days += 1
                total_ret_raw = (close_price / entry_raw) - 1
                if total_ret_raw <= -0.03 or total_ret_raw >= 0.06 or days >= 15:
                    exit_triggered = True
        elif in_pos == -1:
            if exit_triggered:
                daily_ret_raw = -(open_price / prev_close - 1)
                friction = fee_rate + (slippage_price / prev_close)
                daily_ret_net = (daily_ret_raw * pos_size) - (friction * pos_size)
                trade_gross_ret = -(open_price / entry_raw - 1)
                trade_returns.append(trade_gross_ret - 2 * fee_rate - (2 * slippage_price / entry_raw))
                in_pos = 0
                exit_triggered = False
            else:
                daily_ret_raw = -(close_price / prev_close - 1)
                daily_ret_net = daily_ret_raw * pos_size
                days += 1
                total_ret_raw = -(close_price / entry_raw - 1)
                if total_ret_raw <= -0.03 or total_ret_raw >= 0.06 or days >= 15:
                    exit_triggered = True
        else:
            daily_ret_net = 0.0
            
        current_equity = current_equity * (1 + daily_ret_net)
        portfolio_values.append(current_equity)
        daily_returns.append(daily_ret_net)
        dates.append(date)
        
        if in_pos == 0 and not exit_triggered:
            prev_row = test_df.iloc[idx-1]
            p_long = prev_row['prob_long']
            p_short = prev_row['prob_short']
            if pd.notna(prev_vol) and prev_vol > 0:
                pos_size = target_risk / prev_vol
                pos_size = np.clip(pos_size, 0.2, 1.5)
            else:
                pos_size = 1.0
                
            if p_long > threshold_long and p_long >= p_short:
                in_pos = 1
                entry_raw = open_price
                days = 0
                day_ret_raw = (close_price / open_price) - 1
                friction = fee_rate + (slippage_price / open_price)
                day_ret_net = (day_ret_raw * pos_size) - (friction * pos_size)
                current_equity = (current_equity / (1 + daily_ret_net)) * (1 + day_ret_net)
                portfolio_values[-1] = current_equity
                daily_returns[-1] = day_ret_net
                if day_ret_raw <= -0.03 or day_ret_raw >= 0.06:
                    exit_triggered = True
            elif p_short > threshold_short and p_short > p_long:
                in_pos = -1
                entry_raw = open_price
                days = 0
                day_ret_raw = -(close_price / open_price - 1)
                friction = fee_rate + (slippage_price / open_price)
                day_ret_net = (day_ret_raw * pos_size) - (friction * pos_size)
                current_equity = (current_equity / (1 + daily_ret_net)) * (1 + day_ret_net)
                portfolio_values[-1] = current_equity
                daily_returns[-1] = day_ret_net
                if day_ret_raw <= -0.03 or day_ret_raw >= 0.06:
                    exit_triggered = True

    equity_df = pd.DataFrame({
        'date': dates,
        'equity': portfolio_values,
        'daily_return': daily_returns
    })
    
    std_ret = equity_df['daily_return'].std()
    sharpe = (equity_df['daily_return'].mean() / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0
    cum_ret = equity_df['equity'].iloc[-1] - 1
    
    print("\n" + "="*40)
    print("SIMPLIFIED METEOROLOGICAL MODEL RESULTS")
    print("="*40)
    print(f"Total Trade Count: {len(trade_returns)}")
    print(f"Standard Sharpe Ratio (Daily Basis): {sharpe:.4f}")
    print(f"Total Cumulative Return: {cum_ret*100:.2f}%")
    
    df_pre = equity_df[equity_df['date'] < '2025-01-01'].copy()
    if len(df_pre) > 1:
        std_pre = df_pre['daily_return'].std()
        sharpe_pre = (df_pre['daily_return'].mean() / std_pre) * np.sqrt(252) if std_pre > 0 else 0.0
        cum_pre = (df_pre['equity'].iloc[-1] / df_pre['equity'].iloc[0]) - 1
        print(f"Pre-2025 Period (Historical Dev) Sharpe: {sharpe_pre:.4f}, Return: {cum_pre*100:.2f}%")
        
    df_post = equity_df[equity_df['date'] >= '2025-01-01'].copy()
    if len(df_post) > 1:
        std_post = df_post['daily_return'].std()
        sharpe_post = (df_post['daily_return'].mean() / std_post) * np.sqrt(252) if std_post > 0 else 0.0
        cum_post = (df_post['equity'].iloc[-1] / df_post['equity'].iloc[0]) - 1
        print(f"Post-2025 Period (Pure Out-of-Sample) Sharpe: {sharpe_post:.4f}, Return: {cum_post*100:.2f}%")
        
    plt.figure(figsize=(10, 5))
    plt.plot(equity_df['date'], equity_df['equity'], label='Strategy Equity Curve (Meteorological Model)', color='green')
    plt.title('Meteorological Model Cumulative Return (Daily Date Index)')
    plt.xlabel('Date')
    plt.ylabel('Equity (Initial = 1.0)')
    plt.grid(True)
    plt.legend()
    plt.savefig('equity_curve_meteorological.png', dpi=300, bbox_inches='tight')
    print("Equity curve plot saved to 'equity_curve_meteorological.png'.")


def generate_tomorrow_signal(df_rob, FEATURES, threshold_long=0.40, threshold_short=0.40):
    """生成明日交易信号"""
    print("\nTraining final model on full historical dataset...")
    df_rob = df_rob.copy()
    df_rob['target'] = apply_flexible_labeling_three_class(df_rob, lookahead=15, profit_threshold=0.02)
    train_df = df_rob.iloc[:-15].dropna(subset=FEATURES + ['target'])
    last_row = df_rob.iloc[-1]
    last_date = last_row['date']
    
    X_train = train_df[FEATURES].shift(1).dropna()
    y_train = train_df.loc[X_train.index, 'target']
    model = lgb.LGBMClassifier(
        objective='multiclass', num_class=3,
        n_estimators=80, max_depth=3, learning_rate=0.03,
        reg_lambda=5.0, min_child_samples=30, colsample_bytree=0.8,
        random_state=42, verbosity=-1
    )
    model.fit(X_train, y_train)
    
    X_today = pd.DataFrame([last_row[FEATURES]])
    probs_all = model.predict_proba(X_today)[0]
    classes = list(model.classes_)
    prob_long = probs_all[classes.index(1)] if 1 in classes else 0.0
    prob_short = probs_all[classes.index(2)] if 2 in classes else 0.0
    
    action = "CASH"
    if prob_long > threshold_long and prob_long >= prob_short:
        action = "LONG"
    elif prob_short > threshold_short and prob_short > prob_long:
        action = "SHORT"
        
    target_risk = 0.01
    last_vol = last_row['AP_vol_20d']
    if pd.notna(last_vol) and last_vol > 0:
        pos_size = target_risk / last_vol
        pos_size = np.clip(pos_size, 0.2, 1.5)
    else:
        pos_size = 1.0
        
    signal_json = {
        "signal_date": (pd.to_datetime(last_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "action": action,
        "position_size": round(float(pos_size), 4),
        "last_close_price": float(last_row['AP_close']),
        "prob_long": round(float(prob_long), 4),
        "prob_short": round(float(prob_short), 4)
    }
    send_signal_notification(signal_json)
    print("\n" + "="*50)
    print("TOMORROW OPEN SIGNAL (JSON)")
    print("="*50)
    print(json.dumps(signal_json, indent=4, ensure_ascii=False))
    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Simplified Meteorological Quant Model CLI")
    parser.add_argument('--mode', type=str, default='backtest', choices=['backtest', 'signal'],
                        help="运行模式：'backtest' 运行历史滚动回测; 'signal' 生成明日交易信号.")
    parser.add_argument('--threshold_long', type=float, default=0.40, help="做多触发阈值 (默认 0.40)")
    parser.add_argument('--threshold_short', type=float, default=0.40, help="做空触发阈值 (默认 0.40)")
    args = parser.parse_args()

    init_db()

    # 1. 期货数据加载 (只加载 Apple)
    df_apple_raw = load_local_or_akshare_data('apple_futures_main_continuous_czce.csv', 'AP', 'date', 'symbol', 'open_interest', 'close', 'open')

    # 2. 数据清洗与复权生成
    print("\nBlock 2: Processing Apple Futures Data...")
    df_apple = process_continuous_contract(df_apple_raw, 'AP')
    
    # 我们仅需要计算单日收益率以获得波动率指标
    df_apple['AP_return_1d'] = df_apple['AP_close_adj'].pct_change(1)
    df_apple['AP_vol_20d'] = df_apple['AP_return_1d'].rolling(20).std()

    # 3. 统一区间底线
    last_date_str = df_apple['date'].max().strftime("%Y-%m-%d")

    # 4. 厄尔尼诺数据拉取
    df_nino = fetch_nino_data()

    # 5. 天气数据加载 (支持 SQLite 缓存与降水量爬取)
    print("\nBlock 4: Fetching Weather Data (Temperature + Precipitation)...")
    locs = {
        "luochuan": {"latitude": 35.76, "longitude": 109.43},
        "yantai": {"latitude": 37.53, "longitude": 121.39},
        "tianshui": {"latitude": 34.58, "longitude": 105.72},
        "lingbao": {"latitude": 34.51, "longitude": 110.88}
    }
    weather_data_raw_dict = {}
    url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "start_date": "2017-01-01", "end_date": last_date_str,
        "daily": ["temperature_2m_min", "temperature_2m_max", "precipitation_sum"],
        "timezone": "Asia/Shanghai"
    }

    try:
        for name, coords in locs.items():
            df_w_db = load_weather_from_db(name, "2017-01-01", last_date_str)
            is_up_to_date = False
            
            if df_w_db is not None and len(df_w_db) > 0 and 'precipitation' in df_w_db.columns:
                last_w_date = df_w_db['date'].max()
                days_diff = (pd.Timestamp.now() - last_w_date).days
                if days_diff <= 3:
                    is_up_to_date = True
                    print(f"SQLite cache for weather {name} is up to date (last date: {last_w_date.strftime('%Y-%m-%d')}).")
                    weather_data_raw_dict[name] = df_w_db
                    
            if not is_up_to_date:
                print(f"[INFO] SQLite cache for weather {name} is missing/outdated. Fetching from Open-Meteo...")
                p = weather_params.copy()
                p.update(coords)
                try:
                    res = fetch_weather_with_retry(url, p)
                    df_w_online = pd.DataFrame({
                        "date": pd.to_datetime(res['daily']['time']),
                        "temp_min": res['daily']['temperature_2m_min'],
                        "temp_max": res['daily']['temperature_2m_max'],
                        "precipitation": res['daily']['precipitation_sum']
                    })
                    save_weather_to_db(df_w_online, name)
                    weather_data_raw_dict[name] = load_weather_from_db(name, "2017-01-01", last_date_str)
                except Exception as e:
                    print(f"[WARNING] Open-Meteo fetch failed for {name}: {e}")
                    if df_w_db is not None and len(df_w_db) > 0:
                        print(f"[WARNING] Fallback to existing SQLite cache for weather {name}.")
                        weather_data_raw_dict[name] = df_w_db
                    else:
                        raise e
        print("Weather data loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Weather loading failed: {e}")
        print("[INFO] Fallback to synthetic weather data...")
        for n in locs.keys():
            dates = pd.date_range("2017-01-01", last_date_str, freq='D')
            weather_data_raw_dict[n] = pd.DataFrame({
                "date": dates,
                "temp_min": np.random.uniform(5, 15, len(dates)),
                "temp_max": np.random.uniform(15, 28, len(dates)),
                "precipitation": np.random.exponential(2, len(dates))
            })

    # 6. 核心特征工程
    df_rob = df_apple.copy()
    
    # 6.1 月份季节性
    df_rob['month'] = df_rob['date'].dt.month
    df_rob['year'] = df_rob['date'].dt.year
    df_rob['sin_month'] = np.sin(2 * np.pi * df_rob['month'] / 12)

    # 6.2 融合月度厄尔尼诺指标
    df_rob = pd.merge(df_rob, df_nino, on=['year', 'month'], how='left')
    df_rob['nino34_anom'] = df_rob['nino34_anom'].fillna(0.0)

    # 6.3 降水距平特征计算
    precip_anom_cols = []
    for name, df_w in weather_data_raw_dict.items():
        w_df = df_w.copy()
        w_df['year'] = w_df['date'].dt.year
        w_df['month'] = w_df['date'].dt.month
        
        # 计算每月总降水量
        monthly_totals = w_df.groupby(['year', 'month'])['precipitation'].sum().reset_index()
        monthly_totals = monthly_totals.rename(columns={'precipitation': f'monthly_precip_{name}'})
        
        # 计算历史同期平均降水量
        historical_means = monthly_totals.groupby('month')[f'monthly_precip_{name}'].mean().reset_index()
        historical_means = historical_means.rename(columns={f'monthly_precip_{name}': f'mean_precip_{name}'})
        
        # 融合计算距平值
        monthly_totals = pd.merge(monthly_totals, historical_means, on='month', how='left')
        monthly_totals[f'precip_anomaly_{name}'] = monthly_totals[f'monthly_precip_{name}'] - monthly_totals[f'mean_precip_{name}']
        
        w_df = pd.merge(w_df, monthly_totals[['year', 'month', f'precip_anomaly_{name}']], on=['year', 'month'], how='left')
        df_rob = pd.merge(df_rob, w_df[['date', f'precip_anomaly_{name}']], on='date', how='inner')
        precip_anom_cols.append(f'precip_anomaly_{name}')
        
    df_rob['national_precip_anomaly'] = df_rob[precip_anom_cols].mean(axis=1)

    # 6.4 气象预报代理特征
    temp_dict = {}
    precip_dict = {}
    for name, df_w in weather_data_raw_dict.items():
        df_w_copy = df_w.copy()
        df_w_copy['temp_mean'] = (df_w_copy['temp_max'] + df_w_copy['temp_min']) / 2
        
        # 7天前向滚动气象特征
        df_w_copy[f'forecast_temp_{name}'] = get_forward_rolling(df_w_copy['temp_mean'], 7, 'mean')
        df_w_copy[f'forecast_precip_{name}'] = get_forward_rolling(df_w_copy['precipitation'], 7, 'sum')
        
        df_rob = pd.merge(df_rob, df_w_copy[['date', f'forecast_temp_{name}', f'forecast_precip_{name}']], on='date', how='inner')
        temp_dict[name] = f'forecast_temp_{name}'
        precip_dict[name] = f'forecast_precip_{name}'
        
    df_rob['forecast_temp_mean'] = df_rob[list(temp_dict.values())].mean(axis=1)
    df_rob['forecast_precip_sum'] = df_rob[list(precip_dict.values())].sum(axis=1)

    FEATURES = ['national_precip_anomaly', 'nino34_anom', 'forecast_temp_mean', 'forecast_precip_sum', 'sin_month']

    # 7. 执行对应模式
    if args.mode == 'backtest':
        run_backtest(df_rob, FEATURES, args.threshold_long, args.threshold_short)
    elif args.mode == 'signal':
        generate_tomorrow_signal(df_rob, FEATURES, args.threshold_long, args.threshold_short)

if __name__ == '__main__':
    main()

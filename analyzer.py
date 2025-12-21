# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import matplotlib

# 強制使用 Agg 後端以確保穩定性
matplotlib.use('Agg')

# 字體設定
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Noto Sans CJK JP', 'Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 基礎分箱設定
BIN_SIZE = 10.0
X_MIN, X_MAX = -100, 100
BINS = np.arange(X_MIN, X_MAX + 1, BIN_SIZE)

def build_company_list(arr_pct, codes, names, bins):
    """產出 HTML 格式的分箱清單，解鎖 >100% 標的並排序"""
    lines = [f"{'報酬區間':<12} | {'家數(比例)':<14} | 公司清單", "-"*80]
    total = len(arr_pct)
    
    for lo in range(int(X_MIN), int(X_MAX), int(BIN_SIZE)):
        up = lo + 10
        lab = f"{lo}%~{up}%"
        mask = (arr_pct >= lo) & (arr_pct < up)
        cnt = int(mask.sum())
        if cnt == 0: continue
        
        picked_indices = np.where(mask)[0]
        links = [f'{codes[i]}({names[i]})' for i in picked_indices]
        lines.append(f"{lab:<12} | {cnt:>4} ({(cnt/total*100):5.1f}%) | {', '.join(links)}")

    # 解鎖天花板 (大於 100%)
    extreme_mask = (arr_pct >= 100)
    e_cnt = int(extreme_mask.sum())
    if e_cnt > 0:
        e_picked = np.where(extreme_mask)[0]
        sorted_e = sorted(e_picked, key=lambda i: arr_pct[i], reverse=True)
        # HTML 清單中使用紅色粗體顯示
        e_links = [f'<b style="color:red;">{codes[i]}({names[i]}:{arr_pct[i]:.0f}%)</b>' for i in sorted_e]
        lines.append(f"{' > 100%':<12} | {e_cnt:>4} ({(e_cnt/total*100):5.1f}%) | {', '.join(e_links)}")

    return "\n".join(lines)

def run_global_analysis(market_id="tw-share"):
    market_label = market_id.upper()
    print(f"📊 正在啟動 {market_label} 深度矩陣分析...")
    
    data_path = Path("./data") / market_id / "dayK"
    image_out_dir = Path("./output/images") / market_id
    image_out_dir.mkdir(parents=True, exist_ok=True)
    
    all_files = list(data_path.glob("*.csv"))
    if not all_files: return [], pd.DataFrame(), {}

    results = []
    for f in tqdm(all_files, desc=f"分析 {market_label} 數據"):
        try:
            df = pd.read_csv(f)
            if len(df) < 20: continue
            df.columns = [c.lower() for c in df.columns]
            close, high, low = df['close'].values, df['high'].values, df['low'].values
            
            tkr, nm = f.stem.split('_', 1) if '_' in f.stem else (f.stem, f.stem)
            row = {'Ticker': tkr, 'Full_ID': nm}
            
            periods = [('Week', 5), ('Month', 20), ('Year', 250)]
            for p_name, days in periods:
                if len(close) <= days: continue
                prev_c = close[-(days+1)]
                if prev_c <= 0: continue
                row[f'{p_name}_High'] = (max(high[-days:]) - prev_c) / prev_c * 100
                row[f'{p_name}_Close'] = (close[-1] - prev_c) / prev_c * 100
                row[f'{p_name}_Low'] = (min(low[-days:]) - prev_c) / prev_c * 100
            results.append(row)
        except: continue

    df_res = pd.DataFrame(results)
    images = []
    # 基礎配色
    color_map = {'High': '#28a745', 'Close': '#007bff', 'Low': '#dc3545'}
    # 溢出區間配色 (橘紅色)
    EXTREME_COLOR = '#FF4500' 

    plot_bins = np.append(BINS, X_MAX + BIN_SIZE)

    for p_n, p_z in [('Week', '週'), ('Month', '月'), ('Year', '年')]:
        for t_n, t_z in [('High', '最高-進攻'), ('Close', '收盤-實質'), ('Low', '最低-防禦')]:
            col = f"{p_n}_{t_n}"
            if col not in df_res.columns: continue
            data = df_res[col].dropna()
            
            fig, ax = plt.subplots(figsize=(12, 7))
            clipped_data = np.clip(data.values, X_MIN, X_MAX + BIN_SIZE)
            counts, edges = np.histogram(clipped_data, bins=plot_bins)
            
            # 分開畫：一般柱子與最後一根特殊柱子
            normal_counts = counts[:-1]
            extreme_count = counts[-1]
            
            # 畫一般區間
            bars = ax.bar(edges[:-2], normal_counts, width=9, align='edge', 
                          color=color_map[t_n], alpha=0.7, edgecolor='white')
            
            # 畫 >100% 區間 (顏色加深，加上邊框)
            ex_bar = ax.bar(edges[-2], extreme_count, width=9, align='edge', 
                            color=EXTREME_COLOR, alpha=0.9, edgecolor='black', linewidth=1.5)
            
            # 合併所有 bars 進行文字標註
            all_bars = list(bars) + list(ex_bar)
            max_h = counts.max() if len(counts) > 0 else 1
            
            for i, bar in enumerate(all_bars):
                h = bar.get_height()
                if h > 0:
                    # 如果是最後一根柱子，文字標籤變紅加粗
                    is_extreme = (i == len(all_bars) - 1)
                    text_color = 'red' if is_extreme else 'black'
                    text_weight = 'bold' if is_extreme else 'bold'
                    
                    ax.text(bar.get_x() + 4.5, h + (max_h * 0.02), f'{int(h)}\n({h/len(data)*100:.1f}%)', 
                            ha='center', va='bottom', fontsize=10, 
                            fontweight=text_weight, color=text_color)

            ax.set_ylim(0, max_h * 1.4) 
            ax.set_title(f"【{market_label}】{p_z}K {t_z} 報酬分布 (樣本:{len(data)})", fontsize=18, fontweight='bold')
            
            ax.set_xticks(plot_bins)
            x_labels = [f"{int(x)}%" for x in BINS] + [f">{int(X_MAX)}%"]
            ax.set_xticklabels(x_labels, rotation=45)
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            
            img_path = image_out_dir / f"{col.lower()}.png"
            plt.savefig(img_path, dpi=120)
            plt.close()
            images.append({'id': col.lower(), 'path': str(img_path), 'label': f"【{market_label}】{p_z}K {t_z}"})

    text_reports = {}
    for p_n in ['Week', 'Month', 'Year']:
        col = f'{p_n}_High'
        if col in df_res.columns:
            text_reports[p_n] = build_company_list(df_res[col].values, df_res['Ticker'].tolist(), df_res['Full_ID'].tolist(), BINS)
    
    return images, df_res, text_reports

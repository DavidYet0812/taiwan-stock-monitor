# -*- coding: utf-8 -*-
import os
import resend
from datetime import datetime

def send_stock_report(market_name, img_data, report_df, text_reports):
    """
    發送包含分布圖與智慧技術線圖連結的專業電子郵件
    支援市場：台灣 (TW), 美國 (US), 香港 (HK), 中國 (CN), 日本 (JP), 韓國 (KR)
    """
    # 1. 檢查 API Key
    api_key = os.environ.get("RESEND_API_KEY")
    if not api_key:
        print("❌ 錯誤：找不到環境變數 RESEND_API_KEY，郵件發送中斷。")
        return
    resend.api_key = api_key

    now_str = datetime.now().strftime("%Y-%m-%d")
    
    # 2. 判斷市場屬性（智慧識別六大市場）
    market_upper = market_name.upper()
    is_us = "美國" in market_upper or "US" in market_upper
    is_tw = "台灣" in market_upper or "TW" in market_upper
    #is_hk = "香港" in market_upper or "HK" in market_upper
    #is_cn = "中國" in market_upper or "CN" in market_upper
    #is_jp = "日本" in market_upper or "JP" in market_upper
    #is_kr = "韓國" in market_upper or "KR" in market_upper

    # 3. 建立 Top 50 連結區塊邏輯
    def get_top50_links(df, col_name):
        if col_name not in df.columns:
            return "目前無數據"
        
        top50 = df.sort_values(by=col_name, ascending=False).head(50)
        links = []
        
        for _, r in top50.iterrows():
            ticker = str(r["Ticker"])
            
            # --- 智慧連結判定 ---
            if is_us:
                # 🇺🇸 美國：StockCharts
                #url = f"https://stockcharts.com/sc3/ui/?s={ticker}"
            elif is_tw:
                # 🇹🇼 台灣：玩股網
                clean_tkr = ticker.split('.')[0]
                url = f"https://www.wantgoo.com/stock/{clean_tkr}/technical-chart"
            else:
                # 預設跳轉（台股模式）
                clean_tkr = ticker.split('.')[0]
                url = f"https://www.wantgoo.com/stock/{clean_tkr}/technical-chart"
            #elif is_hk:
                # 🇭🇰 香港：AASTOCKS
                #clean_code = ticker.replace(".HK", "").strip().zfill(5)
                #url = f"https://www.aastocks.com/tc/stocks/quote/quick-quote.aspx?symbol={clean_code}"
            #elif is_cn:
                # 🇨🇳 中國 A 股：東方財富
                #prefix = "sh" if ticker.startswith('6') else "sz"
                #url = f"https://quote.eastmoney.com/{prefix}{ticker}.html"
            #elif is_jp:
                # 🇯🇵 日本：樂天證券 (需確保 .T 後綴)
                #clean_ticker = ticker if ".T" in ticker.upper() else f"{ticker.split('.')[0]}.T"
                #url = f"https://www.rakuten-sec.co.jp/web/market/search/quote.html?ric={clean_ticker}"
            #elif is_kr:
                # 🇰🇷 韓國：Naver Finance (僅需代號數字)
                #clean_code = ticker.split('.')[0]
                #url = f"https://finance.naver.com/item/main.naver?code={clean_code}"
            
            
            display_name = r.get("Full_Name", ticker)
            links.append(f'<a href="{url}" style="text-decoration:none; color:#0366d6;">{ticker}({display_name})</a>')
        
        return " | ".join(links)

    # 4. 組合 HTML 郵件內容
    # 動態決定提示文字中的網站名稱
    target_site = 'StockCharts' if is_us else 'AASTOCKS' if is_tw else '玩股網'
    
    html_content = f"""
    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #333; max-width: 850px; margin: auto; border: 1px solid #eee; padding: 20px; border-radius: 10px;">
        <h2 style="color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px;">
            🚀 {market_name} 全方位市場監控報表
        </h2>
        <p style="color: #7f8c8d; font-size: 14px;">報告生成時間: {now_str}</p>
        
        <div style="background-color: #fdfefe; border-left: 5px solid #e74c3c; padding: 10px; margin: 20px 0; font-size: 14px;">
            💡 提示：點擊下方表格中的<b>股票代號</b>，可直接跳轉至 <b>{target_site}</b> 查看即時技術線圖。
        </div>
    """
    
    # 插入 9 張分布圖
    for img in img_data:
        html_content += f"<h3 style='color: #2980b9; margin-top: 30px;'>📍 {img['label']}</h3>"
        html_content += f'<img src="cid:{img["id"]}" style="width:100%; max-width:800px; border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">'

    # 插入分箱清單文字
    html_content += "<div style='background-color: #f4f7f6; padding: 15px; border-radius: 8px; margin-top: 40px;'>"
    for period, report in text_reports.items():
        p_name = {"Week": "週", "Month": "月", "Year": "年"}.get(period, period)
        html_content += f"<h4 style='color: #16a085; margin-bottom: 5px;'>📊 {p_name}K 報酬分布明細 (含飆股清單)</h4>"
        html_content += f"<pre style='background-color: #ffffff; padding: 10px; border: 1px solid #ddd; font-size: 12px; white-space: pre-wrap; word-wrap: break-word;'>{report}</pre>"
    html_content += "</div>"

    # 插入 Top 50 飆股區塊
    html_content += f"""
        <hr style="border: 0; border-top: 1px solid #eee; margin: 40px 0;">
        <h4 style="color: #c0392b;">🔥 本週表現最強動能 Top 50 (點擊跳轉線圖)</h4>
        <div style="line-height: 2; font-size: 13px; color: #34495e;">
            {get_top50_links(report_df, 'Week_High')}
        </div>
        <p style="margin-top: 50px; font-size: 12px; color: #bdc3c7; text-align: center;">
            此報表由阿杰製作AI自動分析生成，僅供研究參考。
        </p>
    </div>
    """

    # 5. 準備附件 (Inline Embedding)
    attachments = []
    for img in img_data:
        try:
            with open(img['path'], "rb") as f:
                attachments.append({
                    "content": list(f.read()),
                    "filename": f"{img['id']}.png",
                    "content_id": img['id'],
                    "disposition": "inline"
                })
        except Exception as e:
            print(f"⚠️ 讀取圖片失敗 {img['path']}: {e}")

    # 6. 執行寄送
    to_emails = ["s714139@gmail.com"]
    
    try:
        resend.Emails.send({
            "from": "StockMonitor <onboarding@resend.dev>",
            "to": to_emails,
            "subject": f"🚀 {market_name} 監控報告 - {now_str}",
            "html": html_content,
            "attachments": attachments
        })
        print(f"✅ 郵件發送成功！市場：{market_name}")
    except Exception as e:
        print(f"❌ 郵件發送失敗 ({market_name}): {e}")








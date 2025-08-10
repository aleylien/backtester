import os
import pandas as pd
from datetime import datetime

def _safe_read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def export_summary_xlsx(run_out: str) -> str:
    """
    Create summary.xlsx with:
      - Sheet 'Overview': Instruments table (readable headers, % formats, filter/freeze) + separate Portfolio table (with sparkline)
      - Sheet 'Charts'  : Portfolio equity, 30-bar return dist, and per-strategy equity charts (if present)
      - Sheet 'Correlations': Asset & Strategy correlation CSVs (if present)
      - Sheet 'Run Info': run path, timestamp, detected instruments/strategies, and file presence
    """

    # ---------- helpers ----------
    def _safe_read_csv(path):
        return pd.read_csv(path) if os.path.exists(path) else None

    def _round_numeric(df: pd.DataFrame, ndigits: int = 4) -> pd.DataFrame:
        df = df.copy()
        num_cols = df.select_dtypes(include=["number"]).columns
        if len(num_cols):
            df[num_cols] = df[num_cols].round(ndigits)
        return df

    def _extract_name_col(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure there's a 'name' column (from CSV index if needed)."""
        out = df.copy()
        if "name" in out.columns:
            return out
        # common case: index saved to CSV as first 'Unnamed: 0'
        first = out.columns[0]
        if first.lower().startswith("unnamed"):
            out = out.rename(columns={first: "name"})
        return out

    def _write_bestworst_sheet(writer: pd.ExcelWriter):
        sheet = "BestWorst"
        # create the sheet
        pd.DataFrame({"init": []}).to_excel(writer, sheet_name=sheet, index=False)
        row = 0
        def _title(t):
            nonlocal row
            pd.DataFrame({"Section": [t]}).to_excel(writer, sheet_name=sheet, index=False, startrow=row)
            row += 2
        def _write_df(df):
            nonlocal row
            if df is None or len(df) == 0:
                pd.DataFrame({"info": ["no data"]}).to_excel(writer, sheet_name=sheet, index=False, startrow=row)
                row += 2
            else:
                df.to_excel(writer, sheet_name=sheet, index=False, startrow=row)
                row += (len(df) + 2)

        # ---- Portfolio ----
        port_dir = os.path.join(run_out, "portfolio")
        top_p   = _safe_read_csv(os.path.join(port_dir, "top_periods.csv"))
        bot_p   = _safe_read_csv(os.path.join(port_dir, "bottom_periods.csv"))
        _title("Portfolio — Top 5")
        _write_df(top_p)
        _title("Portfolio — Bottom 5")
        _write_df(bot_p)

        # ---- Strategies ----
        strat_root = os.path.join(run_out, "strategies")
        if os.path.isdir(strat_root):
            for strat_name in sorted(os.listdir(strat_root)):
                sdir = os.path.join(strat_root, strat_name)
                if not os.path.isdir(sdir):
                    continue
                top_s = _safe_read_csv(os.path.join(sdir, "top_periods.csv"))
                bot_s = _safe_read_csv(os.path.join(sdir, "bottom_periods.csv"))
                _title(f"{strat_name} — Top 5")
                _write_df(top_s)
                _title(f"{strat_name} — Bottom 5")
                _write_df(bot_s)

    def _apply_common_formats(ws, headers, *, fmt_pct2, fmt_num4, fmt_num2):
        cidx = {h: i for i, h in enumerate(headers)}
        pct_cols = ["CAGR", "Ann.Vol", "Max DD", "Avg DD", "Win Rate",
                    "Avg 30d", "Avg 30d +2σ", "Avg 30d -2σ", "Avg 30d CI low", "Avg 30d CI high"
                    , "Ann. Ret. Plain", "Ann. Ret. Log"]
        for h in pct_cols:
            if h in cidx:
                ws.set_column(cidx[h], cidx[h], 12, fmt_pct2)
        for h in ["Sharpe", "Sortino"]:
            if h in cidx:
                ws.set_column(cidx[h], cidx[h], 10, fmt_num4)
        for h in ["Avg DD Dur", "Profit Factor", "Expectancy"]:
            if h in cidx:
                ws.set_column(cidx[h], cidx[h], 12, fmt_num2)

    # rename map for Overview/combined
    nice = {
        "name": "Name",
        "cagr": "CAGR",
        'total_return': "ROE",
        'annualised_return_plain': "Ann. Ret. Plain",
        'annualised_return_log': "Ann. Ret. Log",
        "annual_vol": "Ann.Vol",
        "sharpe": "Sharpe",
        "sortino": "Sortino",
        "max_drawdown": "Max DD",
        "avg_drawdown": "Avg DD",
        "avg_dd_duration": "Avg DD Dur",
        "profit_factor": "Profit Factor",
        "expectancy": "Expectancy",
        "win_rate": "Win Rate",
        "std_daily": "Daily Std",
        "ret_5pct": "5% Tail",
        "ret_95pct": "95% Tail",
        "avg_win": "Avg Win",
        "avg_loss": "Avg Loss",
        "max_loss_pct": "Max Loss (Bar)",
        "avg_30d_ret": "Avg 30d",
        "avg_30d_ret_plus_2std": "Avg 30d +2σ",
        "avg_30d_ret_minus_2std": "Avg 30d -2σ",
        "avg_30d_ret_ci_low": "Avg 30d CI low",
        "avg_30d_ret_ci_high": "Avg 30d CI high",
        "avg_cost_pct": "Avg Cost",
        "cost_sharp": "SR Cost",
    }

    # columns we want to show as percentages (remain numeric; formatted via Excel)
    pct_cols_overview = [
        "CAGR", "Ann.Vol", "Max DD", "Avg DD", "5% Tail", "95% Tail",
        "Avg Win", "Avg Loss", "Max Loss (Bar)", "Avg 30d", "Avg 30d +2σ",
        "Avg 30d -2σ", "Avg 30d CI low", "Avg 30d CI high", "Avg Cost", "Ann. Ret. Plain", "Ann. Ret. Log", "ROE"
        # NOTE: Sharpe is unitless → not formatted as %
    ]

    # Strategy section labels
    # Display names for per-strategy stats (keys must match stats.csv column names)
    strat_label_map = {
        "cagr": "CAGR",
        'total_return': "ROE",
        'annualised_return_plain': "Ann. Ret. Plain",
        'annualised_return_log': "Ann. Ret. Log",
        "annual_vol": "Ann.Vol",
        "sharpe": "Sharpe",
        "sortino": "Sortino",
        "max_drawdown": "Max DD",
        "avg_drawdown": "Avg DD",
        "avg_dd_duration": "Avg DD Dur",
        "profit_factor": "Profit Factor",
        "expectancy": "Expectancy",
        "win_rate": "Win Rate",
        "std_daily": "Daily Std",
        "ret_5pct": "5% Tail",
        "ret_95pct": "95% Tail",
        "avg_win": "Avg Win",
        "avg_loss": "Avg Loss",
        "max_loss_pct": "Max Loss (Bar)",
        "avg_30d_ret": "Avg 30d",
        "avg_30d_ret_plus_2std": "Avg 30d +2σ",
        "avg_30d_ret_minus_2std": "Avg 30d -2σ",
        "avg_30d_ret_ci_low": "Avg 30d CI low",
        "avg_30d_ret_ci_high": "Avg 30d CI high",
    }

    # Paths we'll use
    comb_path = os.path.join(run_out, "combined_stats.csv")

    xlsx_path = os.path.join(run_out, "summary.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        workbook = writer.book

        # Common formats
        fmt_pct2   = workbook.add_format({"num_format": "0.00%"})
        fmt_num4   = workbook.add_format({"num_format": "0.0000"})
        fmt_bold   = workbook.add_format({"bold": True})
        fmt_port   = workbook.add_format({"bold": True, "bg_color": "#FFF2CC"})
        fmt_title  = workbook.add_format({"bold": True, "font_size": 12})
        fmt_wrap   = workbook.add_format({"text_wrap": True})

        # ---------- Sheet 1: Overview ----------
        sh = "Overview"
        ws_over = workbook.add_worksheet(sh)
        writer.sheets[sh] = ws_over
        start_row = 0

        # Load combined and prepare Instruments + Portfolio tables
        comb = _safe_read_csv(comb_path)
        if comb is not None and not comb.empty:
            comb = _extract_name_col(comb)

            # We'll keep everything numeric; apply display names
            comb_disp = comb.rename(columns=nice)

            # Split into instruments and portfolio
            name_col = "Name" if "Name" in comb_disp.columns else None
            if name_col:
                instr = comb_disp[comb_disp[name_col].astype(str).str.lower() != "portfolio"].copy()
                port  = comb_disp[comb_disp[name_col].astype(str).str.lower() == "portfolio"].copy()
            else:
                instr = comb_disp.copy()
                port  = pd.DataFrame(columns=comb_disp.columns)

            # ---- Instruments table ----
            if not instr.empty:
                ws_over.write(start_row, 0, "Instruments", fmt_title)
                start_row += 1
                instr_rounded = _round_numeric(instr)
                instr_rounded.to_excel(writer, sheet_name=sh, startrow=start_row, startcol=0, index=False)
                nrows, ncols = instr_rounded.shape

                # Formats: set percentage columns
                headers = list(instr_rounded.columns)
                col_idx = {h: i for i, h in enumerate(headers)}
                for c in pct_cols_overview:
                    if c in col_idx:
                        j = col_idx[c]
                        ws_over.set_column(j, j, 12, fmt_pct2)

                # Set widths & number formats
                # keep Sharpe with numeric 4-decimal
                if "Sharpe" in col_idx:
                    j = col_idx["Sharpe"]
                    ws_over.set_column(j, j, 10, fmt_num4)

                # Freeze header row of this table and add filter
                ws_over.autofilter(start_row, 0, start_row + nrows, max(0, ncols - 1))
                ws_over.freeze_panes(start_row + 1, 0)

                start_row += nrows + 2  # gap after instruments table

            # ---- Portfolio table ----
            if not port.empty:
                ws_over.write(start_row, 0, "Portfolio", fmt_title)
                start_row += 1

                port_rounded = _round_numeric(port)
                port_rounded.to_excel(writer, sheet_name=sh, startrow=start_row, startcol=0, index=False)
                _apply_common_formats(ws_over, list(port_rounded.columns),
                                      fmt_pct2=fmt_pct2, fmt_num4=fmt_num4, fmt_num2=fmt_num4)

                prow, pcols = port_rounded.shape
                # percentage formats on portfolio row
                pheaders = list(port_rounded.columns)
                pidx = {h: i for i, h in enumerate(pheaders)}
                for c in pct_cols_overview:
                    if c in pidx:
                        j = pidx[c]
                        ws_over.set_column(j, j, 12, fmt_pct2)
                if "Sharpe" in pidx:
                    ws_over.set_column(pidx["Sharpe"], pidx["Sharpe"], 10, fmt_num4)

                # Highlight portfolio row WITHOUT breaking number formats:
                # apply a conditional format that only sets fill/bold, leaving num_format to column formats
                pheaders = list(port_rounded.columns)
                pcols = len(pheaders)
                ws_over.conditional_format(
                    start_row + 1, 0,  # first cell of the data row (below header)
                    start_row + 1, max(0, pcols - 1),  # last cell in that row
                    {
                        "type": "no_blanks",
                        "format": fmt_port,  # fmt_port should NOT have a num_format property
                    }
                )

                start_row += prow + 2

        # ---------- Per-asset tests ----------
        rows = []
        for d in sorted(os.listdir(run_out)):
            sub = os.path.join(run_out, d)
            if not os.path.isdir(sub) or d in ("portfolio", "strategies"):
                continue
            row = {"Instrument": d}
            p1 = _safe_read_csv(os.path.join(sub, "permutation_test_oos.csv"))
            p2 = _safe_read_csv(os.path.join(sub, "permutation_test_training.csv"))
            pr = _safe_read_csv(os.path.join(sub, "partition_return.csv"))
            if p1 is not None and not p1.empty:
                row["Test1_p"] = p1.iloc[0, 0]
            if p2 is not None and not p2.empty:
                row["Test2_p"] = p2.iloc[0, 0]
            if pr is not None and not pr.empty:
                for c in pr.columns:
                    row[f"PR_{c}"] = pr.iloc[0][c]
            if len(row) > 1:
                rows.append(row)
        if rows:
            df_tests = pd.DataFrame(rows)
            df_tests = _round_numeric(df_tests)
            ws_over.write(start_row, 0, "Per-Asset Tests", fmt_title)
            start_row += 1
            df_tests.to_excel(writer, sheet_name=sh, startrow=start_row, index=False)
            start_row += len(df_tests) + 2

        # ---------- Multiple-system selection bias (if present) ----------
        multi = _safe_read_csv(os.path.join(run_out, "permutation_test_multiple.csv"))
        if multi is not None:
            multi = _round_numeric(multi)
            ws_over.write(start_row, 0, "Multiple-System Selection Bias", fmt_title)
            start_row += 1
            multi.to_excel(writer, sheet_name=sh, startrow=start_row, index=False)
            start_row += len(multi) + 2

        # ---------- Strategy Stats section ----------
        strat_dir = os.path.join(run_out, "strategies")
        strat_rows = []
        if os.path.isdir(strat_dir):
            for strat in sorted(os.listdir(strat_dir)):
                sdir = os.path.join(strat_dir, strat)
                if not os.path.isdir(sdir):
                    continue
                fstats = os.path.join(sdir, "stats.csv")
                if not os.path.exists(fstats):
                    continue
                df = pd.read_csv(fstats)
                if df is None or df.empty:
                    continue

                # Build one row per strategy with display labels
                rec = {"Strategy": strat}
                # keep a stable order based on strat_label_map declaration
                for raw_col, disp in strat_label_map.items():
                    if raw_col in df.columns:
                        rec[disp] = df.iloc[0][raw_col]
                strat_rows.append(rec)

        if strat_rows:
            # Make dataframe with the exact order: Strategy first, then mapped columns in mapping order
            ordered_cols = ["Strategy"] + [v for v in strat_label_map.values() if any(v in r for r in strat_rows)]
            df_strat = pd.DataFrame(strat_rows)
            # keep only columns that exist (in case some metrics were missing)
            df_strat = df_strat[[c for c in ordered_cols if c in df_strat.columns]]
            df_strat = _round_numeric(df_strat)

            ws_over.write(start_row, 0, "Strategy Stats", fmt_title)
            start_row += 1
            df_strat.to_excel(writer, sheet_name=sh, startrow=start_row, index=False)

            # ---- Column formatting ----
            headers2 = list(df_strat.columns)
            cidx2 = {h: i for i, h in enumerate(headers2)}

            # Percent-style columns
            pct_cols_strat = [
                "CAGR", "Ann. Ret. Plain", "Ann. Ret. Log", "Ann.Vol", "Max DD", "Avg DD", "Daily Std",
                "5% Tail", "95% Tail", "Avg Win", "Avg Loss", "Max Loss (Bar)",
                "Avg 30d", "Avg 30d +2σ", "Avg 30d -2σ", "Avg 30d CI low", "Avg 30d CI high",
                "Win Rate", "ROE"

            ]
            for h in pct_cols_strat:
                if h in cidx2:
                    ws_over.set_column(cidx2[h], cidx2[h], 12, fmt_pct2)

            # Numeric columns
            if "Sharpe" in cidx2:
                ws_over.set_column(cidx2["Sharpe"], cidx2["Sharpe"], 10, fmt_num4)
            if "Sortino" in cidx2:
                ws_over.set_column(cidx2["Sortino"], cidx2["Sortino"], 10, fmt_num4)
            if "Profit Factor" in cidx2:
                ws_over.set_column(cidx2["Profit Factor"], cidx2["Profit Factor"], 12, fmt_num4)
            if "Expectancy" in cidx2:
                # likely currency-like; keep as plain numeric with 2 decimals
                ws_over.set_column(cidx2["Expectancy"], cidx2["Expectancy"], 14, fmt_num4)

            start_row += len(df_strat) + 2

        # ---------- Sheet 2: Charts ----------
        ws_charts = workbook.add_worksheet("Charts")
        writer.sheets["Charts"] = ws_charts

        # Helper: how many worksheet rows a PNG will occupy (after scaling)
        def _rows_needed_for_image(img_path: str, y_scale: float) -> int:
            # Excel default row height ≈ 15 points ≈ 20 px at 100%
            ROW_PX = 20.0
            try:
                from PIL import Image  # optional; if missing we fallback below
                with Image.open(img_path) as im:
                    h_px = float(im.height)
            except Exception:
                # Fallback guess if Pillow not available
                h_px = 720.0
            return int((h_px * y_scale) / ROW_PX) + 3  # +3 rows padding/title

        # Use a mild downscale so charts fit nicely; tweak if you like
        X_SCALE = 0.85
        Y_SCALE = 0.85

        row = 0
        images = [
            ("Portfolio Equity", os.path.join(run_out, "portfolio", "portfolio_equity_rebased.png")),
            ("30-Bar Return Dist", os.path.join(run_out, "portfolio_30bar_return_distribution.png")),
            ("Drawdown Dist", os.path.join(run_out, "portfolio", "drawdown_distribution.png")),
            ("Drawdown Duration vs. Magnitude", os.path.join(run_out, "portfolio", "dd_duration_vs_magnitude.png")),
        ]
        for title, img in images:
            if os.path.exists(img):
                ws_charts.write(row, 0, title, fmt_bold)
                ws_charts.insert_image(row + 1, 0, img, {"x_scale": X_SCALE, "y_scale": Y_SCALE})
                row += _rows_needed_for_image(img, Y_SCALE)

        # Per-strategy equity charts (with titles; each spaced by actual height)
        strat_dir = os.path.join(run_out, "strategies")
        if os.path.isdir(strat_dir):
            for strat in sorted(os.listdir(strat_dir)):
                sdir = os.path.join(strat_dir, strat)
                if not os.path.isdir(sdir):
                    continue
                img = os.path.join(sdir, "portfolio_equity_rebased.png")
                if os.path.exists(img):
                    ws_charts.write(row, 0, f"{strat} Equity", fmt_bold)
                    ws_charts.insert_image(row + 1, 0, img, {"x_scale": X_SCALE, "y_scale": Y_SCALE})
                    row += _rows_needed_for_image(img, Y_SCALE)

        # ---------- Sheet 3: Correlations ----------
        ws_corr = workbook.add_worksheet("Correlations")
        writer.sheets["Correlations"] = ws_corr
        start = 0
        a_corr = _safe_read_csv(os.path.join(run_out, "asset_correlation.csv"))
        s_corr = _safe_read_csv(os.path.join(run_out, "strategy_correlation.csv"))
        if a_corr is not None:
            a_corr = _round_numeric(a_corr, 2)
            ws_corr.write(start, 0, "Asset Correlation", fmt_title)
            start += 1
            a_corr.to_excel(writer, sheet_name="Correlations", startrow=start, index=False)
            start += len(a_corr) + 2
        if s_corr is not None:
            s_corr = _round_numeric(s_corr, 2)
            ws_corr.write(start, 0, "Strategy Correlation", fmt_title)
            start += 1
            s_corr.to_excel(writer, sheet_name="Correlations", startrow=start, index=False)
            start += len(s_corr) + 2

        # ---------- Sheet 4: Best/Worst ----------
        try:
            _write_bestworst_sheet(writer)
        except Exception as e:
            # Don't fail the whole export if Best/Worst reading had an issue
            pass

        # ---------- Sheet 5: Run Info ----------
        ws_info = workbook.add_worksheet("Run Info")
        writer.sheets["Run Info"] = ws_info
        info = []
        info.append(["Run folder", run_out])
        info.append(["Generated on", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        # instruments (from combined)
        try:
            if comb is not None and not comb.empty:
                comb_names = _extract_name_col(comb)
                names = comb_names.iloc[:, 0].astype(str).tolist()
                instr_names = [n for n in names if n.lower() != "portfolio"]
                info.append(["Instruments (#)", len(instr_names)])
                info.append(["Instruments", ", ".join(instr_names)])
        except Exception:
            pass
        # strategies detected
        found_strats = []
        if os.path.isdir(strat_dir):
            for strat in sorted(os.listdir(strat_dir)):
                if os.path.isdir(os.path.join(strat_dir, strat)):
                    found_strats.append(str(strat))
        info.append(["Strategies (#)", len(found_strats)])
        info.append(["Strategies", ", ".join(found_strats) if found_strats else "—"])
        # files
        info.append(["Has portfolio_equity.png", "Yes" if os.path.exists(os.path.join(run_out, "portfolio", "portfolio_equity_rebased.png")) else "No"])

        for r, (k, v) in enumerate(info):
            ws_info.write(r, 0, k, fmt_bold)
            ws_info.write(r, 1, v)

    return xlsx_path


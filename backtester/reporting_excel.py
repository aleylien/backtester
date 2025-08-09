import os
import pandas as pd


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
    import os
    import pandas as pd
    import numpy as np
    from datetime import datetime

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

    # rename map for Overview/combined
    nice = {
        "name": "Name",
        "cagr": "CAGR",
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
    }

    # columns we want to show as percentages (remain numeric; formatted via Excel)
    pct_cols_overview = [
        "CAGR", "Ann.Vol", "Max DD", "Avg DD", "5% Tail", "95% Tail",
        "Avg Win", "Avg Loss", "Max Loss (Bar)", "Avg 30d", "Avg 30d +2σ",
        "Avg 30d -2σ", "Avg 30d CI low", "Avg 30d CI high", "Avg Cost",
        # NOTE: Sharpe is unitless → not formatted as %
    ]

    # Strategy section labels
    strat_label_map = {"cagr": "CAGR", "ann_vol": "Ann.Vol", "sharpe": "Sharpe", "max_dd": "MaxDD"}

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

                # Highlight portfolio row
                ws_over.set_row(start_row + 1, None, fmt_port)  # +1 because header occupies first row of this table

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
                if os.path.exists(fstats):
                    df = pd.read_csv(fstats)
                    if not df.empty:
                        rec = {"Strategy": strat}
                        # rename inside the record
                        for k, v in strat_label_map.items():
                            if k in df.columns:
                                rec[v] = df.iloc[0][k]
                        strat_rows.append(rec)
        if strat_rows:
            df_strat = pd.DataFrame(strat_rows)
            df_strat = _round_numeric(df_strat)
            ws_over.write(start_row, 0, "Strategy Stats", fmt_title)
            start_row += 1
            df_strat.to_excel(writer, sheet_name=sh, startrow=start_row, index=False)
            # formats
            headers2 = list(df_strat.columns)
            cidx2 = {h: i for i, h in enumerate(headers2)}
            for h in ["CAGR", "Ann.Vol", "MaxDD"]:
                if h in cidx2:
                    ws_over.set_column(cidx2[h], cidx2[h], 12, fmt_pct2)
            if "Sharpe" in cidx2:
                ws_over.set_column(cidx2["Sharpe"], cidx2["Sharpe"], 10, fmt_num4)
            start_row += len(df_strat) + 2

        # ---------- Sheet 2: Charts ----------
        ws_charts = workbook.add_worksheet("Charts")
        writer.sheets["Charts"] = ws_charts
        row = 0
        images = [
            ("Portfolio Equity", os.path.join(run_out, "portfolio", "portfolio_equity.png")),
            ("30-Bar Return Dist", os.path.join(run_out, "portfolio_30bar_return_distribution.png")),
        ]
        for title, img in images:
            if os.path.exists(img):
                ws_charts.write(row, 0, title, fmt_bold)
                ws_charts.insert_image(row + 1, 0, img)
                row += 32

        # per-strategy equity charts
        if os.path.isdir(strat_dir):
            for strat in sorted(os.listdir(strat_dir)):
                sdir = os.path.join(strat_dir, strat)
                if not os.path.isdir(sdir):
                    continue
                img = os.path.join(sdir, "equity.png")
                if os.path.exists(img):
                    ws_charts.write(row, 0, f"{strat} Equity", fmt_bold)
                    ws_charts.insert_image(row + 1, 0, img)
                    row += 32

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

        # ---------- Sheet 4: Run Info ----------
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
        info.append(["Has portfolio_equity.png", "Yes" if os.path.exists(os.path.join(run_out, "portfolio", "portfolio_equity.png")) else "No"])

        for r, (k, v) in enumerate(info):
            ws_info.write(r, 0, k, fmt_bold)
            ws_info.write(r, 1, v)

    return xlsx_path


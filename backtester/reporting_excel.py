import os
import pandas as pd


def _safe_read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def export_summary_xlsx(run_out: str) -> str:
    """
    Create summary.xlsx with:
      - Sheet 'Overview': Combined Statistics + Per-Asset tests + Multiple-System Selection Bias
      - Sheet 'Charts'  : Embed portfolio equity + 30-bar return distribution PNGs
      - Sheet 'Correlations': Asset & Strategy correlation CSVs
    """
    xlsx_path = os.path.join(run_out, "summary.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        # --- Sheet 1: Overview ---
        start_row = 0
        sh = "Overview"
        comb = _safe_read_csv(os.path.join(run_out, "combined_stats.csv"))
        if comb is not None:
            comb.to_excel(writer, sheet_name=sh, startrow=start_row, index=False)
            start_row += len(comb) + 2

        # Per-asset tests (stack whatever exists per subfolder)
        rows = []
        for d in sorted(os.listdir(run_out)):
            sub = os.path.join(run_out, d)
            if not os.path.isdir(sub) or d == "portfolio":
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
            df_tests.to_excel(writer, sheet_name=sh, startrow=start_row, index=False)
            start_row += len(df_tests) + 2

        multi = _safe_read_csv(os.path.join(run_out, "permutation_test_multiple.csv"))
        if multi is not None:
            multi.to_excel(writer, sheet_name=sh, startrow=start_row, index=False)

        # --- Sheet 2: Charts ---
        workbook = writer.book
        ws = workbook.add_worksheet("Charts")
        writer.sheets["Charts"] = ws
        images = [
            ("Portfolio Equity", os.path.join(run_out, "portfolio", "portfolio_equity.png")),
            ("30-Bar Return Dist", os.path.join(run_out, "portfolio_30bar_return_distribution.png")),
        ]
        row = 0
        for title, img in images:
            if os.path.exists(img):
                ws.write(row, 0, title)
                ws.insert_image(row + 1, 0, img)
                row += 30

        # --- Sheet 3: Correlations ---
        ws3 = "Correlations"
        start = 0
        a_corr = _safe_read_csv(os.path.join(run_out, "asset_correlation.csv"))
        s_corr = _safe_read_csv(os.path.join(run_out, "strategy_correlation.csv"))
        if a_corr is not None:
            a_corr.to_excel(writer, sheet_name=ws3, startrow=start, index=False)
            start += len(a_corr) + 2
        if s_corr is not None:
            s_corr.to_excel(writer, sheet_name=ws3, startrow=start, index=False)

    return xlsx_path

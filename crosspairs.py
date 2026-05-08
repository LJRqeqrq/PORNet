def select_best_cross_pairs(
    train_lis,
    train_text,
    train_doc,
    train_num,
    train_pacs,
    train_op_doc,
    train_y,
    val_lis,
    val_text,
    val_doc,
    val_num,
    val_pacs,
    val_op_doc,
    val_y,
    top_k=10,
    allow_query=None,
    allow_kv=None,
    hidden_dim=256,
    epochs=6,
    lr=2e-3,
    metric_key="Macro_F1",
    save_csv_path=None,
):
    modalities = ["lis", "text", "doctor", "num", "pacs", "operating_doctor"]

    if allow_query is None:
        allow_query = modalities
    if allow_kv is None:
        allow_kv = modalities

    train_feats = {
        "lis": train_lis,
        "text": train_text,
        "doctor": train_doc,
        "num": train_num,
        "pacs": train_pacs,
        "operating_doctor": train_op_doc,
    }
    val_feats = {
        "lis": val_lis,
        "text": val_text,
        "doctor": val_doc,
        "num": val_num,
        "pacs": val_pacs,
        "operating_doctor": val_op_doc,
    }

    num_labels = np.asarray(train_y).shape[1] if np.asarray(train_y).ndim == 2 else 1

    candidates = []
    for q in allow_query:
        for kv in allow_kv:
            if q == kv:
                continue
            candidates.append((q, kv))

    results = []
    logging.info(f"Start evaluating cross_pairs candidates: {len(candidates)} ...")

    for (q, kv) in tqdm(candidates, desc="Scoring cross_pairs"):
        try:
            score, best_t, metrics = score_one_pair(
                q_name=q,
                kv_name=kv,
                train_feats=train_feats,
                train_y=train_y,
                val_feats=val_feats,
                val_y=val_y,
                num_labels=num_labels,
                hidden_dim=hidden_dim,
                epochs=epochs,
                lr=lr,
                metric_key=metric_key,
            )
            results.append(
                {
                    "q": q,
                    "kv": kv,
                    "score": score,
                    "metric_key": metric_key,
                    "Macro_F1": metrics.get("Macro_F1", np.nan),
                    "Micro_F1": metrics.get("Micro_F1", np.nan),
                    "Macro_AUC": metrics.get("Macro_AUC", np.nan),
                    "Micro_AUC": metrics.get("Micro_AUC", np.nan),
                    "Macro_Accuracy": metrics.get("Macro_Accuracy", np.nan),
                    "Micro_Accuracy": metrics.get("Micro_Accuracy", np.nan),
                }
            )
        except Exception as e:
            logging.exception(f"pair=({q},{kv}) scoring failed: {e}")

    if len(results) == 0:
        raise RuntimeError("No pairs were scored successfully. Please check the input data.")

    df = (
        pd.DataFrame(results)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    if save_csv_path is not None:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
        logging.info(f"Saved cross_pairs scores table: {save_csv_path}")

    best_pairs = [(row["q"], row["kv"]) for _, row in df.head(top_k).iterrows()]
    logging.info(f"Top-{top_k} cross_pairs = {best_pairs}")
    return best_pairs, df

def _apply_subanalysis_filters(config, annotations_df, filters, outcome):
    """
    Apply subanalysis filters (binary outcome, cohort, histology, treatment,
    institution, etc.) to an existing filter dict based on config flags.
    Mutates and returns the filters dict.
    """
    use_cohort2  = config.get("USE_COHORT2_FILTER", False)
    use_all_mods = config.get("FILTER_ALL_MODS", False)

    # ─── binary‐outcome filter ────────────────────
    for b in ["os_months_6","os_months_24","DCR","ORR"]:
        if outcome.lower() == b.lower():
            filters[outcome] = ["1.0","0.0"]

    # ─── : apply cohort-2 if requested ────────
    if use_cohort2 and "COHORT_2" in annotations_df:
        print(" Filtering COHORT_2 == 1")
        filters["COHORT_2"] = "1"

    # ─── : apply "all modalities" filter if requested ─────────
    if use_all_mods and "HAS_ALL_MODALITIES" in annotations_df:
        print(" Filtering HAS_ALL_MODALITIES == 1")
        filters["HAS_ALL_MODALITIES"] = "1"

    # ─── : apply squamous filter if set to 0 or 1 ──────────────
    sq_flag = config.get("FILTER_SQUAMOUS", None)
    if sq_flag in {"0.0", "1.0"} and "NSCLC_HISTOLOGY_SQUAMOUS" in annotations_df:
        print(f" Filtering NSCLC_HISTOLOGY_SQUAMOUS == {sq_flag}")
        filters["NSCLC_HISTOLOGY_SQUAMOUS"] = [sq_flag]

    # ─── : apply chemo+immuno filter if set to 0 or 1 ──────────
    ci_flag = config.get("FILTER_CHEMO_IMMUNO", None)
    if ci_flag in ("0", "1") and "IO_CHT" in annotations_df:
        print(f" Filtering IO_CHT == {ci_flag}")
        filters["IO_CHT"] = [int(ci_flag)]

    # ─── : apply PDL1_GROUP filter if set to "low" or "high" ──────────
    pdl1_flag = config.get("FILTER_PDL1", None)
    if pdl1_flag in ("low", "high") and "PDL1_GROUP" in annotations_df:
        print(f" Filtering PDL1_GROUP == {pdl1_flag}")
        filters["PDL1_GROUP"] = [pdl1_flag]
    elif pdl1_flag in ("0.0", "1.0", "2.0") and "PDL1_CATEGORY" in annotations_df:
        print(f" Filtering PDL1_CATEGORY == {pdl1_flag}")
        filters["PDL1_CATEGORY"] = [pdl1_flag]

    # ─── : apply adenocarcinoma filter ──────────
    adeno_flag = config.get("ADENO", None)
    if adeno_flag in {"0.0", "1.0"} and "NSCLC_HISTOLOGY_ADENOCARCINOMA" in annotations_df:
        print(f" Filtering NSCLC_HISTOLOGY_ADENOCARCINOMA == {adeno_flag}")
        filters["NSCLC_HISTOLOGY_ADENOCARCINOMA"] = [adeno_flag]

    # ─── : apply institution FOLD filters ────────
    int_filter = config.get("FILTER_INT", None)
    if int_filter:
        print(" Filtering FILTER_INT == 1")
        filters["FOLD"] = "INT"

    vhio_filter = config.get("FILTER_VHIO", None)
    if vhio_filter:
        print(" Filtering FILTER_vhio == 1")
        filters["FOLD"] = "VHIO"

    ghd_filter = config.get("FILTER_GHD", None)
    if ghd_filter:
        print(" Filtering FILTER_GHD == 1")
        filters["FOLD"] = "GHD"

    szmc_filter = config.get("FILTER_SZMC", None)
    if szmc_filter:
        print(" Filtering FILTER_SZMC == 1")
        filters["FOLD"] = "SZMC"

    return filters


def get_datasets(P, training_type, config, fold, folds, outcome,
                 tile_px=256, tile_um=129):
    """
    Load train/val datasets, applying:
    - CV vs standard split
    - early stopping
    - binary‐outcome filter
    - optional cohort-2 / squamous / chemo-immuno filters from config
    """
    import pandas as pd

    print("\n Loading annotations from:", P.annotations)
    annotations_df = pd.read_csv(P.annotations)
    print(" Annotations shape:", annotations_df.shape)

    fold_col       = f"fold_{outcome}"
    early_stop_col = f"early_stopping_{outcome}"
    train_filter, val_filter = {}, {}

    # ─── your existing CV / standard logic ────────
    if training_type == "cross_validation":
        int_filter = config.get("FILTER_INT", None)
        if int_filter:
            train_filter['INT_ONLY_FOLDS'] = [f for f in folds if f != fold]
            val_filter['INT_ONLY_FOLDS'] = [fold]
        else:
            train_filter[fold_col] = [f for f in folds if f != fold]
            val_filter[fold_col]   = [fold]
            train_filter[f"dataset_{outcome}"] = "train"
            val_filter[f"dataset_{outcome}"] = "train"
        if early_stop_col in annotations_df:
            # Check if there are any 'yes' values for early stopping
            has_early_stop = (annotations_df[early_stop_col] == "yes").any()
            if has_early_stop:
                train_filter[early_stop_col] = "no"
                val_filter[early_stop_col]   = "yes"
            else:
                print(f" Warning: {early_stop_col} column exists but has no 'yes' values. Skipping early stopping split.")

    elif training_type == "standard":
        if early_stop_col in annotations_df:
            # Check if there are any 'yes' values for early stopping
            has_early_stop = (annotations_df[early_stop_col] == "yes").any()
            if has_early_stop:
                train_filter[f"dataset_{outcome}"] = "train"
                train_filter[early_stop_col]       = "no"
                val_filter[f"dataset_{outcome}"] = "train"
                val_filter[early_stop_col]         = "yes"
            else:
                print(f" Warning: {early_stop_col} column exists but has no 'yes' values. Using train/test split instead.")
                train_filter[f"dataset_{outcome}"] = "train"
                val_filter[f"dataset_{outcome}"]   = "test"
        else:
            train_filter[f"dataset_{outcome}"] = "train"
            val_filter[f"dataset_{outcome}"]   = "test"
    else:
        raise ValueError(f"Unknown training type: {training_type}")

    # ─── apply subanalysis filters ────────────────
    _apply_subanalysis_filters(config, annotations_df, train_filter, outcome)
    _apply_subanalysis_filters(config, annotations_df, val_filter, outcome)

    # ─── finalize & load ─────────────────────────
    print("\n FINAL train_filter:", train_filter)
    print(" FINAL val_filter:  ", val_filter)

    train_ds = P.dataset(tile_px=tile_px, tile_um=tile_um, filters=train_filter)
    val_ds   = P.dataset(tile_px=tile_px, tile_um=tile_um, filters=val_filter)

    return train_ds, val_ds


def get_eval_dataset(P, config, outcome, tile_px=256, tile_um=129):
    """
    Load evaluation dataset (ext_val), applying the same subanalysis filters
    used during training (cohort, histology, treatment, institution, etc.).
    """
    import pandas as pd

    if config.get("FILTER_INT"):
        raise ValueError(
            "Evaluation mode is not supported with FILTER_INT enabled. "
            "The INT subset does not have an external validation split."
        )

    print("\n Loading annotations from:", P.annotations)
    annotations_df = pd.read_csv(P.annotations)
    print(" Annotations shape:", annotations_df.shape)

    eval_filter = {
        f"dataset_{outcome}": "ext_val",
        f"early_stopping_{outcome}": "no",
    }

    _apply_subanalysis_filters(config, annotations_df, eval_filter, outcome)

    print("\n FINAL eval_filter:", eval_filter)

    return P.dataset(tile_px=tile_px, tile_um=tile_um, filters=eval_filter)

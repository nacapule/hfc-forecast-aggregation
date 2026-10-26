RAW_DIRECTORY = r"data/raw"
# Set the raw-data folder above. Absolute path or relative to this script
# It needs rct-a-questions-answers.csv (original CSV, not .tab) and rct-a-prediction-sets.csv
# Install numpy, pandas and matplotlib with pip, then run: python analysis.py
# Produces results.txt, daily_forecasts.csv and the memo figure decay-comparison.png
# Data and scoring: https://doi.org/10.7910/DVN/S0K6TV
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The geometric methods raise lower probabilities to this; see geometric_pool
# To reproduce the supplement's sensitivity check, rerun with 0.0001 and 0.01
EPSILON = 0.001
TIMEZONE = "America/New_York"
# HFC scores the last forecast before 2:01 pm eastern
CUTOFF = pd.Timedelta(hours=14, seconds=59)
METHODS = ["mean", "median", "geometric_mean", "trimmed_mean", "geometric_odds"]
GEOMETRIC_METHODS = ["geometric_mean", "geometric_odds"]
# The improvement weights these by forecast age; the arithmetic mean is a comparison
DECAY_METHODS = ["mean"] + GEOMETRIC_METHODS
# Half-lives in days: 0.01 is 14.4 minutes and 1 / 24 is one hour
# ln(2) / ln(10) = 0.301 days leaves a forecast one tenth of its weight per day
HALF_LIVES = [.01, 1 / 24, .1, .25, np.log(2) / np.log(10), .5, 1., 3., 7., 14., 30., 60.]
PREVIOUS_DAY_METHODS = METHODS + [f"{method}_age_{h:.9g}d" for method in DECAY_METHODS for h in HALF_LIVES]
IMPROVED = [f"{method}_age_1d" for method in DECAY_METHODS]


# Data loading and checks
def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_sources(directory):
    # The files are linked by question ID and answer sort order
    question_columns = {
        "discover question id": "question_id",
        "question status": "status",
        "question starts at": "opens_at",
        "question ends at": "closes_at",
        "question correctness known at": "known_at",
        "use ordinal scoring": "ordinal_scoring",
        "answer sort order": "source_option_order",
        "answer name": "option_label",
        "answer resolved probability": "outcome",
    }
    forecast_columns = {
        "prediction set id": "submission_id",
        "membership guid": "account_id",
        "discover question id": "question_id",
        "prediction set created at": "created_at",
        "answer sort order": "source_option_order",
        "forecasted probability": "probability",
        "site name": "platform",
    }
    # keep_default_na=False preserves answer labels such as "NA" as text
    questions = pd.read_csv(
        directory / "rct-a-questions-answers.csv",
        usecols=list(question_columns), keep_default_na=False,
    )
    forecasts = pd.read_csv(
        directory / "rct-a-prediction-sets.csv",
        usecols=list(forecast_columns), keep_default_na=False,
        dtype={"prediction set id": "int64", "membership guid": "str"},
    )
    return questions.rename(columns=question_columns), forecasts.rename(columns=forecast_columns)


def prepare_questions(questions):
    fields = ["status", "opens_at", "closes_at", "known_at", "ordinal_scoring"]
    require(not questions.duplicated(["question_id", "source_option_order"]).any(),
            "Duplicate question/option key.")
    # Question details repeat on every answer row, so all copies must agree
    require((questions.groupby("question_id")[fields].nunique() == 1).all().all(),
            "Question details differ between answer rows.")
    metadata = questions.drop_duplicates("question_id").set_index("question_id")
    metadata = metadata.loc[metadata.status == "resolved", fields].copy()
    for field in ("opens_at", "closes_at", "known_at"):
        metadata[field] = pd.to_datetime(metadata[field], utc=True)
    require(metadata[["opens_at", "closes_at", "known_at"]].notna().all().all(),
            "Missing question time.")
    # Forecasts count until the question closes or its outcome is known, whichever is first
    metadata["eligible_end"] = metadata[["closes_at", "known_at"]].min(axis=1)
    require((metadata.opens_at < metadata.eligible_end).all(), "Invalid question window.")
    ordinal = metadata.ordinal_scoring.astype(str).str.upper()
    require(ordinal.isin(["TRUE", "FALSE"]).all(), "Unexpected ordinal-scoring flag.")
    metadata["question_type"] = np.where(ordinal == "TRUE", "ordered", "unordered")

    columns = ["question_id", "source_option_order", "option_label", "outcome"]
    resolved = questions.question_id.isin(metadata.index)
    options = questions.loc[resolved, columns].copy()
    options = options.sort_values(["question_id", "source_option_order"])
    options["outcome"] = pd.to_numeric(options.outcome)
    options["option_order"] = options.groupby("question_id").cumcount()
    option_counts = options.groupby("question_id").size()
    binary = options.question_id.map(option_counts) == 1
    require((options.loc[binary, "option_label"].str.strip().str.lower() == "yes").all(),
            "A single stored answer must represent Yes.")
    # Binary records store only Yes, we derive No
    no = options.loc[binary].copy()
    no["option_order"] = 1
    no["option_label"] = "No"
    no["outcome"] = 1 - no.outcome
    no["source_option_order"] = pd.NA
    options = pd.concat([options, no]).sort_values(["question_id", "option_order"])
    outcome_counts = options.groupby("question_id").outcome.sum()
    require(options.outcome.isin([0, 1]).all() and (outcome_counts == 1).all(),
            "Each question must have exactly one resolved outcome.")
    metadata.loc[no.question_id, "question_type"] = "binary"
    return metadata, options.reset_index(drop=True)


def scoring_day(timestamps):
    """Date of the first 14:00:59 Eastern cutoff at or after each time, including DST changes."""
    local = timestamps.dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    dates = local.dt.normalize()
    return dates.where(local - dates <= CUTOFF, dates + pd.Timedelta(days=1))


def clean_submissions(forecasts, questions, metadata):
    """Check submissions, keep the eligible ones and count what each rule removes."""
    # A submission spans answer rows. Check it as a whole before selecting updates
    fields = ["account_id", "question_id", "created_at"]
    groups = forecasts.groupby("submission_id")
    consistent = groups[fields].nunique(dropna=False) == 1
    require(consistent.all().all(), "Inconsistent submission identity/time.")
    require((forecasts.account_id.str.strip() != "").all(), "Missing account identity.")
    require(forecasts.probability.between(0, 1).all(), "Invalid input probability.")
    require(not forecasts.duplicated(["submission_id", "source_option_order"]).any(),
            "Repeated option.")
    keys = ["question_id", "source_option_order"]
    matched = forecasts[keys].merge(questions[keys], how="left", indicator=True)
    require((matched._merge == "both").all(), "Unmatched answer option.")
    submissions = forecasts.drop_duplicates("submission_id").set_index("submission_id")
    submissions = submissions[fields].copy()
    row_counts = groups.size()
    question_option_counts = questions.groupby("question_id").size()
    expected_counts = submissions.question_id.map(question_option_counts)
    complete = row_counts.reindex(submissions.index) == expected_counts
    require(complete.all(), "Incomplete submission.")

    submissions["created_at"] = pd.to_datetime(submissions.created_at, utc=True)
    require(submissions.created_at.notna().all(), "Missing submission time.")
    # Keep resolved questions, from opening until closing or the known outcome
    resolved = submissions.question_id.isin(metadata.index)
    early = submissions.created_at < submissions.question_id.map(metadata.opens_at)
    late = submissions.created_at >= submissions.question_id.map(metadata.eligible_end)
    counts = {"Removed: question not resolved": (~resolved).sum(),
              "Removed: made before the question opened": early.sum(),
              "Removed: made at or after closing or the known outcome": late.sum()}
    submissions = submissions.loc[resolved & ~early & ~late].copy()
    submissions["day"] = scoring_day(submissions.created_at)
    cutoffs = submissions.day + CUTOFF
    cutoffs = cutoffs.dt.tz_localize(TIMEZONE).dt.tz_convert("UTC")
    # A question can close between two cutoffs. Its last partial day has no
    # cutoff while it is open, so those submissions are never scored
    eligible = cutoffs < submissions.question_id.map(metadata.eligible_end)
    counts["Removed: made after the question's last cutoff"] = (~eligible).sum()
    submissions = submissions.loc[eligible].copy()
    counts["Eligible submissions"] = len(submissions)
    forecasts = forecasts.loc[forecasts.submission_id.isin(submissions.index)].copy()

    totals = forecasts.groupby("submission_id").probability.sum()
    option_counts = row_counts.reindex(totals.index)
    multiple = option_counts > 1
    # K four-decimal entries allow K*0.00005 rounding error in their sum
    rounding_allowance = option_counts * 0.00005 + 1e-12
    valid_sum = (totals > 0) & ((totals - 1).abs() <= rounding_allowance)
    require((valid_sum | ~multiple).all(), "Invalid probability sum.")
    # Rescale complete multicategory forecasts
    divisors = totals.where(multiple, 1)
    forecasts["probability"] /= forecasts.submission_id.map(divisors)

    # Select the latest eligible account/question/day update, after filtering above.
    # A later ineligible update must not replace it; larger IDs break time ties
    submissions = submissions.sort_values(["created_at", "submission_id"])
    submissions = submissions.drop_duplicates(["question_id", "account_id", "day"], keep="last")
    counts["Removed: replaced later that day by the same account"] = (
        counts["Eligible submissions"] - len(submissions))
    forecasts = forecasts.loc[forecasts.submission_id.isin(submissions.index)].copy()
    require(not submissions.empty and set(submissions.question_id) == set(metadata.index),
            "Some resolved questions have no eligible updates; inspect coverage.")
    return forecasts, submissions, counts


# Aggregation formulas and HFC scoring
def normalize(values):
    return values / values.sum()


def geometric_pool(probabilities, binary, weights=None):
    """GM and GMO of the forecasts, one row per account, optionally weighted."""
    # A geometric mean is zero if any forecast is zero, so one account could rule out
    # an answer for everyone. Raise probabilities below EPSILON, then rescale each row
    adjusted = np.maximum(probabilities, EPSILON)
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    gm = np.exp(np.average(np.log(adjusted), axis=0, weights=weights))

    # Binary GM averages Yes geometrically and gives No the remainder
    # https://github.com/forecastingresearch/aggutils/blob/main/R/agg.R
    geometric = np.array([gm[0], 1 - gm[0]]) if binary else normalize(gm)
    # The geometric mean of binary odds is gm[0] / gm[1]. Converting it back
    # to a probability gives gm[0] / sum(gm). Normalizing the whole vector
    # also gives the article's extension for questions with more than two answers
    # https://forum.effectivealtruism.org/posts/sMjcjnnpoAQCcedL2/when-pooling-forecasts-use-the-geometric-mean-of-odds
    odds = normalize(gm)
    return geometric, odds


def aggregate(probabilities, binary):
    # The five basic methods; rows are accounts and columns answers
    mean = probabilities.mean(axis=0)
    medians = np.median(probabilities, axis=0)
    # Trim the lowest and highest 10% separately for each answer; round down
    trim = len(probabilities) // 10
    middle = np.sort(probabilities, axis=0)[trim:len(probabilities) - trim]
    trimmed = middle.mean(axis=0)
    geometric, odds = geometric_pool(probabilities, binary)
    # If every answer has zero median or trimmed mean, use the arithmetic mean
    return {
        "mean": mean,
        "median": normalize(medians) if medians.sum() > 0 else mean,
        "geometric_mean": geometric,
        "trimmed_mean": normalize(trimmed) if trimmed.sum() > 0 else mean,
        "geometric_odds": odds,
    }


def brier_score(prediction, outcome, ordered):
    """HFC Brier score for one question-day: 0 is perfect and 2 the worst."""
    error = prediction - outcome
    if ordered:
        # Split at each of the K-1 boundaries into "this answer or an earlier one"
        # versus "a later one", score each split as a Yes/No question and average
        # Each binary split has errors e and -e, so its score is 2 * e**2
        return 2 * np.square(error.cumsum()[:-1]).mean()
    return np.square(error).sum()


def scheduled_cutoff(day):
    # Apply the cutoff in Eastern time before converting to UTC, to respect daylight saving
    return (pd.Timestamp(day) + CUTOFF).tz_localize(TIMEZONE).tz_convert("UTC")


def age_predictions(probabilities, binary, ages):
    """Forecasts from the standing forecasts at the previous day's cutoff."""
    if len(probabilities) == 0:
        # With no forecasts available at yesterday's cutoff, give each answer equal probability
        uniform = np.full(probabilities.shape[1], 1 / probabilities.shape[1])
        return {method: uniform.copy() for method in PREVIOUS_DAY_METHODS}
    result = aggregate(probabilities, binary)
    for half_life in HALF_LIVES:
        # The newest forecast gets weight 1 and each extra half-life halves the weight.
        # This shift leaves the averages unchanged but stops all weights rounding to zero
        weights = 0.5 ** ((ages - ages.min()) / half_life)
        result[f"mean_age_{half_life:.9g}d"] = np.average(probabilities, axis=0, weights=weights)
        gm, gmo = geometric_pool(probabilities, binary, weights)
        result[f"geometric_mean_age_{half_life:.9g}d"] = gm
        result[f"geometric_odds_age_{half_life:.9g}d"] = gmo
    return result


# Follow each question through its days with new eligible submissions
def calculate(forecasts, submissions, metadata, options):
    """Score the previous-day comparison before adding each day's updates, and same-day scores after."""
    daily_rows, score_rows = [], []
    forecast_groups = forecasts.groupby("question_id")
    for question_id, updates in submissions.groupby("question_id", sort=True):
        question = metadata.loc[question_id]
        binary = question.question_type == "binary"
        labels = options.loc[options.question_id == question_id].sort_values("option_order")
        stored_order = labels.source_option_order.dropna().astype(int)
        wide = forecast_groups.get_group(question_id).pivot(
            index="submission_id", columns="source_option_order", values="probability")
        # Match rows to updates and columns to the question's answer order once.
        # Later, positions index these rows; they are not submission IDs.
        wide = wide.reindex(index=updates.index, columns=stored_order)
        probabilities = wide.to_numpy(dtype=float)
        if binary:
            # Only Yes is stored; add No as the second answer
            probabilities = np.column_stack([probabilities[:, 0], 1 - probabilities[:, 0]])
        outcome = labels.outcome.to_numpy()
        updates = updates.assign(position=np.arange(len(updates)))
        latest_positions = {}  # account -> row of its latest eligible forecast

        def record(predictions, day, stage, cutoff, accounts):
            # Save each method's error and one forecast row per answer
            common = {"stage": stage, "question_id": question_id,
                      "question_type": question.question_type, "day": day.date().isoformat(),
                      "information_cutoff_utc": cutoff.isoformat(), "accounts": accounts}
            losses = {method: brier_score(p, outcome, question.question_type == "ordered")
                      for method, p in predictions.items()}
            score_rows.append(dict(common, **losses))
            for option in labels.itertuples(index=False):
                row = dict(common, option_order=option.option_order,
                           option_label=option.option_label, outcome=option.outcome)
                row.update({method: p[option.option_order] for method, p in predictions.items()})
                daily_rows.append(row)

        for day, daily_updates in updates.groupby("day", sort=True):
            # Before today's updates enter, the standing set is what was known
            # at yesterday's cutoff. The previous-day comparison scores every method on that set.
            prior = scheduled_cutoff(day - pd.Timedelta(days=1))
            if prior >= question.opens_at:
                positions = list(latest_positions.values())
                created_at = updates.iloc[positions].created_at
                ages = (prior - created_at).dt.total_seconds().to_numpy() / 86400
                predictions = age_predictions(probabilities[positions], binary, ages)
                record(predictions, day, "previous_day", prior, len(positions))

            # Same-day scores include today's updates and carry other accounts forward
            latest_positions.update(zip(daily_updates.account_id, daily_updates.position))
            positions = list(latest_positions.values())
            standing = probabilities[positions]
            predictions = aggregate(standing, binary)
            predictions["geometric_mean_no_first"] = predictions["geometric_mean"]
            if binary:
                # Calculate GM from No instead of Yes, then restore Yes/No answer order
                reversed_gm, _ = geometric_pool(standing[:, ::-1], binary=True)
                predictions["geometric_mean_no_first"] = reversed_gm[::-1]
            record(predictions, day, "same_day", scheduled_cutoff(day), len(standing))
    return pd.DataFrame(daily_rows), pd.DataFrame(score_rows)


def historical_selection(losses, metadata):
    """Choose a half-life at each cutoff using only outcomes known by then."""
    selected = losses[["question_id"]].copy()
    choices = []
    cutoffs = losses.groupby("information_cutoff_utc").groups
    for method in DECAY_METHODS:
        candidates = [method] + [f"{method}_age_{h:.9g}d" for h in [1, 3, 7, 14, 30, 60]]
        # A resolved question contributes its average over all scored days. The
        # known_at filter below admits that average only once its outcome is known.
        question_scores = losses.groupby("question_id")[candidates].mean()
        for cutoff, indices in cutoffs.items():
            resolved_questions = metadata.index[metadata.known_at <= pd.Timestamp(cutoff)]
            resolved_questions = question_scores.index.intersection(resolved_questions)
            # Use no decay until an outcome is known
            chosen_method = method
            if len(resolved_questions) > 0:
                chosen_method = question_scores.loc[resolved_questions].mean().idxmin()
            selected.loc[indices, method] = losses.loc[indices, chosen_method]
            choices.append((method, cutoff, len(resolved_questions), chosen_method))
    return selected.groupby("question_id").mean(), pd.DataFrame(
        choices, columns=["method", "cutoff", "known_outcomes", "choice"]
    )


# Reproduce the tables and the uncertainty and selection results
def data_audit(raw, counts, updates, metadata, scores):
    # Data scale and cleaning counts, printed first in results.txt
    same_day = scores.loc[scores.stage == "same_day"]
    previous_day = scores.loc[scores.stage == "previous_day"]
    types = metadata.question_type.value_counts()
    return pd.Series({
        "Resolved questions": ", ".join(f"{n} {kind}" for kind, n in types.items()),
        "Prediction file: answer rows": len(raw),
        "Prediction file: submissions": raw.submission_id.nunique(),
        "Prediction file: accounts": raw.account_id.nunique(),
        "Prediction file: platforms": raw.platform.nunique(),
        **counts,
        "Kept submissions": len(updates),
        "Kept accounts": updates.account_id.nunique(),
        "Same-day question-days (new submissions that day)": len(same_day),
        "Previous-day question-days": len(previous_day),
        "Previous-day first days dropped (opened after previous cutoff)": len(same_day) - len(previous_day),
        "Previous-day days with no earlier forecast (uniform)": (previous_day.accounts == 0).sum(),
    })


def write_results(path, audit, scores, metadata, options):
    # Average days within questions first so every question has equal weight
    same_day = scores.loc[scores.stage == "same_day"].groupby("question_id").mean(numeric_only=True)
    previous_day_daily = scores.loc[scores.stage == "previous_day"]
    previous_day = previous_day_daily.groupby("question_id")[PREVIOUS_DAY_METHODS].mean()
    selected, choices = historical_selection(previous_day_daily, metadata)
    with path.open("w", encoding="utf-8") as report:
        def show(title, value):
            if isinstance(value, pd.DataFrame):
                value = value.rename(columns={"mean": "Arithmetic mean"})
            if isinstance(value, (pd.DataFrame, pd.Series)):
                value = value.rename(index={"mean": "Arithmetic mean"})
                value = value.to_string(float_format="{:.6f}".format)
            text = f"\n{title}\n{value}\n"
            report.write(text)
            print(text, end="")

        show("Data audit: counts of submissions unless stated", audit)
        # Table 1 averages over all questions, then within each question type
        table1 = pd.DataFrame({"overall": same_day[METHODS].mean()})
        for kind, questions in metadata.groupby("question_type"):
            table1[kind] = same_day.loc[questions.index, METHODS].mean()
        show("Memo Table 1 and scores by question type (lower is better)", table1)
        table2 = previous_day[METHODS + IMPROVED].mean()
        show("Memo Table 2: all methods use the previous day's cutoff", table2)
        show("GM calculated directly from binary No probabilities, overall", same_day.geometric_mean_no_first.mean())
        # Binary GM gives Yes no more than GMO does; compare questions by outcome
        binary_ids = metadata.index[metadata.question_type == "binary"]
        yes = options.loc[(options.option_order == 0) & options.question_id.isin(binary_ids)]
        resolved_as = yes.set_index("question_id").outcome.map({0: "resolved No", 1: "resolved Yes"})
        show("Binary outcomes", resolved_as.value_counts())
        binary_scores = same_day.loc[binary_ids, METHODS + ["geometric_mean_no_first"]]
        show("Binary questions by outcome: Table 1 methods",
             binary_scores.groupby(resolved_as).mean().T)
        sweep = pd.DataFrame({method: [previous_day[f"{method}_age_{h:.9g}d"].mean() for h in HALF_LIVES]
                              for method in DECAY_METHODS}, index=HALF_LIVES)
        sweep.index.name = "Half-life (days)"
        show("Scores at all 12 tested half-lives", sweep)
        gains = {method: 100 * (1 - previous_day[improved].mean() / previous_day[method].mean())
                 for method, improved in zip(DECAY_METHODS, IMPROVED)}
        show("Error reduction from daily halving (%)", pd.Series(gains))

        # One paired difference per question, so the SE treats questions as the unit
        pairs = [("geometric_mean", "geometric_mean_age_1d"),
                 ("geometric_odds", "geometric_odds_age_1d"),
                 ("geometric_mean_age_1d", "geometric_odds_age_1d")]
        rows = []
        for first, second in pairs:
            difference = previous_day[first] - previous_day[second]
            se = difference.std(ddof=1) / np.sqrt(len(difference))
            mean = difference.mean()
            rows.append((first + " minus " + second, mean, se, 100 * se / abs(mean), abs(mean) / se))
        show("Supplement: paired question differences. SEs assume independent questions and "
             "ignore that daily halving was chosen on these data",
             pd.DataFrame(rows, columns=["comparison", "difference", "SE", "SE / difference (%)", "difference / SE"])
             .set_index("comparison"))
        historical = pd.DataFrame({"selected": selected.mean(),
                                   "no_decay": previous_day[DECAY_METHODS].mean()})
        historical["reduction (%)"] = 100 * (1 - historical.selected / historical.no_decay)
        show("Supplement: historical rate selection, a retrospective check on explored data", historical)
        show("Historical choices, counted once per cutoff", choices.groupby(["method", "choice"]).size())
    return previous_day


def plot_decay(scores, directory):
    half_lives = [h for h in HALF_LIVES if h <= 14]
    means = scores.mean()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, method, title, color in zip(axes, GEOMETRIC_METHODS,
            ["Geometric mean (GM)", "Geometric mean of odds (GMO)"], ["#245d89", "#9a4d27"]):
        ax.axhline(means[method], color=color, ls="--", lw=1, alpha=.6)
        ax.text(14, means[method] + .003, "No decay", ha="right", va="bottom", color=color)
        curve = [means[f"{method}_age_{h:.9g}d"] for h in half_lives]
        ax.plot(half_lives, curve, "o-", color=color, lw=2, ms=4)
        one_day = means[f"{method}_age_1d"]
        ax.scatter([1], [one_day], s=70, facecolors="white", edgecolors=color, lw=1.8, zorder=4)
        ax.text(1, one_day + .013, "1 day\n(daily halving)", ha="center", va="bottom")
        ax.set_xscale("log")
        ax.set_xlim(.008, 18)
        ax.set_xticks([.01, 1 / 24, 1, 14], ["14.4 min", "1 h", "1 d", "14 d"])
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("Forecast half-life", labelpad=6)
        ax.text(0, -.32, "Faster decay", transform=ax.transAxes, ha="left", va="top")
        ax.text(1, -.32, "Slower decay", transform=ax.transAxes, ha="right", va="top")
        ax.grid(axis="y", alpha=.16)
    axes[0].set_ylabel("HFC Brier score\n(lower is better)")
    fig.subplots_adjust(bottom=.30, top=.9, wspace=.24)
    fig.savefig(directory / "decay-comparison.png", dpi=220, facecolor="white")
    plt.close(fig)


def main():
    directory = Path(__file__).resolve().parent
    raw = (directory / Path(RAW_DIRECTORY).expanduser()).resolve()
    print("Reading and checking individual submissions...")
    questions, raw_forecasts = read_sources(raw)
    metadata, options = prepare_questions(questions)
    forecasts, updates, counts = clean_submissions(raw_forecasts, questions, metadata)
    print("Calculating both cutoff comparisons and the decay study...")
    daily, scores = calculate(forecasts, updates, metadata, options)
    daily.to_csv(directory / "daily_forecasts.csv", index=False, float_format="%.17g")
    audit = data_audit(raw_forecasts, counts, updates, metadata, scores)
    previous_day_question_scores = write_results(directory / "results.txt", audit, scores, metadata, options)
    plot_decay(previous_day_question_scores, directory)
    print(f"\nSaved results.txt, daily_forecasts.csv and decay-comparison.png in {directory}")


if __name__ == "__main__":
    main()

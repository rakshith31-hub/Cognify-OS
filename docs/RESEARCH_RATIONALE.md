# Research rationale for the action mapping

This is the reasoning behind `config/default_config.yaml`. Sources are
summarized in my own words below; follow the links for the originals.

## FOCUS → suppress notifications (Do Not Disturb)

Gloria Mark's UC Irvine field studies (observing knowledge workers directly,
over ~20 years) are the primary basis here. Two findings matter:

- After an interruption, it takes on the order of **20+ minutes** to
  return to a task at the original level of depth — not the length of the
  interruption itself.
- People typically don't return to the interrupted task immediately; her
  observational data shows roughly **two intervening tasks** happen first.

So the single highest-leverage change during a *confirmed* focus window is
simply not interrupting — hence `dnd_on` is the only action for `focus`.
(The original 23-minutes-15-seconds figure is widely cited from interviews
with Mark; a controlled follow-up experiment — "The Cost of Interrupted
Work: More Speed and Stress" — is the peer-reviewed source and is worth
reading directly if you want the methodology, not just the headline
number.)

- Mark's CHI'08 paper: https://ics.uci.edu/~gmark/chi08-mark.pdf

## FATIGUE → suggest a short break + reduce brightness

Meta-analytic evidence supports short breaks measurably reducing fatigue
and restoring vigor:

- Albulescu et al. (2022), *PLOS ONE* — systematic review/meta-analysis
  across 22 studies (~2,300 participants): microbreaks significantly
  increase vigor and reduce fatigue.
- Lab studies on "systematic microbreaks" (e.g. brief pauses every
  7.5–9.5 minutes during sustained cognitive tasks) show improved
  concentration-time-ratio and reduced perceived workload (NASA-TLX)
  versus no-break conditions.
- The University of Waterloo's summary of this literature notes low-load
  tasks recover with microbreaks of 10 minutes or less, while high-load
  cognitive work (like most knowledge work) benefits from breaks but may
  need more than a few seconds.

Brightness reduction targets the eye-strain component specifically:
`feature_importance.png` shows EAR (eye aspect ratio) and blink rate as
top fatigue predictors in this model, both classic droopy-eyelid /
dry-eye signals, and glare/brightness is a standard, well-established
lever for that.

- Meta-analysis: https://doi.org/10.1371/journal.pone.0272460

## OVERLOAD → strip extraneous load, don't add more prompts

Based on Cognitive Load Theory (Sweller): working memory has a limited
capacity, and *extraneous* load (things unrelated to the task itself —
visual clutter, competing notifications, too many open threads) degrades
performance on top of whatever *intrinsic* load the task already carries.
The model's top overload drivers here are backspace/error rate and pupil
diameter (a standard cognitive-load proxy in eye-tracking research), so
the intervention removes load (DND) rather than adding a new decision for
the user to make right when they're already overloaded — hence a single,
low-friction notification plus DND, not a menu of options.

## DISTRACTED → reduce further fragmentation, optional self-chosen blocking

Same interruption-recovery research as `focus` applies here in reverse:
attention is already fragmented, so the intervention avoids adding more
fragmentation. Domain blocking is opt-in and limited to domains you name
yourself in the config — this tool does not maintain or apply any
third-party blocklist.

## What this does *not* claim

This app's own model is trained on **synthetic data** (see main README) —
it demonstrates a plausible pipeline and a defensible, literature-grounded
action mapping, but the *state classification itself* is not yet validated
against real labeled human sessions. Treat the interventions as reasonable
defaults to tune, not as a clinically validated system. If you retrain on
real per-user data (README section "About the training data"), it's worth
re-checking whether the same thresholds (`consecutive_windows_required`,
`min_confidence_to_act`) still make sense for your accuracy profile.

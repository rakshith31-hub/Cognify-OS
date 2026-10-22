"""
config_validator.py
--------------------
Validates config/default_config.yaml (or a user config derived from it)
BEFORE the daemon starts acting on it.

Motivation: most of the "it's not working" bugs debugged in this project
were not code bugs — they were configs that parsed fine and ran silently
wrong for hours (enabled: false left as the literal example value,
site_blocking and the per-state block_domains action both enabled and
fighting each other, a typo'd action type that just logs one easy-to-miss
warning line and no-ops forever). None of that raises an exception, so
none of it was ever caught until someone noticed the *behavior* was wrong.
This runs once at startup and turns those into an explicit, itemized
report instead.

Two severities:
  - ERRORS   — config is structurally broken enough that the daemon
               shouldn't start (missing required keys, wrong types,
               out-of-range values that would make the state machine
               misbehave in an unrecoverable way).
  - WARNINGS — config is valid and the daemon CAN start, but something
               about it is very likely not what the person meant.
"""

from dataclasses import dataclass, field

KNOWN_ACTION_TYPES = {
    "dnd_on", "dnd_off", "notify", "brightness_step", "color_filter",
    "high_contrast", "taskbar_hide", "alarm", "break_popup",
    "block_domains", "run_command",
}
KNOWN_STATES = {"focus", "fatigue", "overload", "distracted"}
KNOWN_TOP_LEVEL_KEYS = {"general", "states", "revert_on_state_change", "site_blocking"}
KNOWN_GENERAL_KEYS = {
    "model_type", "ensemble_cnn_weight",
    "predict_interval_seconds", "consecutive_windows_required", "action_cooldown_seconds",
    "min_confidence_to_act", "low_confidence_revert_after", "idle_pause_enabled",
    "idle_pause_after_minutes", "personal_calibration_enabled", "personal_calibration_min_windows",
    "label_prompt_enabled", "label_session_interval_minutes", "label_session_duration_minutes",
    "label_session_prompt_interval_seconds",
    "label_max_age_seconds", "auto_retrain_on_startup", "auto_retrain_min_new_samples",
    "head_pose_override_enabled",
    "head_pose_off_screen_ratio_threshold", "head_yaw_threshold_deg", "head_pitch_threshold_deg",
    "head_pose_calibration_frames", "head_pose_smoothing_window", "log_dir", "log_file",
    "open_dashboard_on_exit", "dashboard_port", "bandit_db_file", "bandit_cold_start_trials",
}


@dataclass
class ValidationResult:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Daemon may start — errors is what actually gates that, warnings don't."""
        return len(self.errors) == 0

    def format_report(self) -> str:
        lines = []
        if self.errors:
            lines.append(f"CONFIG ERRORS ({len(self.errors)}) — must fix before starting:")
            for e in self.errors:
                lines.append(f"  ✗ {e}")
        if self.warnings:
            lines.append(f"CONFIG WARNINGS ({len(self.warnings)}) — probably not what you meant:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        if not self.errors and not self.warnings:
            lines.append("Config looks good — no errors or warnings.")
        return "\n".join(lines)


def _check_range(result, general, key, lo, hi, required_type=(int, float)):
    if key not in general:
        return
    val = general[key]
    if not isinstance(val, required_type) or isinstance(val, bool):
        result.errors.append(f"general.{key} must be a number, got {val!r}")
        return
    if not (lo <= val <= hi):
        result.errors.append(f"general.{key}={val} is outside the sane range [{lo}, {hi}]")


def validate_config(config: dict) -> ValidationResult:
    result = ValidationResult()

    if not isinstance(config, dict) or not config:
        result.errors.append("Config file is empty or not a valid YAML mapping.")
        return result

    # ---- unknown top-level keys (usually a typo or bad indentation) ----
    for key in config:
        if key not in KNOWN_TOP_LEVEL_KEYS:
            result.warnings.append(
                f"Unrecognized top-level key '{key}' — typo, or a setting that landed at "
                f"the wrong indentation level (e.g. it belongs under 'general:')?"
            )

    # ---- general ----
    general = config.get("general")
    if not isinstance(general, dict):
        result.errors.append("Missing or invalid 'general:' section.")
        general = {}
    else:
        for key in general:
            if key not in KNOWN_GENERAL_KEYS:
                result.warnings.append(f"Unrecognized key 'general.{key}' — typo?")

        model_type = general.get("model_type", "rf")
        if model_type not in ("rf", "cnn", "ensemble"):
            result.errors.append(
                f"general.model_type='{model_type}' is not recognized "
                f"(expected 'rf', 'cnn', or 'ensemble')."
            )
        if model_type in ("cnn", "ensemble"):
            result.warnings.append(
                f"general.model_type='{model_type}' requires torch to be installed "
                f"(pip install torch) — the default 'rf' does not."
            )
        _check_range(result, general, "ensemble_cnn_weight", 0.0, 1.0)

        _check_range(result, general, "predict_interval_seconds", 1, 300)
        _check_range(result, general, "consecutive_windows_required", 1, 20)
        _check_range(result, general, "action_cooldown_seconds", 0, 3600)
        _check_range(result, general, "min_confidence_to_act", 0.0, 1.0)
        _check_range(result, general, "low_confidence_revert_after", 1, 20)
        _check_range(result, general, "idle_pause_after_minutes", 0.5, 240)
        _check_range(result, general, "label_session_interval_minutes", 5, 1440)
        _check_range(result, general, "label_session_duration_minutes", 1, 60)
        _check_range(result, general, "label_session_prompt_interval_seconds", 10, 600)
        if ("label_session_duration_minutes" in general and "label_session_prompt_interval_seconds" in general
                and general["label_session_duration_minutes"] * 60 < general["label_session_prompt_interval_seconds"]):
            result.warnings.append(
                "label_session_duration_minutes is shorter than label_session_prompt_interval_seconds "
                "— a session will only ever fire its initial prompt, never a follow-up."
            )
        _check_range(result, general, "label_max_age_seconds", 10, 3600)
        _check_range(result, general, "auto_retrain_min_new_samples", 1, 10000)
        _check_range(result, general, "personal_calibration_min_windows", 5, 1000)
        _check_range(result, general, "head_pose_off_screen_ratio_threshold", 0.0, 1.0)
        _check_range(result, general, "head_yaw_threshold_deg", 1, 90)
        _check_range(result, general, "head_pitch_threshold_deg", 1, 90)
        _check_range(result, general, "head_pose_calibration_frames", 1, 300)
        _check_range(result, general, "head_pose_smoothing_window", 1, 60)
        _check_range(result, general, "dashboard_port", 1024, 65535)
        _check_range(result, general, "bandit_cold_start_trials", 0, 1000)

        if "head_pose_smoothing_window" in general and "head_pose_calibration_frames" in general:
            if general["head_pose_smoothing_window"] > general["head_pose_calibration_frames"]:
                result.warnings.append(
                    "general.head_pose_smoothing_window is larger than "
                    "head_pose_calibration_frames — smoothing will always run on fewer "
                    "frames than intended right after calibration finishes."
                )

    # ---- states ----
    states = config.get("states")
    if not isinstance(states, dict) or not states:
        result.errors.append("Missing or empty 'states:' section — the daemon has nothing to do.")
        states = {}

    for missing in KNOWN_STATES - set(states.keys()):
        result.warnings.append(
            f"No actions configured for state '{missing}' — the daemon will detect it but do "
            f"nothing when it fires. Intentional (e.g. you don't want any intervention for "
            f"'{missing}') or a section that got deleted by accident?"
        )

    site_blocking_via_distracted = False
    for state_name, state_cfg in states.items():
        if not isinstance(state_cfg, dict) or not isinstance(state_cfg.get("actions"), list):
            result.errors.append(f"states.{state_name}.actions must be a list of action dicts.")
            continue
        for i, action in enumerate(state_cfg["actions"]):
            if not isinstance(action, dict) or "type" not in action:
                result.errors.append(f"states.{state_name}.actions[{i}] is missing a 'type' key.")
                continue
            action_type = action["type"]
            if action_type not in KNOWN_ACTION_TYPES:
                result.warnings.append(
                    f"states.{state_name}.actions[{i}] has unknown type '{action_type}' — "
                    f"typo? This silently no-ops (logs one warning line, does nothing) "
                    f"instead of failing loudly, so it's easy to miss for a long time."
                )
            if action_type == "block_domains":
                if action.get("enabled", False) and not action.get("domains"):
                    result.warnings.append(
                        f"states.{state_name}.actions[{i}] block_domains has enabled: true "
                        f"but an empty domains list — nothing will actually be blocked."
                    )
                for d in action.get("domains", []) or []:
                    if "://" in d or "/" in d:
                        result.warnings.append(
                            f"states.{state_name}.actions[{i}] domain '{d}' looks like a full "
                            f"URL, not a bare domain — it's silently normalized to a bare "
                            f"hostname before blocking, but writing it as e.g. 'youtube.com' "
                            f"instead of 'https://www.youtube.com/' is clearer and avoids "
                            f"relying on that normalization."
                        )
                if action.get("enabled", False):
                    site_blocking_via_distracted = True

    # ---- site_blocking vs per-state block_domains conflict ----
    site_blocking = config.get("site_blocking", {})
    if isinstance(site_blocking, dict):
        sb_enabled = site_blocking.get("enabled", False)
        if sb_enabled and not site_blocking.get("domains"):
            result.warnings.append(
                "site_blocking.enabled is true but domains is empty — nothing will actually "
                "be blocked."
            )
        if sb_enabled and site_blocking_via_distracted:
            result.warnings.append(
                "BOTH site_blocking.enabled AND a per-state block_domains action are enabled "
                "at the same time. These share the same hosts-file mechanism and will fight "
                "each other — the per-state action's duration_minutes timer can prematurely "
                "unblock your session-wide list. Use one or the other, not both."
            )
        pct = site_blocking.get("dim_percent")
        if isinstance(pct, (int, float)) and not (0 <= pct <= 100):
            result.errors.append(f"site_blocking.dim_percent={pct} must be between 0 and 100.")
        for d in site_blocking.get("domains", []) or []:
            if "://" in d or "/" in d:
                result.warnings.append(
                    f"site_blocking domain '{d}' looks like a full URL, not a bare domain — "
                    f"it's silently normalized to a bare hostname before blocking, but writing "
                    f"it as e.g. 'youtube.com' instead of 'https://www.youtube.com/' is clearer "
                    f"and avoids relying on that normalization."
                )
    elif "site_blocking" in config:
        result.errors.append("'site_blocking:' must be a mapping (enabled/domains/...).")

    return result


def validate_config_file(path: str) -> ValidationResult:
    import yaml
    with open(path) as f:
        config = yaml.safe_load(f)
    return validate_config(config)


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "config/default_config.yaml"
    result = validate_config_file(target)
    print(result.format_report())
    sys.exit(1 if not result.ok else 0)

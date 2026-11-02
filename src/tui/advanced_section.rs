//! Advanced settings: less-common knobs the TUI surfaces in one place.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use super::app::{Action, App};
use super::common::{self, FeedbackLevel, FormRowSpec, TextInput, TextInputResult};
use super::config_editor::{ConfigEditor, EditorError};
use crate::config::TranscriptionEngine;

/// Upper bound the config schema accepts for the root `idle_unload_secs`
/// key (one day). Kept here so the form can reject values the daemon would
/// reject during save validation, with a useful message instead of a
/// generic parse error.
const MAX_IDLE_UNLOAD_SECS: u64 = 86_400;

/// One minute. Arrow keys nudge the idle-unload window in this step; Enter
/// opens the text editor for an exact second count.
const IDLE_UNLOAD_STEP: u64 = 60;

#[derive(Debug, Clone)]
pub struct TextEdit {
    pub field: Field,
    pub input: TextInput,
}

#[derive(Debug, Clone)]
pub struct AdvancedState {
    pub gpu_isolation: bool,
    /// The `[whisper] on_demand_loading` toggle, edited from this form.
    pub on_demand_loading: bool,
    /// Engine picked by the root `engine` key. The idle-unload window follows
    /// whichever engine is active.
    pub engine: TranscriptionEngine,
    /// The active engine's own `on_demand_loading` as read from its table.
    /// Unused for Whisper, where the toggle above is the same setting.
    pub engine_on_demand_loading: bool,
    /// Seconds the active engine's primary model stays resident after a
    /// dictation when that engine loads on demand. 0 unloads after every
    /// dictation.
    pub idle_unload_secs: u64,
    pub flash_attention: bool,
    pub eager_processing: bool,
    pub gpu_device: Option<i64>,
    pub streaming: bool,
    pub field: Field,
    pub editing: Option<TextEdit>,
    pub feedback: Option<(FeedbackLevel, String)>,
    pub dirty_since_load: bool,
    /// Name of a Parakeet model the user just auto-switched into via the
    /// streaming-toggle save handler, waiting for the user to confirm with
    /// `d` that they want voxtype to fork the download in the background.
    /// `None` when no prompt is pending. See #423.
    pub pending_download: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Field {
    GpuIsolation,
    OnDemand,
    IdleUnload,
    FlashAttention,
    Eager,
    GpuDevice,
    Streaming,
}
impl Field {
    const ALL: &'static [Field] = &[
        Field::GpuIsolation,
        Field::OnDemand,
        Field::IdleUnload,
        Field::FlashAttention,
        Field::Eager,
        Field::GpuDevice,
        Field::Streaming,
    ];
}

/// Table holding the active engine's `on_demand_loading`. `None` for Soniox,
/// a cloud backend with nothing to load.
fn engine_on_demand_table(engine: TranscriptionEngine) -> Option<&'static str> {
    match engine {
        TranscriptionEngine::Whisper => Some("whisper"),
        TranscriptionEngine::Parakeet => Some("parakeet"),
        TranscriptionEngine::Moonshine => Some("moonshine"),
        TranscriptionEngine::SenseVoice => Some("sensevoice"),
        TranscriptionEngine::Paraformer => Some("paraformer"),
        TranscriptionEngine::Dolphin => Some("dolphin"),
        TranscriptionEngine::Omnilingual => Some("omnilingual"),
        TranscriptionEngine::Cohere => Some("cohere"),
        TranscriptionEngine::OpenVino => Some("openvino"),
        TranscriptionEngine::Soniox => None,
    }
}

impl AdvancedState {
    pub fn load() -> Result<Self, EditorError> {
        let ed = ConfigEditor::load()?;
        Ok(Self::from_editor(&ed))
    }

    /// Build the form state from an already-loaded editor. Split out so tests
    /// can point the section at a synthetic config file without touching the
    /// process-wide TUI config path.
    fn from_editor(ed: &ConfigEditor) -> Self {
        // The idle-unload window belongs to whichever engine is active, so the
        // applicability check has to know both the engine and that engine's
        // own on-demand toggle.
        let engine = ed
            .get_string("", "engine")
            .and_then(|name| name.parse::<TranscriptionEngine>().ok())
            .unwrap_or_default();
        let engine_on_demand_loading = engine_on_demand_table(engine)
            .and_then(|table| ed.get_bool(table, "on_demand_loading"))
            .unwrap_or(false);
        Self {
            gpu_isolation: ed.get_bool("whisper", "gpu_isolation").unwrap_or(false),
            on_demand_loading: ed.get_bool("whisper", "on_demand_loading").unwrap_or(false),
            engine,
            engine_on_demand_loading,
            // The schema bounds this to 0..=86400. Clamp on read so a
            // hand-edited out-of-range value can still be displayed and
            // fixed in the form instead of overflowing the u64.
            idle_unload_secs: ed
                .get_int("", "idle_unload_secs")
                .unwrap_or(0)
                .clamp(0, MAX_IDLE_UNLOAD_SECS as i64) as u64,
            flash_attention: ed.get_bool("whisper", "flash_attention").unwrap_or(false),
            eager_processing: ed.get_bool("whisper", "eager_processing").unwrap_or(false),
            gpu_device: ed.get_int("whisper", "gpu_device"),
            streaming: ed.get_bool("parakeet", "streaming").unwrap_or(false),
            field: Field::GpuIsolation,
            editing: None,
            feedback: None,
            dirty_since_load: false,
            pending_download: None,
        }
    }

    pub fn save(&mut self) -> Action {
        let mut ed = match ConfigEditor::load() {
            Ok(e) => e,
            Err(e) => {
                self.feedback = Some((FeedbackLevel::Err, format!("load: {}", e)));
                return Action::None;
            }
        };
        self.save_to(&mut ed)
    }

    /// Apply the form to `ed` and save it. Split out so tests can save into a
    /// synthetic file instead of the user's real config.
    fn save_to(&mut self, ed: &mut ConfigEditor) -> Action {
        ed.set_bool("whisper", "gpu_isolation", self.gpu_isolation);
        ed.set_bool("whisper", "on_demand_loading", self.on_demand_loading);
        ed.set_int("", "idle_unload_secs", self.idle_unload_secs as i64);
        ed.set_bool("whisper", "flash_attention", self.flash_attention);
        ed.set_bool("whisper", "eager_processing", self.eager_processing);
        match self.gpu_device {
            Some(n) if n >= 0 => ed.set_int("whisper", "gpu_device", n),
            _ => ed.unset("whisper", "gpu_device"),
        }
        // Read the previous streaming value BEFORE we overwrite it so we can
        // detect the "just turned on" transition. Used below to decide whether
        // to auto-switch the configured Parakeet model to a streaming-capable
        // one.
        let was_streaming = ed.get_bool("parakeet", "streaming").unwrap_or(false);
        ed.set_bool("parakeet", "streaming", self.streaming);

        // Streaming dictation requires toggle activation: typing at the cursor
        // while a PTT key is held confuses libinput's held-key tracker on
        // Hyprland/Sway/River, so the key-release event never reaches voxtype.
        // Rewrite the user's hotkey mode rather than silently letting the
        // daemon fight it at startup.
        let promoted_hotkey_mode = if self.streaming
            && ed
                .get_string("hotkey", "mode")
                .map(|m| m == "push_to_talk")
                .unwrap_or(false)
        {
            ed.set_string("hotkey", "mode", "toggle");
            true
        } else {
            false
        };

        // Streaming dictation also requires a model that ships
        // tokenizer.model (parakeet-rs::ParakeetUnified::load fails without
        // it). When the user flips streaming on from off, check the active
        // Parakeet model and auto-switch to the canonical streaming-capable
        // one if needed. Tell them clearly so they know to run
        // the model download next — it isn't blocking on the
        // TUI save action (it's ~2.7 GB). #423.
        let auto_switched_model: Option<(String, &'static str)> =
            if self.streaming && !was_streaming {
                let current = ed
                    .get_string("parakeet", "model")
                    .unwrap_or_else(|| "parakeet-tdt-0.6b-v3".to_string());
                if !crate::setup::model::is_streaming_compatible_parakeet(&current) {
                    ed.set_string(
                        "parakeet",
                        "model",
                        crate::setup::model::DEFAULT_PARAKEET_STREAMING_MODEL,
                    );
                    Some((
                        current,
                        crate::setup::model::DEFAULT_PARAKEET_STREAMING_MODEL,
                    ))
                } else {
                    None
                }
            } else {
                None
            };

        match ed.save() {
            Ok(()) => {
                self.dirty_since_load = false;
                // Compose feedback from whichever auto-fixes fired this save.
                // Both can fire on the same save (PTT→toggle and model swap),
                // so combine them rather than picking one.
                let mut parts: Vec<String> = Vec::new();
                parts.push(format!("Saved to {}", ed.path().display()));
                if promoted_hotkey_mode {
                    parts.push(
                        "Streaming requires toggle mode: hotkey activation \
                         auto-promoted from push_to_talk to toggle."
                            .to_string(),
                    );
                }
                let did_auto_switch = auto_switched_model.is_some();
                if let Some((from, to)) = auto_switched_model {
                    parts.push(format!(
                        "Streaming requires a model with tokenizer.model. \
                         Switched [parakeet] model from '{from}' to '{to}'. \
                         Press 'd' to download in the background (~2.7 GB), \
                         any other key to skip."
                    ));
                    // Park the model name on state so the next 'd' keypress
                    // can fork the download. Any other key is treated as
                    // "skip"; the user can re-trigger by running
                    // `voxtype setup --download --model <NAME>` directly.
                    self.pending_download = Some(to.to_string());
                }
                let level = if promoted_hotkey_mode || did_auto_switch {
                    FeedbackLevel::Warn
                } else {
                    FeedbackLevel::Ok
                };
                self.feedback = Some((level, parts.join(" ")));
            }
            Err(e) => self.feedback = Some((FeedbackLevel::Err, format!("save: {}", e))),
        }
        Action::None
    }
    pub fn reset(&mut self) {
        if let Ok(fresh) = Self::load() {
            let field = self.field;
            *self = fresh;
            self.field = field;
            self.feedback = Some((FeedbackLevel::Ok, "Reverted".to_string()));
        }
    }
    fn move_field(&mut self, delta: i32) {
        let len = Field::ALL.len() as i32;
        let cur = Field::ALL
            .iter()
            .position(|f| *f == self.field)
            .unwrap_or(0) as i32;
        self.field = Field::ALL[((cur + delta).rem_euclid(len)) as usize];
    }

    /// The idle-unload window only means something when the active engine's
    /// primary model is loaded on demand and runs in-process. Resident mode
    /// never unloads it, and Whisper's GPU isolation runs each transcription
    /// in a subprocess that exits, so there is nothing to retain. Matches the
    /// daemon's own condition. Other engines are unaffected by Whisper's
    /// GPU isolation.
    fn idle_unload_applicable(&self) -> bool {
        self.active_engine_on_demand()
            && !(self.engine == TranscriptionEngine::Whisper && self.gpu_isolation)
    }

    /// On-demand setting of the engine the idle-unload window applies to. For
    /// Whisper that is the toggle edited in this form, so flipping it updates
    /// applicability immediately; other engines own their own toggle in the
    /// Engine section, which this form does not edit.
    fn active_engine_on_demand(&self) -> bool {
        match self.engine {
            TranscriptionEngine::Whisper => self.on_demand_loading,
            _ => self.engine_on_demand_loading,
        }
    }

    fn start_edit_if_idle_unload(&mut self) -> bool {
        if self.field != Field::IdleUnload || !self.idle_unload_applicable() {
            return false;
        }
        self.editing = Some(TextEdit {
            field: Field::IdleUnload,
            input: TextInput::new(self.idle_unload_secs.to_string()),
        });
        true
    }

    /// Commit the typed second count. Invalid input leaves the previous value
    /// alone and reports why, rather than saving something the daemon's
    /// schema would reject on load.
    fn commit_idle_unload(&mut self, buffer: String) {
        let trimmed = buffer.trim();
        match trimmed.parse::<u64>() {
            Ok(secs) if secs <= MAX_IDLE_UNLOAD_SECS => {
                self.idle_unload_secs = secs;
                self.dirty_since_load = true;
                self.feedback = None;
            }
            _ => {
                self.feedback = Some((
                    FeedbackLevel::Err,
                    format!(
                        "Idle unload must be a whole number of seconds from 0 to {MAX_IDLE_UNLOAD_SECS}."
                    ),
                ));
            }
        }
    }

    fn cycle(&mut self, delta: i32) {
        match self.field {
            Field::GpuIsolation => self.gpu_isolation = !self.gpu_isolation,
            Field::OnDemand => self.on_demand_loading = !self.on_demand_loading,
            Field::IdleUnload => {
                if !self.idle_unload_applicable() {
                    return;
                }
                if delta > 0 {
                    self.idle_unload_secs =
                        (self.idle_unload_secs + IDLE_UNLOAD_STEP).min(MAX_IDLE_UNLOAD_SECS);
                } else {
                    self.idle_unload_secs = self.idle_unload_secs.saturating_sub(IDLE_UNLOAD_STEP);
                }
            }
            Field::FlashAttention => self.flash_attention = !self.flash_attention,
            Field::Eager => self.eager_processing = !self.eager_processing,
            Field::GpuDevice => {
                let cur = self.gpu_device.unwrap_or(-1);
                let next = cur + delta as i64;
                self.gpu_device = if next < 0 { None } else { Some(next.min(7)) };
            }
            Field::Streaming => self.streaming = !self.streaming,
        }
        self.dirty_since_load = true;
        self.feedback = None;
    }
}

pub fn render(f: &mut Frame, area: Rect, app: &App) {
    let state = match &app.advanced {
        Some(s) => s,
        None => {
            let block = Block::default().borders(Borders::ALL).title("Advanced");
            let inner = block.inner(area);
            f.render_widget(block, area);
            f.render_widget(
                Paragraph::new("Failed to load config.").wrap(Wrap { trim: true }),
                inner,
            );
            return;
        }
    };
    render_state(f, area, state);
}

/// Render from a state value so the layout can be tested without building a
/// full `App` (which probes daemon status and the installed binary).
fn render_state(f: &mut Frame, area: Rect, state: &AdvancedState) {
    let rows = vec![
        FormRowSpec::new(
            state.field == Field::GpuIsolation,
            "GPU isolation (subprocess)",
            yesno(state.gpu_isolation),
        ),
        FormRowSpec::new(
            state.field == Field::OnDemand,
            "On-demand model loading",
            yesno(state.on_demand_loading),
        ),
        FormRowSpec::new(
            state.field == Field::IdleUnload,
            "Idle unload (seconds)",
            match state.editing.as_ref() {
                Some(e) if e.field == Field::IdleUnload => e.input.caret_string(),
                _ => format_idle_unload(state.idle_unload_secs),
            },
        )
        .dimmed(!state.idle_unload_applicable()),
        FormRowSpec::new(
            state.field == Field::FlashAttention,
            "Flash attention",
            yesno(state.flash_attention),
        ),
        FormRowSpec::new(
            state.field == Field::Eager,
            "Eager input processing",
            yesno(state.eager_processing),
        ),
        FormRowSpec::new(
            state.field == Field::GpuDevice,
            "GPU device index",
            state
                .gpu_device
                .map(|n| n.to_string())
                .unwrap_or_else(|| "auto".to_string()),
        ),
        FormRowSpec::new(
            state.field == Field::Streaming,
            "Streaming output (experimental, Parakeet)",
            yesno(state.streaming),
        ),
    ];

    let feedback_pair = state
        .feedback
        .as_ref()
        .map(|(lvl, msg)| (*lvl, msg.as_str()));

    common::render_form_with_guidance(
        f,
        area,
        "Advanced",
        state.dirty_since_load,
        feedback_pair,
        &rows,
        guidance_for_field(state),
    );
}

fn yesno(b: bool) -> String {
    (if b { "yes" } else { "no" }).to_string()
}

/// Compact value for the idle-unload row. The guidance panel carries the
/// full explanation; this only has to be scannable in the narrow value column.
fn format_idle_unload(secs: u64) -> String {
    match secs {
        0 => "0 s (immediate)".to_string(),
        s if s % 60 == 0 => format!("{} s ({} min)", s, s / 60),
        s => format!("{} s", s),
    }
}

fn heading<'a>(text: &'a str) -> Line<'a> {
    Line::from(Span::styled(
        text,
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    ))
}

fn guidance_for_field(state: &AdvancedState) -> Vec<Line<'_>> {
    match state.field {
        Field::GpuIsolation => vec![
            heading("GPU isolation"),
            Line::from(""),
            Line::from(
                "Runs each transcription in a short-lived subprocess that \
                 exits afterward. The GPU releases all VRAM between recordings \
                 instead of holding the model resident.",
            ),
            Line::from(""),
            Line::from(Span::styled(
                "Turn it on if:",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(
                "  • You're on a laptop with hybrid graphics and want the \
                 discrete GPU to power down between dictations.",
            ),
            Line::from(
                "  • You see VRAM use creep upward over a long voxtype \
                 session.",
            ),
            Line::from(""),
            Line::from(Span::styled(
                "Leave it off if:",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(
                "  • Latency matters more than VRAM. Subprocess startup adds \
                 ~100-300 ms per recording.",
            ),
        ],
        Field::OnDemand => vec![
            heading("On-demand model loading"),
            Line::from(""),
            Line::from(
                "When on, voxtype loads the model only when recording starts \
                 and unloads it when the Idle unload window below expires (0 \
                 unloads right after each dictation). When off, the model \
                 stays resident from daemon start.",
            ),
            Line::from(""),
            Line::from(Span::styled(
                "Turn it on if:",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(
                "  • You only dictate occasionally and don't want the daemon \
                 holding ~1-2 GB of RAM in the background.",
            ),
            Line::from(""),
            Line::from(Span::styled(
                "Leave it off if:",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(
                "  • You dictate frequently. Resident-mode transcription \
                 starts instantly; on-demand loads add a one-shot delay on \
                 the first key press.",
            ),
            Line::from(""),
            Line::from(Span::styled(
                "If you turn this on but dictate in bursts, set Idle unload \
                 below so the model survives between dictations.",
                Style::default().fg(Color::Gray),
            )),
        ],
        Field::IdleUnload => {
            let mut lines = vec![
                heading("Idle unload (on-demand retention)"),
                Line::from(""),
                Line::from(
                    "Controls how long the primary model of the active engine \
                     stays in memory after a dictation. It only has an effect \
                     while that engine loads models on demand — turn that on \
                     in the Engine section.",
                ),
                Line::from(""),
                Line::from(Span::styled(
                    "0",
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from("  • Unload the model after every dictation. Each one pays the load."),
                Line::from(Span::styled(
                    "N (1-86400)",
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from(
                    "  • Keep the model for N seconds after a dictation ends. \
                     Each dictation restarts the window, so back-to-back \
                     dictations reuse it, and the model unloads after N seconds \
                     with no dictation.",
                ),
                Line::from(""),
                Line::from(Span::styled(
                    "Enter types an exact value. ← → adjust by one minute.",
                    Style::default().fg(Color::Gray),
                )),
                Line::from(""),
                Line::from(Span::styled(
                    "For Whisper this also needs max_loaded_models > 0. Not the \
                     same as cold_model_timeout_secs: that setting covers \
                     secondary models, and 0 there means never unload.",
                    Style::default().fg(Color::Gray),
                )),
            ];
            if !state.active_engine_on_demand() {
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "Inactive: on-demand loading is off for the active engine. \
                     Enable it in the Engine section.",
                    Style::default().fg(Color::Yellow),
                )));
            }
            if state.engine == TranscriptionEngine::Whisper && state.gpu_isolation {
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "Inactive: GPU isolation runs each recording in a \
                     subprocess that exits afterward, so there is no resident \
                     model to keep.",
                    Style::default().fg(Color::Yellow),
                )));
            }
            lines
        }
        Field::FlashAttention => vec![
            heading("Flash attention"),
            Line::from(""),
            Line::from(
                "A GPU-only inference optimization that reduces memory \
                 bandwidth pressure in the attention layers. Speeds up \
                 transcription on capable cards.",
            ),
            Line::from(""),
            Line::from(Span::styled(
                "Turn it on if:",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(
                "  • You're on Vulkan or CUDA with a recent GPU. \
                 Particularly noticeable on large-v3 and large-v3-turbo.",
            ),
            Line::from(""),
            Line::from(Span::styled(
                "Leave it off if:",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(
                "  • You're CPU-only or on older hardware (no benefit, may \
                 cause crashes on a few drivers).",
            ),
        ],
        Field::Eager => vec![
            heading("Eager input processing"),
            Line::from(""),
            Line::from(
                "Voxtype starts transcribing audio chunks while you're still \
                 recording, instead of waiting until you release the PTT key. \
                 The final transcript stitches the chunks together.",
            ),
            Line::from(""),
            Line::from(Span::styled(
                "Turn it on if:",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(
                "  • You record long-form (>10 sec) and the post-recording \
                 wait feels like dead time.",
            ),
            Line::from(""),
            Line::from(Span::styled(
                "Leave it off if:",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(
                "  • You record short bursts (a few seconds). The chunked \
                 transcripts can occasionally split a sentence awkwardly.",
            ),
            Line::from(
                "  • You're on a laptop and CPU/GPU heat matters. Eager \
                 mode keeps the inference engine busy during recording.",
            ),
        ],
        Field::GpuDevice => vec![
            heading("GPU device index"),
            Line::from(""),
            Line::from(
                "Picks which GPU voxtype targets on multi-GPU systems. The \
                 default (auto) leaves the choice to the driver, which often \
                 picks the integrated GPU on hybrid laptops.",
            ),
            Line::from(""),
            Line::from(Span::styled(
                "Set a specific index if:",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(
                "  • Transcription is slower than expected and you suspect \
                 the iGPU is being used. Try 1 (or 2) to target the \
                 discrete card.",
            ),
            Line::from(""),
            Line::from(Span::styled(
                "Run `vulkaninfo --summary` or `nvidia-smi -L` to see your \
                 device numbering.",
                Style::default().fg(Color::Gray),
            )),
        ],
        Field::Streaming => vec![
            heading("Streaming output (experimental)"),
            Line::from(""),
            Line::from(Span::styled(
                "EXPERIMENTAL in 0.7.2 — Parakeet only.",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(
                "Voxtype types text incrementally while you're still speaking \
                 instead of waiting for hotkey release. Uses the parakeet-rs \
                 ParakeetUnified cache-aware streaming API; sub-second first \
                 token on most hardware.",
            ),
            Line::from(""),
            Line::from(Span::styled(
                "Turn it on if:",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(
                "  • You're on Parakeet TDT v3 and want a live feel rather \
                 than waiting for the final stitched transcript.",
            ),
            Line::from(
                "  • You record long-form and want partials to appear as \
                 you speak.",
            ),
            Line::from(""),
            Line::from(Span::styled(
                "Leave it off if:",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(
                "  • You're on Whisper or another batch engine. Streaming \
                 returns an error rather than silently shimming — switch \
                 engine first via the Engine section.",
            ),
            Line::from(
                "  • You use heavy text post-processing (LLM cleanup). \
                 Streaming types partials before post_process can see the \
                 final transcript.",
            ),
            Line::from(""),
            Line::from(Span::styled(
                "Requires a streaming-capable model (TDT v3 family with \
                 tokenizer.model). Voxtype prints a clear error at daemon \
                 start if the active model can't stream.",
                Style::default().fg(Color::Gray),
            )),
        ],
    }
}

pub fn handle_key(app: &mut App, key: KeyEvent) -> Action {
    let state = match app.advanced.as_mut() {
        Some(s) => s,
        None => return Action::None,
    };

    // A text edit owns the keyboard until Enter (commit) or Esc (cancel), so
    // digits and `s`/`r` type into the field instead of triggering save/reset.
    if state.editing.is_some() {
        return handle_edit_key(state, key);
    }

    // When a streaming-model auto-switch fires during save, `d` confirms the
    // background download. Any other key dismisses the prompt — including
    // navigation and re-save. Handle this BEFORE the regular key dispatch so
    // a stale prompt can't bleed into a later edit.
    if state.pending_download.is_some() {
        if let KeyCode::Char(c) = key.code {
            if c.eq_ignore_ascii_case(&'d') {
                let model_name = state.pending_download.take().expect("checked above");
                spawn_background_model_download(model_name.clone());
                state.feedback = Some((
                    FeedbackLevel::Ok,
                    format!(
                        "Downloading {} in the background. Watch for a desktop notification when it completes.",
                        model_name
                    ),
                ));
                return Action::None;
            }
        }
        // Any other key dismisses the prompt and falls through to normal handling.
        state.pending_download = None;
    }

    match key.code {
        KeyCode::Up | KeyCode::Char('k') => {
            state.move_field(-1);
            Action::None
        }
        KeyCode::Down | KeyCode::Char('j') => {
            state.move_field(1);
            Action::None
        }
        KeyCode::Left | KeyCode::Char('h') => {
            state.cycle(-1);
            Action::None
        }
        KeyCode::Right | KeyCode::Char('l') | KeyCode::Char(' ') => {
            state.cycle(1);
            Action::None
        }
        KeyCode::Enter | KeyCode::Char('i') => {
            state.start_edit_if_idle_unload();
            Action::None
        }
        KeyCode::Char('s') => state.save(),
        KeyCode::Char('r') => {
            state.reset();
            Action::None
        }
        _ => Action::None,
    }
}

fn handle_edit_key(state: &mut AdvancedState, key: KeyEvent) -> Action {
    let Some(editing) = state.editing.as_mut() else {
        return Action::None;
    };
    match editing.input.handle_key(key) {
        TextInputResult::Continue => Action::None,
        TextInputResult::Commit => {
            let buffer = editing.input.buffer().to_string();
            state.editing = None;
            state.commit_idle_unload(buffer);
            Action::None
        }
        TextInputResult::Cancel => {
            state.editing = None;
            Action::None
        }
    }
}

/// Fork a detached worker thread that downloads the named Parakeet model and
/// sends desktop notifications around the start, success, and failure
/// transitions. The thread is fire-and-forget — voxtype's TUI doesn't
/// supervise it, doesn't join on it, and survives a TUI exit (`std::thread`
/// is detached, and `download_parakeet_model` writes to the on-disk models
/// dir which the daemon picks up on next start).
///
/// All three notifications go through `notification::send_status_sync`, which
/// owns the `--replace-id` bookkeeping keeping every voxtype notification in
/// one slot that replaces in place (#679). Shelling out to `notify-send`
/// directly here got a fresh server ID each time, so these stacked on KDE
/// while real dictation notifications replaced each other — the one place a
/// user watches notifications to judge their setup behaved unlike actual use.
///
/// Urgency levels:
/// - `low` for the "starting" notification (informational, just letting the
///   user know background work began)
/// - `normal` for "complete" (user should see it but it's not urgent)
/// - `critical` for "failed" (the user took an explicit action and it didn't
///   succeed; they need to know to retry)
fn spawn_background_model_download(model_name: String) {
    std::thread::spawn(move || {
        // Starting notification — low urgency, just informational.
        crate::notification::send_status_sync(
            "Voxtype",
            &format!("Downloading model {} in the background…", model_name),
            "low",
        );

        // Synchronous download in this worker thread. `download_parakeet_model`
        // is `fn` (not `async fn`), so no tokio runtime is required here.
        let result = crate::setup::model::download_parakeet_model(&model_name);

        match result {
            Ok(()) => {
                crate::notification::send_status_sync(
                    "Voxtype",
                    &format!(
                        "Model download complete: {}. Restart voxtype to use it.",
                        model_name
                    ),
                    "normal",
                );
            }
            Err(e) => {
                crate::notification::send_status_sync(
                    "Voxtype",
                    &format!(
                        "Model download failed: {}. Run `voxtype setup --download --model {}` to retry.",
                        e, model_name
                    ),
                    "critical",
                );
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::CONFIG_KEYS;
    use crate::config::{load_config, Config};
    use std::fs;
    use std::path::PathBuf;

    fn temp_config(contents: &str) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(&path, contents).unwrap();
        (dir, path)
    }

    /// Every field in this form edits a key the schema knows about. Extending
    /// `Field::ALL` without extending this match fails to compile, which is
    /// the point: the compiler forces a schema decision for each new row.
    #[test]
    fn every_advanced_field_maps_to_a_schema_key() {
        let covered: Vec<&str> = Field::ALL
            .iter()
            .map(|f| match f {
                Field::GpuIsolation => "whisper.gpu_isolation",
                Field::OnDemand => "whisper.on_demand_loading",
                Field::IdleUnload => "idle_unload_secs",
                Field::FlashAttention => "whisper.flash_attention",
                Field::Eager => "whisper.eager_processing",
                Field::GpuDevice => "whisper.gpu_device",
                Field::Streaming => "parakeet.streaming",
            })
            .collect();

        let missing: Vec<&&str> = covered
            .iter()
            .filter(|k| !CONFIG_KEYS.iter().any(|s| s.key == **k))
            .collect();
        assert!(
            missing.is_empty(),
            "TUI fields editing keys absent from the schema: {missing:?}"
        );
    }

    /// Both on-demand controls must be reachable from the same section, and
    /// the idle-retention key must be in the schema's Advanced group. This is
    /// the regression guard for `idle_unload_secs` existing in the config and
    /// CLI but being invisible in `voxtype configure`.
    #[test]
    fn both_on_demand_settings_have_a_form_field() {
        for key in ["whisper.on_demand_loading", "idle_unload_secs"] {
            assert!(
                Field::ALL.iter().any(|f| match f {
                    Field::OnDemand => key == "whisper.on_demand_loading",
                    Field::IdleUnload => key == "idle_unload_secs",
                    _ => false,
                }),
                "{key} has no Advanced form field"
            );
            let spec = crate::config::schema::scalar_keys()
                .find(|s| s.key == key)
                .unwrap_or_else(|| panic!("{key} must be user-settable"));
            assert_eq!(spec.section, "Advanced", "{key} should live in Advanced");
        }

        // The idle-retention spec is the one the form must mirror.
        assert_eq!(
            crate::config::schema::scalar_keys()
                .find(|s| s.key == "idle_unload_secs")
                .map(|s| s.ty),
            Some(crate::config::schema::KeyType::Int {
                min: 0,
                max: MAX_IDLE_UNLOAD_SECS as i64,
            })
        );
        // Guard against a duplicate copy of the default in the UI.
        assert_eq!(Config::default().idle_unload_secs, 0);
    }

    #[test]
    fn load_reads_a_nonzero_idle_unload_and_the_toggle() {
        let (_dir, path) =
            temp_config("idle_unload_secs = 300\n[whisper]\non_demand_loading = true\n");
        let ed = ConfigEditor::load_from_path(path).unwrap();
        let state = AdvancedState::from_editor(&ed);
        assert!(state.on_demand_loading);
        assert_eq!(state.idle_unload_secs, 300);
        assert!(state.idle_unload_applicable());
    }

    #[test]
    fn save_writes_root_idle_unload_secs_and_keeps_on_demand() {
        let (_dir, path) = temp_config("[whisper]\non_demand_loading = true\n");
        let mut ed = ConfigEditor::load_from_path(path.clone()).unwrap();
        let mut state = AdvancedState::from_editor(&ed);
        state.idle_unload_secs = 900;
        state.save_to(&mut ed);

        assert_eq!(
            ConfigEditor::load_from_path(path.clone())
                .unwrap()
                .get_int("", "idle_unload_secs"),
            Some(900)
        );

        // The daemon's own loader must see both values, not just the TOML
        // editor's view of the file.
        let cfg = load_config(Some(&path)).unwrap();
        assert_eq!(cfg.idle_unload_secs, 900);
        assert!(
            cfg.whisper.on_demand_loading,
            "saving idle retention must not turn on-demand loading off"
        );
        assert!(!state.dirty_since_load);
    }

    #[test]
    fn invalid_idle_unload_values_are_rejected_without_touching_state() {
        let mut state = AdvancedState {
            gpu_isolation: false,
            on_demand_loading: true,
            engine: TranscriptionEngine::Whisper,
            engine_on_demand_loading: true,
            idle_unload_secs: 120,
            flash_attention: false,
            eager_processing: false,
            gpu_device: None,
            streaming: false,
            field: Field::IdleUnload,
            editing: None,
            feedback: None,
            dirty_since_load: false,
            pending_download: None,
        };

        for bad in ["", "abc", "-5", "1.5", "86401"] {
            state.commit_idle_unload(bad.to_string());
            assert_eq!(
                state.idle_unload_secs, 120,
                "{bad:?} must not replace the previous value"
            );
            assert!(!state.dirty_since_load, "{bad:?} must not dirty the form");
            assert!(
                matches!(state.feedback, Some((FeedbackLevel::Err, _))),
                "{bad:?} must explain why it was rejected"
            );
            state.feedback = None;
        }

        state.commit_idle_unload("45".to_string());
        assert_eq!(state.idle_unload_secs, 45);
        assert!(state.dirty_since_load);
        assert!(state.feedback.is_none());
    }

    #[test]
    fn zero_is_a_valid_value_and_saves_as_zero() {
        let (_dir, path) =
            temp_config("idle_unload_secs = 300\n[whisper]\non_demand_loading = true\n");
        let mut ed = ConfigEditor::load_from_path(path.clone()).unwrap();
        let mut state = AdvancedState::from_editor(&ed);
        state.commit_idle_unload("0".to_string());
        state.save_to(&mut ed);

        assert_eq!(
            ConfigEditor::load_from_path(path.clone())
                .unwrap()
                .get_int("", "idle_unload_secs"),
            Some(0)
        );
        assert_eq!(load_config(Some(&path)).unwrap().idle_unload_secs, 0);
    }

    #[test]
    fn idle_unload_is_inactive_without_on_demand_or_with_gpu_isolation() {
        let mut state = AdvancedState {
            gpu_isolation: false,
            on_demand_loading: true,
            engine: TranscriptionEngine::Whisper,
            engine_on_demand_loading: true,
            idle_unload_secs: 60,
            flash_attention: false,
            eager_processing: false,
            gpu_device: None,
            streaming: false,
            field: Field::IdleUnload,
            editing: None,
            feedback: None,
            dirty_since_load: false,
            pending_download: None,
        };
        assert!(state.idle_unload_applicable());
        assert!(state.start_edit_if_idle_unload());
        state.editing = None;

        // Resident mode: the model never idles out, so the window is inert.
        state.on_demand_loading = false;
        assert!(!state.idle_unload_applicable());
        assert!(!state.start_edit_if_idle_unload());
        state.cycle(1);
        assert_eq!(state.idle_unload_secs, 60, "inactive field must stay inert");

        // GPU isolation: the subprocess exits, so there is nothing to retain.
        state.on_demand_loading = true;
        state.gpu_isolation = true;
        assert!(!state.idle_unload_applicable());
        assert!(!state.start_edit_if_idle_unload());
    }

    /// The window follows the *active* engine's on-demand setting. Whisper's
    /// GPU isolation is meaningless for Parakeet, which loads in-process, so it
    /// must not disable the row there.
    #[test]
    fn idle_unload_follows_the_active_engine() {
        let (_dir, path) = temp_config(
            "engine = \"parakeet\"\nidle_unload_secs = 300\n\
             [whisper]\ngpu_isolation = true\n\
             [parakeet]\non_demand_loading = true\n",
        );
        let ed = ConfigEditor::load_from_path(path).unwrap();
        let state = AdvancedState::from_editor(&ed);
        assert_eq!(state.engine, TranscriptionEngine::Parakeet);
        assert!(
            state.idle_unload_applicable(),
            "Whisper's gpu_isolation must not disable Parakeet's idle window"
        );

        // Same engine, on-demand off: nothing to retain, so the row goes inert.
        let (_dir, path) =
            temp_config("engine = \"parakeet\"\n[parakeet]\non_demand_loading = false\n");
        let ed = ConfigEditor::load_from_path(path).unwrap();
        let state = AdvancedState::from_editor(&ed);
        assert_eq!(state.engine, TranscriptionEngine::Parakeet);
        assert!(!state.idle_unload_applicable());

        // Whisper keeps its existing rule: GPU isolation runs each recording
        // in a subprocess, so there is no resident model to keep.
        let (_dir, path) = temp_config(
            "engine = \"whisper\"\n[whisper]\non_demand_loading = true\ngpu_isolation = true\n",
        );
        let ed = ConfigEditor::load_from_path(path).unwrap();
        let state = AdvancedState::from_editor(&ed);
        assert_eq!(state.engine, TranscriptionEngine::Whisper);
        assert!(!state.idle_unload_applicable());
    }

    /// Soniox is cloud-only, so nothing is ever loaded on demand.
    #[test]
    fn soniox_engine_never_applies_the_idle_window() {
        let (_dir, path) = temp_config("engine = \"soniox\"\nidle_unload_secs = 600\n");
        let ed = ConfigEditor::load_from_path(path).unwrap();
        let state = AdvancedState::from_editor(&ed);
        assert_eq!(state.engine, TranscriptionEngine::Soniox);
        assert_eq!(state.idle_unload_secs, 600);
        assert!(!state.idle_unload_applicable());
    }

    #[test]
    fn arrow_keys_adjust_by_a_minute_and_clamp() {
        let mut state = AdvancedState {
            gpu_isolation: false,
            on_demand_loading: true,
            engine: TranscriptionEngine::Whisper,
            engine_on_demand_loading: true,
            idle_unload_secs: 0,
            flash_attention: false,
            eager_processing: false,
            gpu_device: None,
            streaming: false,
            field: Field::IdleUnload,
            editing: None,
            feedback: None,
            dirty_since_load: false,
            pending_download: None,
        };
        state.cycle(1);
        assert_eq!(state.idle_unload_secs, IDLE_UNLOAD_STEP);
        state.cycle(-1);
        assert_eq!(state.idle_unload_secs, 0);
        state.cycle(-1);
        assert_eq!(state.idle_unload_secs, 0, "cannot go below zero");

        state.idle_unload_secs = MAX_IDLE_UNLOAD_SECS;
        state.cycle(1);
        assert_eq!(state.idle_unload_secs, MAX_IDLE_UNLOAD_SECS);
    }

    fn buffer_text(buf: &ratatui::buffer::Buffer) -> String {
        let area = buf.area;
        let mut out = String::new();
        for y in area.top()..area.bottom() {
            for x in area.left()..area.right() {
                out.push_str(buf.cell((x, y)).map(|c| c.symbol()).unwrap_or(" "));
            }
            out.push('\n');
        }
        out
    }

    /// The row has to be visible next to the existing on-demand toggle, with
    /// a value a user can read at a glance, and the row has to go dim when the
    /// daemon would ignore the setting.
    #[test]
    fn idle_row_shows_its_value_and_dims_when_inapplicable() {
        use ratatui::{backend::TestBackend, Terminal};

        let mut state = AdvancedState {
            gpu_isolation: false,
            on_demand_loading: true,
            engine: TranscriptionEngine::Whisper,
            engine_on_demand_loading: true,
            idle_unload_secs: 300,
            flash_attention: false,
            eager_processing: false,
            gpu_device: None,
            streaming: false,
            field: Field::IdleUnload,
            editing: None,
            feedback: None,
            dirty_since_load: false,
            pending_download: None,
        };

        let mut terminal = Terminal::new(TestBackend::new(140, 36)).unwrap();
        terminal
            .draw(|f| render_state(f, f.area(), &state))
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        let text = buffer_text(&buf);
        assert!(
            text.contains("On-demand model loading"),
            "the existing toggle must stay visible: {text}"
        );
        assert!(
            text.contains("Idle unload (seconds)"),
            "the new field must be discoverable: {text}"
        );
        assert!(
            text.contains("300 s (5 min)"),
            "the loaded value must be readable: {text}"
        );
        assert!(
            text.contains("Idle unload (on-demand retention)"),
            "the guidance panel must explain the field: {text}"
        );
        let needle = "Idle unload (seconds)";
        let needle_len = needle.chars().count() as u16;
        let pos = (0u16..36)
            .find_map(|y| {
                (0u16..140).find_map(|x| {
                    let mut s = String::new();
                    for i in 0..needle_len {
                        s.push_str(buf.cell((x + i, y)).map(|c| c.symbol()).unwrap_or_default());
                    }
                    (s == needle).then_some((x, y))
                })
            })
            .expect("idle row label in settings panel");
        assert_ne!(
            buf.cell(pos).unwrap().fg,
            Color::DarkGray,
            "applicable row should not be dimmed"
        );

        // Resident mode: the daemon never idles the model out, so the field
        // is inert and the guidance says why.
        state.on_demand_loading = false;
        terminal
            .draw(|f| render_state(f, f.area(), &state))
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        let text = buffer_text(&buf);
        assert!(
            text.contains("Inactive: on-demand loading is off for the active engine"),
            "inapplicability must be explained: {text}"
        );
        assert_eq!(
            buf.cell(pos).unwrap().fg,
            Color::DarkGray,
            "inapplicable row should be dimmed"
        );
    }
}

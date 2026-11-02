//! Model manager for multi-model support
//!
//! Manages the lifecycle of Whisper models, supporting:
//! - LRU caching of loaded models (when gpu_isolation = false)
//! - On-demand loading with automatic eviction
//! - Fresh subprocess per model (when gpu_isolation = true)
//! - Remote backend model selection

use crate::config::{WhisperConfig, WhisperMode};
use crate::error::TranscribeError;
use crate::transcribe::{self, Transcriber};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A loaded model with usage tracking
struct LoadedModel {
    transcriber: Arc<dyn Transcriber>,
    last_used: Instant,
    is_primary: bool,
}

/// Manages multiple Whisper models with LRU eviction
pub struct ModelManager {
    /// Whisper configuration
    config: WhisperConfig,
    /// Path to config file (for subprocess transcribers)
    config_path: Option<PathBuf>,
    /// Loaded models (keyed by model name)
    loaded_models: HashMap<String, LoadedModel>,
    /// Maximum models to keep loaded
    max_loaded: usize,
    /// Timeout before evicting idle secondary models
    cold_timeout: Duration,
    /// How long an on-demand primary model stays resident while idle.
    /// Zero preserves the unload-after-each-dictation behavior.
    idle_unload: Duration,
    /// Whether the configured lifecycle retains the primary between
    /// dictations. Cached at construction so the per-tick eviction check
    /// cannot re-run `effective_mode()` (and its deprecation warnings).
    primary_retention: bool,
}

impl ModelManager {
    /// Create a new model manager
    pub fn new(
        config: &WhisperConfig,
        config_path: Option<PathBuf>,
        idle_unload_secs: u64,
    ) -> Self {
        let primary_retention = config.on_demand_loading
            && !config.gpu_isolation
            && config.effective_mode() == WhisperMode::Local
            && config.max_loaded_models > 0
            && idle_unload_secs > 0;
        Self {
            config: config.clone(),
            config_path,
            loaded_models: HashMap::new(),
            max_loaded: config.max_loaded_models,
            cold_timeout: Duration::from_secs(config.cold_model_timeout_secs),
            idle_unload: Duration::from_secs(idle_unload_secs),
            primary_retention,
        }
    }

    /// Whether the primary model is retained between dictations while idle.
    ///
    /// Only on-demand local Whisper without GPU isolation qualifies; remote,
    /// CLI, subprocess-per-recording, preloaded, and `idle_unload_secs = 0`
    /// configurations keep their previous lifecycle.
    pub fn retains_primary(&self) -> bool {
        self.primary_retention
    }

    /// Return a cached transcriber and mark it as just used, without loading.
    pub fn cached_transcriber(&mut self, model: &str) -> Option<Arc<dyn Transcriber>> {
        let loaded = self.loaded_models.get_mut(model)?;
        loaded.last_used = Instant::now();
        Some(Arc::clone(&loaded.transcriber))
    }

    /// Renew the idle deadline of the retained primary model.
    ///
    /// Called when the daemon returns to idle after using the model so the
    /// countdown starts at completion, cancellation, or failure rather than
    /// at the start of the load.
    pub fn touch_primary(&mut self) {
        self.touch_primary_at(Instant::now());
    }

    fn touch_primary_at(&mut self, now: Instant) {
        if let Some(loaded) = self.loaded_models.get_mut(&self.config.model) {
            loaded.last_used = now;
        }
    }

    /// Keep an already-constructed transcriber in the cache.
    ///
    /// Used for the on-demand primary so the next dictation reuses the same
    /// `Arc` instead of reloading. Respects `max_loaded_models`, evicting an
    /// LRU entry first when necessary.
    pub fn adopt_transcriber(&mut self, model: &str, transcriber: Arc<dyn Transcriber>) {
        if self.max_loaded == 0 {
            return;
        }

        if !self.loaded_models.contains_key(model) {
            while self.loaded_models.len() >= self.max_loaded {
                if !self.evict_one_for_capacity() {
                    break;
                }
            }
        }

        let is_primary = model == self.config.model;
        self.loaded_models.insert(
            model.to_string(),
            LoadedModel {
                transcriber,
                last_used: Instant::now(),
                is_primary,
            },
        );
    }

    /// Free cached slots so `model` can be loaded, respecting
    /// `max_loaded_models`.
    ///
    /// Unlike idle eviction this may release the retained primary when it is
    /// the only occupant and capacity is needed for a different model.
    /// Returns how many models were released.
    pub fn make_room_for(&mut self, model: &str) -> usize {
        if self.max_loaded == 0 || self.loaded_models.contains_key(model) {
            return 0;
        }

        let mut released = 0;
        while self.loaded_models.len() >= self.max_loaded {
            if !self.evict_one_for_capacity() {
                break;
            }
            released += 1;
        }
        released
    }

    /// Check if a model is available (configured as primary, secondary, or in available_models)
    pub fn is_model_available(&self, model: &str) -> bool {
        if model == self.config.model {
            return true;
        }
        if let Some(ref secondary) = self.config.secondary_model {
            if model == secondary {
                return true;
            }
        }
        self.config.available_models.contains(&model.to_string())
    }

    /// Get a transcriber for the specified model
    ///
    /// For GPU isolation mode, creates a fresh subprocess transcriber each time.
    /// For non-isolation mode, returns cached transcriber or loads on demand.
    pub fn get_transcriber(
        &mut self,
        model: Option<&str>,
    ) -> Result<Arc<dyn Transcriber>, TranscribeError> {
        // Clone the model name to avoid borrow issues
        let model_name = model
            .map(|s| s.to_string())
            .unwrap_or_else(|| self.config.model.clone());

        // Validate model is available
        if !self.is_model_available(&model_name) {
            tracing::warn!(
                "Requested model '{}' not in available models, falling back to default '{}'",
                model_name,
                self.config.model
            );
            return self.get_transcriber(None);
        }

        // For remote backend, preserve configured remote_model unless the
        // caller supplied an explicit runtime model override.
        if self.config.effective_mode() == WhisperMode::Remote {
            return self.create_remote_transcriber(model);
        }

        // For CLI backend, create transcriber each time (no caching needed)
        if self.config.effective_mode() == WhisperMode::Cli {
            return self.create_cli_transcriber(&model_name);
        }

        // For GPU isolation, always create fresh subprocess
        if self.config.gpu_isolation {
            return self.create_subprocess_transcriber(&model_name);
        }

        // For non-isolated local backend, use LRU cache
        self.get_or_load_cached(&model_name)
    }

    /// Create a remote transcriber with optional runtime model override
    fn create_remote_transcriber(
        &self,
        model_override: Option<&str>,
    ) -> Result<Arc<dyn Transcriber>, TranscribeError> {
        let mut config = self.config.clone();
        if let Some(model) = model_override {
            config.remote_model = Some(model.to_string());
        }
        let transcriber = transcribe::remote::RemoteTranscriber::new(&config)?;
        Ok(Arc::new(transcriber))
    }

    /// Create a CLI transcriber with model override
    fn create_cli_transcriber(&self, model: &str) -> Result<Arc<dyn Transcriber>, TranscribeError> {
        let mut config = self.config.clone();
        config.model = model.to_string();
        tracing::info!("Using whisper-cli subprocess backend");
        let transcriber = transcribe::cli::CliTranscriber::new(&config)?;
        Ok(Arc::new(transcriber))
    }

    /// Create a subprocess transcriber for the specified model
    fn create_subprocess_transcriber(
        &self,
        model: &str,
    ) -> Result<Arc<dyn Transcriber>, TranscribeError> {
        let mut config = self.config.clone();
        config.model = model.to_string();
        let transcriber =
            transcribe::subprocess::SubprocessTranscriber::new(&config, self.config_path.clone())?;
        Ok(Arc::new(transcriber))
    }

    /// Get transcriber from cache or load on demand (non-isolated mode)
    fn get_or_load_cached(&mut self, model: &str) -> Result<Arc<dyn Transcriber>, TranscribeError> {
        // Check if already loaded
        if let Some(loaded) = self.loaded_models.get_mut(model) {
            loaded.last_used = Instant::now();
            tracing::debug!("Using cached model '{}'", model);
            return Ok(Arc::clone(&loaded.transcriber));
        }

        // Need to load - first check if we need to evict
        while self.loaded_models.len() >= self.max_loaded {
            if !self.evict_one_for_capacity() {
                break;
            }
        }

        // Load the model
        tracing::info!("Loading model '{}' into cache", model);
        let mut config = self.config.clone();
        config.model = model.to_string();

        let transcriber = transcribe::whisper::WhisperTranscriber::new(&config)?;
        let is_primary = model == self.config.model;

        self.loaded_models.insert(
            model.to_string(),
            LoadedModel {
                transcriber: Arc::new(transcriber),
                last_used: Instant::now(),
                is_primary,
            },
        );

        Ok(Arc::clone(
            &self.loaded_models.get(model).unwrap().transcriber,
        ))
    }

    /// Evict one cached model to free a slot.
    ///
    /// Prefers the LRU non-primary entry. The LRU primary is a fallback only
    /// when retention is enabled: resident (`on_demand_loading = false`) and
    /// `idle_unload_secs = 0` configurations must keep their primary even if a
    /// capacity load fails. Returns false when nothing may be evicted.
    fn evict_one_for_capacity(&mut self) -> bool {
        let victim = self
            .loaded_models
            .iter()
            .filter(|(_, m)| !m.is_primary)
            .min_by_key(|(_, m)| m.last_used)
            .map(|(name, _)| name.clone())
            .or_else(|| {
                if !self.retains_primary() {
                    return None;
                }
                self.loaded_models
                    .iter()
                    .min_by_key(|(_, m)| m.last_used)
                    .map(|(name, _)| name.clone())
            });

        match victim {
            Some(model) => {
                tracing::info!("Evicting model '{}' from cache (capacity)", model);
                self.loaded_models.remove(&model);
                true
            }
            None => false,
        }
    }

    /// Drop every cached model, returning how many were released.
    ///
    /// Used after a transcription task panics: the engine may hold state the
    /// panic left inconsistent, and reusing it risks compounding the fault.
    /// Unlike the idle eviction methods this ignores both the idle timeout and
    /// the is_primary flag, because the point is to discard a suspect instance
    /// rather than to reclaim memory (#643).
    pub fn drop_loaded_models(&mut self) -> usize {
        let count = self.loaded_models.len();
        self.loaded_models.clear();
        count
    }

    /// Evict models that haven't been used recently, returning how many were
    /// released.
    ///
    /// Call this periodically (e.g., every second while the daemon is idle).
    /// Secondary models follow `cold_model_timeout_secs`; the on-demand
    /// primary follows `idle_unload_secs` and is only touched when retention
    /// is enabled.
    pub fn evict_idle_models(&mut self) -> usize {
        self.evict_idle_models_at(Instant::now())
    }

    /// Evict idle cached models as of a specific instant.
    ///
    /// Split from `evict_idle_models` so tests can drive the deadlines
    /// deterministically instead of sleeping. Returns how many models were
    /// released, so the caller can trim the heap only when something actually
    /// dropped.
    pub fn evict_idle_models_at(&mut self, now: Instant) -> usize {
        let mut released = 0;

        if self.retains_primary() {
            let timeout = self.idle_unload;
            let to_evict: Vec<String> = self
                .loaded_models
                .iter()
                .filter(|(_, m)| {
                    m.is_primary
                        && now
                            .checked_duration_since(m.last_used)
                            .is_some_and(|age| age >= timeout)
                })
                .map(|(name, _)| name.clone())
                .collect();

            for model in to_evict {
                tracing::info!(
                    "Evicting idle primary model '{}' from cache (unused for {}s)",
                    model,
                    timeout.as_secs()
                );
                self.loaded_models.remove(&model);
                released += 1;
            }
        }

        if self.cold_timeout.is_zero() {
            return released; // Secondary auto-eviction disabled
        }

        let to_evict: Vec<String> = self
            .loaded_models
            .iter()
            .filter(|(_, m)| {
                !m.is_primary
                    && now
                        .checked_duration_since(m.last_used)
                        .is_some_and(|age| age >= self.cold_timeout)
            })
            .map(|(name, _)| name.clone())
            .collect();

        for model in to_evict {
            tracing::info!(
                "Evicting idle model '{}' from cache (unused for {}s)",
                model,
                self.cold_timeout.as_secs()
            );
            self.loaded_models.remove(&model);
            released += 1;
        }

        released
    }

    /// Preload the primary model (if on_demand_loading is false)
    pub fn preload_primary(&mut self) -> Result<(), TranscribeError> {
        if self.config.on_demand_loading {
            tracing::debug!("Skipping primary model preload (on_demand_loading=true)");
            return Ok(());
        }

        if self.config.gpu_isolation {
            tracing::debug!("Skipping primary model preload (gpu_isolation=true)");
            return Ok(());
        }

        if self.config.effective_mode() == WhisperMode::Remote {
            tracing::debug!("Skipping primary model preload (remote backend)");
            return Ok(());
        }

        if self.config.effective_mode() == WhisperMode::Cli {
            tracing::debug!("Skipping primary model preload (cli backend)");
            return Ok(());
        }

        let model = self.config.model.clone();
        tracing::info!("Preloading primary model '{}'", model);
        let _ = self.get_or_load_cached(&model)?;
        Ok(())
    }

    /// Prepare a model for transcription (called when recording starts)
    ///
    /// For subprocess mode, this spawns the worker early so it can load
    /// the model while the user is speaking. The actual worker spawn and
    /// model load happen on a blocking thread, so the async event loop
    /// (and audio capture) is not blocked. Returns a JoinHandle that the
    /// caller must await before invoking transcription, so that we don't
    /// race against the in-flight prepare and end up spawning a second
    /// worker.
    pub fn prepare_model(
        &mut self,
        model: Option<&str>,
    ) -> Result<Option<tokio::task::JoinHandle<()>>, TranscribeError> {
        let model_name = model
            .map(|s| s.to_string())
            .unwrap_or_else(|| self.config.model.clone());

        // Validate model
        if !self.is_model_available(&model_name) {
            tracing::warn!(
                "Cannot prepare unavailable model '{}', will use default",
                model_name
            );
            return Ok(None);
        }

        // For GPU isolation, spawn subprocess early
        if self.config.gpu_isolation && self.config.effective_mode() == WhisperMode::Local {
            let transcriber = self.create_subprocess_transcriber(&model_name)?;
            // Store the Arc immediately so get_prepared_transcriber can retrieve it.
            // The worker spawn happens on a blocking thread; the prepared_worker
            // mutex inside SubprocessTranscriber is populated when ready.
            self.loaded_models.insert(
                format!("_prepared_{}", model_name),
                LoadedModel {
                    transcriber: transcriber.clone(),
                    last_used: Instant::now(),
                    is_primary: false,
                },
            );
            let handle = tokio::task::spawn_blocking(move || {
                transcriber.prepare();
            });
            return Ok(Some(handle));
        }

        Ok(None)
    }

    /// Get a prepared transcriber (if available) or create one
    ///
    /// This checks for a prepared subprocess transcriber first,
    /// then falls back to normal get_transcriber.
    pub fn get_prepared_transcriber(
        &mut self,
        model: Option<&str>,
    ) -> Result<Arc<dyn Transcriber>, TranscribeError> {
        let model_name = model
            .map(|s| s.to_string())
            .unwrap_or_else(|| self.config.model.clone());
        let prepared_key = format!("_prepared_{}", model_name);

        // Check for prepared transcriber
        if let Some(prepared) = self.loaded_models.remove(&prepared_key) {
            tracing::debug!("Using prepared transcriber for model '{}'", model_name);
            return Ok(prepared.transcriber);
        }

        // No prepared transcriber, get normally. Preserve None so remote mode
        // can use whisper.remote_model instead of the local default model.
        self.get_transcriber(model)
    }

    /// Register an already-constructed transcriber as a cached model, so
    /// tests can exercise cache lifecycle paths without a real model file
    /// on disk.
    #[cfg(test)]
    pub(crate) fn insert_loaded_model_for_test(
        &mut self,
        name: &str,
        transcriber: Arc<dyn Transcriber>,
        is_primary: bool,
    ) {
        self.insert_loaded_model_at_for_test(name, transcriber, is_primary, Instant::now());
    }

    /// As `insert_loaded_model_for_test`, but with an explicit last-used
    /// instant so idle-deadline tests are deterministic.
    #[cfg(test)]
    pub(crate) fn insert_loaded_model_at_for_test(
        &mut self,
        name: &str,
        transcriber: Arc<dyn Transcriber>,
        is_primary: bool,
        last_used: Instant,
    ) {
        self.loaded_models.insert(
            name.to_string(),
            LoadedModel {
                transcriber,
                last_used,
                is_primary,
            },
        );
    }

    /// Get the list of currently loaded models (for debugging/status)
    pub fn loaded_model_names(&self) -> Vec<&str> {
        self.loaded_models
            .keys()
            .filter(|k| !k.starts_with("_prepared_"))
            .map(|s| s.as_str())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stand-in engine for cache tests: cheap to construct, never touches
    /// a model file.
    struct StubTranscriber;

    impl Transcriber for StubTranscriber {
        fn transcribe(&self, _samples: &[f32]) -> Result<String, TranscribeError> {
            Ok(String::new())
        }
    }

    /// #643: after a panicked transcription the cached engine is suspect, so
    /// it is dropped wholesale — including the primary, which normal idle
    /// eviction deliberately keeps.
    #[test]
    fn drop_loaded_models_clears_everything_and_reports_the_count() {
        let mut mm = ModelManager::new(&test_config(), None, 0);
        assert_eq!(mm.drop_loaded_models(), 0, "nothing cached yet");

        mm.insert_loaded_model_for_test("base.en", Arc::new(StubTranscriber), true);
        mm.insert_loaded_model_for_test("large-v3-turbo", Arc::new(StubTranscriber), false);
        assert_eq!(mm.loaded_model_names().len(), 2);

        assert_eq!(
            mm.drop_loaded_models(),
            2,
            "the primary is dropped too, unlike idle eviction"
        );
        assert!(mm.loaded_models.is_empty());
        assert!(mm.loaded_model_names().is_empty());
    }

    fn retention_config() -> WhisperConfig {
        WhisperConfig {
            mode: Some(WhisperMode::Local),
            model: "base.en".to_string(),
            threads: Some(4),
            on_demand_loading: true,
            max_loaded_models: 1,
            cold_model_timeout_secs: 300,
            ..Default::default()
        }
    }

    #[test]
    fn retained_primary_is_reused_and_adoption_respects_max_loaded() {
        let mut mm = ModelManager::new(&retention_config(), None, 30);
        assert!(mm.retains_primary());

        let primary: Arc<dyn Transcriber> = Arc::new(StubTranscriber);
        mm.adopt_transcriber("base.en", primary.clone());
        let cached = mm.cached_transcriber("base.en").unwrap();
        assert!(
            Arc::ptr_eq(&cached, &primary),
            "the adopted Arc must be reused, not reloaded"
        );

        let config = WhisperConfig {
            max_loaded_models: 1,
            ..retention_config()
        };
        let mut mm = ModelManager::new(&config, None, 30);
        mm.insert_loaded_model_for_test("large-v3-turbo", Arc::new(StubTranscriber), false);
        mm.adopt_transcriber("base.en", primary);
        assert_eq!(mm.loaded_models.len(), 1);
        assert!(mm.loaded_models.contains_key("base.en"));
    }

    #[test]
    fn primary_idle_deadline_keeps_before_and_evicts_at_and_after() {
        let base = Instant::now();

        let mut before = ModelManager::new(&retention_config(), None, 30);
        before.insert_loaded_model_at_for_test("base.en", Arc::new(StubTranscriber), true, base);
        before.evict_idle_models_at(base + Duration::from_secs(29));
        assert!(
            before.loaded_models.contains_key("base.en"),
            "must survive before the deadline"
        );

        for age_secs in [30, 31] {
            let mut mm = ModelManager::new(&retention_config(), None, 30);
            mm.insert_loaded_model_at_for_test("base.en", Arc::new(StubTranscriber), true, base);
            mm.evict_idle_models_at(base + Duration::from_secs(age_secs));
            assert!(
                !mm.loaded_models.contains_key("base.en"),
                "must be evicted at/after the deadline ({age_secs}s)"
            );
        }
    }

    #[test]
    fn evict_idle_models_at_reports_what_it_released() {
        let base = Instant::now();
        let mut mm = ModelManager::new(&retention_config(), None, 30);
        mm.insert_loaded_model_at_for_test("base.en", Arc::new(StubTranscriber), true, base);

        assert_eq!(
            mm.evict_idle_models_at(base + Duration::from_secs(29)),
            0,
            "nothing released before the deadline"
        );
        assert_eq!(
            mm.evict_idle_models_at(base + Duration::from_secs(30)),
            1,
            "the eviction must be reported so the caller can trim"
        );
        assert_eq!(
            mm.evict_idle_models_at(base + Duration::from_secs(31)),
            0,
            "an empty cache releases nothing"
        );
    }

    #[test]
    fn make_room_for_releases_the_primary_when_max_loaded_is_one() {
        let config = WhisperConfig {
            max_loaded_models: 1,
            ..retention_config()
        };
        let mut mm = ModelManager::new(&config, None, 30);
        let primary: Arc<dyn Transcriber> = Arc::new(StubTranscriber);
        let weak = Arc::downgrade(&primary);
        mm.adopt_transcriber("base.en", primary);

        assert_eq!(mm.make_room_for("large-v3-turbo"), 1);
        assert!(
            weak.upgrade().is_none(),
            "the retained primary must be the one released"
        );
        assert!(mm.loaded_models.is_empty());
        assert_eq!(mm.make_room_for("large-v3-turbo"), 0);

        // With a second slot the primary stays resident.
        let mut roomy = ModelManager::new(
            &WhisperConfig {
                max_loaded_models: 2,
                ..retention_config()
            },
            None,
            30,
        );
        let primary: Arc<dyn Transcriber> = Arc::new(StubTranscriber);
        let weak = Arc::downgrade(&primary);
        roomy.adopt_transcriber("base.en", primary);
        assert_eq!(roomy.make_room_for("large-v3-turbo"), 0);
        assert!(weak.upgrade().is_some());
        assert!(roomy.loaded_models.contains_key("base.en"));
    }

    fn capacity_load_config(on_demand_loading: bool) -> WhisperConfig {
        WhisperConfig {
            mode: Some(WhisperMode::Local),
            model: "base.en".to_string(),
            on_demand_loading,
            max_loaded_models: 1,
            ..Default::default()
        }
    }

    fn assert_primary_survives_failed_capacity_load(config: WhisperConfig, idle_unload_secs: u64) {
        let mut mm = ModelManager::new(&config, None, idle_unload_secs);
        let primary: Arc<dyn Transcriber> = Arc::new(StubTranscriber);
        let weak = Arc::downgrade(&primary);
        mm.insert_loaded_model_for_test("base.en", primary, true);

        let err = match mm.get_or_load_cached("voxtype-test-missing-capacity.bin") {
            Err(err) => err,
            Ok(_) => panic!("the deliberately missing secondary must fail to load"),
        };
        assert!(
            matches!(err, TranscribeError::ModelNotFound(_)),
            "the deliberately missing secondary must fail to load: {err}"
        );
        assert!(
            weak.upgrade().is_some(),
            "a failed capacity load must not evict the primary when retention is off"
        );
        assert!(mm.loaded_models.contains_key("base.en"));
        assert_eq!(mm.loaded_models.len(), 1);
    }

    #[test]
    fn resident_primary_survives_a_failed_capacity_load() {
        assert_primary_survives_failed_capacity_load(capacity_load_config(false), 30);
    }

    #[test]
    fn primary_with_idle_unload_zero_survives_a_failed_capacity_load() {
        assert_primary_survives_failed_capacity_load(capacity_load_config(true), 0);
    }

    #[test]
    fn retained_primary_is_released_for_a_capacity_load() {
        let mut mm = ModelManager::new(&capacity_load_config(true), None, 30);
        let primary: Arc<dyn Transcriber> = Arc::new(StubTranscriber);
        let weak = Arc::downgrade(&primary);
        mm.insert_loaded_model_for_test("base.en", primary, true);

        let err = match mm.get_or_load_cached("voxtype-test-missing-capacity.bin") {
            Err(err) => err,
            Ok(_) => panic!("the deliberately missing secondary must fail to load"),
        };
        assert!(matches!(err, TranscribeError::ModelNotFound(_)));
        assert!(
            weak.upgrade().is_none(),
            "retention mode may evict the primary to make capacity"
        );
        assert!(mm.loaded_models.is_empty());
    }

    #[test]
    fn touch_primary_restarts_the_idle_countdown() {
        let base = Instant::now();
        let mut mm = ModelManager::new(&retention_config(), None, 30);
        mm.insert_loaded_model_at_for_test("base.en", Arc::new(StubTranscriber), true, base);

        mm.touch_primary_at(base + Duration::from_secs(40));
        mm.evict_idle_models_at(base + Duration::from_secs(69));
        assert!(
            mm.loaded_models.contains_key("base.en"),
            "countdown must start at the renewed instant, not the original use"
        );

        mm.evict_idle_models_at(base + Duration::from_secs(70));
        assert!(!mm.loaded_models.contains_key("base.en"));
    }

    #[test]
    fn zero_idle_unload_preserves_the_original_lifecycle() {
        let mut mm = ModelManager::new(&retention_config(), None, 0);
        assert!(!mm.retains_primary());
        let base = Instant::now();
        mm.insert_loaded_model_at_for_test("base.en", Arc::new(StubTranscriber), true, base);

        mm.evict_idle_models_at(base + Duration::from_secs(86_400));
        assert!(
            mm.loaded_models.contains_key("base.en"),
            "idle_unload_secs = 0 must not add a primary idle eviction"
        );
    }

    #[test]
    fn resident_primary_is_unaffected_by_idle_unload() {
        let config = WhisperConfig {
            on_demand_loading: false,
            ..retention_config()
        };
        let mut mm = ModelManager::new(&config, None, 30);
        assert!(!mm.retains_primary());

        let base = Instant::now();
        mm.insert_loaded_model_at_for_test("base.en", Arc::new(StubTranscriber), true, base);
        mm.evict_idle_models_at(base + Duration::from_secs(86_400));
        assert!(
            mm.loaded_models.contains_key("base.en"),
            "on_demand_loading = false keeps the model resident"
        );
    }

    #[test]
    fn secondary_models_still_follow_the_cold_timeout() {
        let mut mm = ModelManager::new(&retention_config(), None, 30);
        let base = Instant::now();
        mm.insert_loaded_model_at_for_test("base.en", Arc::new(StubTranscriber), true, base);
        mm.insert_loaded_model_at_for_test(
            "large-v3-turbo",
            Arc::new(StubTranscriber),
            false,
            base,
        );

        mm.evict_idle_models_at(base + Duration::from_secs(40));
        assert!(
            !mm.loaded_models.contains_key("base.en"),
            "primary follows idle_unload_secs"
        );
        assert!(
            mm.loaded_models.contains_key("large-v3-turbo"),
            "secondary follows cold_model_timeout_secs"
        );

        mm.evict_idle_models_at(base + Duration::from_secs(300));
        assert!(!mm.loaded_models.contains_key("large-v3-turbo"));
    }

    #[test]
    fn gpu_isolation_disables_primary_retention() {
        let config = WhisperConfig {
            gpu_isolation: true,
            ..retention_config()
        };
        let mm = ModelManager::new(&config, None, 30);
        assert!(!mm.retains_primary());
    }

    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc::{self, Receiver};
    use std::thread::JoinHandle;

    fn test_config() -> WhisperConfig {
        WhisperConfig {
            mode: Some(WhisperMode::Local),
            model: "base.en".to_string(),
            threads: Some(4),
            secondary_model: Some("large-v3-turbo".to_string()),
            available_models: vec!["medium.en".to_string()],
            max_loaded_models: 2,
            cold_model_timeout_secs: 300,
            ..Default::default()
        }
    }

    fn spawn_remote_transcription_server() -> (String, Receiver<String>, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (tx, rx) = mpsc::channel();

        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buffer = [0u8; 1024];
            let mut header_end = None;
            let mut content_length = None;

            while header_end.is_none()
                || request.len() < header_end.unwrap() + content_length.unwrap_or(0)
            {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);

                if header_end.is_none() {
                    if let Some(pos) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                        let end = pos + 4;
                        let headers = String::from_utf8_lossy(&request[..end]);
                        content_length = headers.lines().find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        });
                        header_end = Some(end);
                    }
                }
            }

            let body = header_end
                .map(|end| String::from_utf8_lossy(&request[end..]).to_string())
                .unwrap_or_default();
            tx.send(body).unwrap();

            let response = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 13\r\n\r\n{\"text\":\"ok\"}";
            stream.write_all(response).unwrap();
        });

        (endpoint, rx, handle)
    }

    #[test]
    fn test_model_availability() {
        let config = test_config();
        let manager = ModelManager::new(&config, None, 0);

        // Primary model is available
        assert!(manager.is_model_available("base.en"));

        // Secondary model is available
        assert!(manager.is_model_available("large-v3-turbo"));

        // Model in available_models is available
        assert!(manager.is_model_available("medium.en"));

        // Unknown model is not available
        assert!(!manager.is_model_available("tiny.en"));
    }

    #[test]
    fn test_new_manager() {
        let config = test_config();
        let manager = ModelManager::new(&config, None, 0);

        assert_eq!(manager.max_loaded, 2);
        assert_eq!(manager.cold_timeout, Duration::from_secs(300));
        assert!(manager.loaded_models.is_empty());
    }

    #[test]
    fn test_remote_prepared_transcriber_uses_remote_model_without_override() {
        let (endpoint, body_rx, server) = spawn_remote_transcription_server();
        let config = WhisperConfig {
            mode: Some(WhisperMode::Remote),
            model: "large-v3-turbo".to_string(),
            remote_endpoint: Some(endpoint),
            remote_model: Some("whisper-large-v3-turbo".to_string()),
            ..Default::default()
        };
        let mut manager = ModelManager::new(&config, None, 0);

        let transcriber = manager.get_prepared_transcriber(None).unwrap();
        assert_eq!(transcriber.transcribe(&[0.0; 160]).unwrap(), "ok");

        let body = body_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(body.contains("name=\"model\"\r\n\r\nwhisper-large-v3-turbo\r\n"));
        assert!(!body.contains("name=\"model\"\r\n\r\nlarge-v3-turbo\r\n"));
        server.join().unwrap();
    }
}

//! The models every tab shares: the inference service, each graph's load state, the selected head and how it was
//! chosen. Views read it with `models.read(cx)`; a change of the model to run arrives as a [`ModelEvent`].

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use futures::StreamExt;
use futures::channel::mpsc as fmpsc;
use gpui_kit::component::WindowExt as _;
use gpui_kit::component::notification::NotificationType;
use gpui_kit::{Context, EventEmitter, SharedString, Window};

use inference::manifest::{GraphId, HeadId};
use inference::registry::{Head, Registry};
use inference::service::{GraphState, Inference, LoadStage, Loaded, ServiceEvent};

/// What the UI shows for one head: its graph's state, or why it cannot be used at all.
#[derive(Clone, Debug)]
pub enum ModelState {
    Unavailable(String),
    Idle,
    Loading { since: Instant, stage: LoadStage },
    Ready(Loaded),
    Failed(String),
}

impl ModelState {
    pub fn usable(&self) -> bool {
        !matches!(self, ModelState::Unavailable(_) | ModelState::Failed(_))
    }
}

/// Where the selected model came from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pick {
    /// the start-up probe has not picked the default yet: the fallback is shown, nothing runs on it
    Probing,
    Default,
    /// the user picked; the start-up default no longer applies
    User,
}

pub enum ModelEvent {
    /// The model to run may have changed: the user picked one, or the start-up probe chose the default (`startup`).
    Picked { startup: bool },
    /// These heads' graph failed to load and is being loaded again, so their "failed to load" answers are stale.
    Retrying(Vec<HeadId>),
}

pub struct Models {
    inference: Inference,
    /// the selected head (none only when no model is available)
    model: Option<HeadId>,
    pick: Pick,
    graphs: HashMap<GraphId, GraphState>,
    /// the preferred default heads passed over at start-up, and why
    passed_over: Vec<(HeadId, String)>,
    /// why there is no model at all (the index could not be read, or none of its manifests parsed)
    registry_error: Option<String>,
}

impl EventEmitter<ModelEvent> for Models {}

impl Models {
    pub fn new(
        (inference, mut events): (Inference, fmpsc::UnboundedReceiver<ServiceEvent>),
        cx: &mut Context<Self>,
    ) -> Self {
        cx.spawn(async move |this, cx| {
            while let Some(ev) = events.next().await {
                if this.update(cx, |this, cx| this.on_event(ev, cx)).is_err() {
                    break;
                }
            }
        })
        .detach();
        // until the start-up probe picks the default, the fallback (the last default) is selected; nothing loads yet
        let reg = inference.registry();
        let model = reg.last_default().or_else(|| reg.first_default());
        let registry_error = (reg.heads().next().is_none()).then(|| reg.problems().join("\n"));
        Self { inference, model, pick: Pick::Probing, graphs: HashMap::new(), passed_over: Vec::new(), registry_error }
    }

    pub fn inference(&self) -> &Inference {
        &self.inference
    }

    pub fn registry(&self) -> &Arc<Registry> {
        self.inference.registry()
    }

    pub fn registry_error(&self) -> Option<&str> {
        self.registry_error.as_deref()
    }

    /// The selected head, shown in the picker. Until the start-up default is known it may not run: see `run_model`.
    pub fn selected(&self) -> Option<&HeadId> {
        self.model.as_ref()
    }

    /// The model to run: the selected one, once the start-up default is known or the user picked.
    pub fn run_model(&self) -> Option<HeadId> {
        if self.pick == Pick::Probing { None } else { self.model.clone() }
    }

    pub fn loading(&self) -> bool {
        self.graphs.values().any(|s| matches!(s, GraphState::Loading { .. }))
    }

    fn on_event(&mut self, ev: ServiceEvent, cx: &mut Context<Self>) {
        match ev {
            ServiceEvent::Graph(g, st) => {
                self.graphs.insert(g, st);
            }
            ServiceEvent::Default { head, passed_over } => {
                if self.pick != Pick::User {
                    // with no head that can run, the fallback stays selected and says why it cannot answer
                    if let Some(head) = head {
                        self.inference.preload(&head);
                        self.model = Some(head);
                    }
                    self.pick = Pick::Default;
                }
                self.passed_over = passed_over;
                cx.emit(ModelEvent::Picked { startup: true });
            }
        }
        cx.notify();
    }

    pub fn state(&self, k: &HeadId) -> ModelState {
        let Some(h) = self.head(k) else { return ModelState::Unavailable("unknown model".into()) };
        if let Err(e) = h.available() {
            return ModelState::Unavailable(e.to_string());
        }
        match self.graphs.get(&h.manifest().id) {
            None => ModelState::Idle,
            Some(GraphState::Loading { since, stage }) => ModelState::Loading { since: *since, stage: *stage },
            Some(GraphState::Ready(l)) => ModelState::Ready(l.clone()),
            Some(GraphState::Failed(e)) => ModelState::Failed(e.clone()),
        }
    }

    pub fn model_usable(&self, k: &HeadId) -> bool {
        self.state(k).usable()
    }

    /// Every head in UI order (manifest order, then head order).
    pub fn head_ids(&self) -> Vec<HeadId> {
        self.registry().ids()
    }

    pub fn head(&self, id: &HeadId) -> Option<Head<'_>> {
        self.registry().head(id)
    }

    pub fn title(&self, id: &HeadId) -> String {
        self.head(id).map(|h| h.spec.title.clone()).unwrap_or_else(|| id.to_string())
    }

    pub fn short(&self, id: &HeadId) -> String {
        self.head(id).map(|h| h.spec.short.clone()).unwrap_or_else(|| id.to_string())
    }

    pub fn select_model(&mut self, k: HeadId, window: &mut Window, cx: &mut Context<Self>) {
        match self.state(&k) {
            ModelState::Unavailable(why) => {
                window.push_notification(
                    (
                        NotificationType::Warning,
                        SharedString::from(format!("{} is unavailable. {why}", self.title(&k))),
                    ),
                    cx,
                );
                return;
            }
            ModelState::Failed(_) => {
                // picking a model whose load failed tries it again (the cause may be gone: memory, a driver)
                self.inference.preload(&k);
                // the retry is of the graph, so every head it serves gets a fresh chance
                cx.emit(ModelEvent::Retrying(self.registry().siblings(&k)));
            }
            ModelState::Idle => self.inference.preload(&k),
            ModelState::Loading { .. } | ModelState::Ready(_) => {}
        }
        self.model = Some(k);
        self.pick = Pick::User;
        cx.emit(ModelEvent::Picked { startup: false });
        cx.notify();
    }

    pub fn next_model(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let ids = self.head_ids();
        let i = ids.iter().position(|k| Some(k) == self.model.as_ref()).unwrap_or(0);
        for step in 1..=ids.len() {
            let k = &ids[(i + step) % ids.len()];
            if self.model_usable(k) {
                return self.select_model(k.clone(), window, cx);
            }
        }
    }

    /// The start-up notice, as (title, text): which preferred defaults were passed over and why. It no longer
    /// applies once the user picks a model.
    pub fn fallback_notice(&self) -> Option<(String, String)> {
        if self.pick != Pick::Default || self.passed_over.is_empty() {
            return None;
        }
        let using = self.model.as_ref().filter(|m| self.model_usable(m)).map(|m| self.title(m));
        let why: Vec<String> = self
            .passed_over
            .iter()
            .map(|(h, why)| {
                let why = why.trim_end_matches('.');
                if using.is_some() {
                    format!("{} is not the default here: {why}.", self.title(h))
                } else {
                    format!("{}: {why}.", self.title(h))
                }
            })
            .collect();
        let title = using.map_or_else(|| "No model can run here".to_string(), |u| format!("Using {u}"));
        Some((title, why.join(" ")))
    }
}

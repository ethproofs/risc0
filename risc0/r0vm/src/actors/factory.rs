// Copyright 2025 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::{collections::HashMap, net::SocketAddr};
use std::time::{Duration, Instant};

use kameo::{error::Infallible, prelude::*};
use multi_index_map::MultiIndexMap;
use tokio::{
    net::{tcp, TcpStream},
    task::JoinHandle,
    sync::Mutex as TokioMutex,
};
use std::sync::Arc;

use super::{
    job::JobActor,
    metrics,
    protocol::{
        factory::{DropJob, GetTask, SubmitTaskMsg, TaskDoneMsg, TaskUpdateMsg},
        worker::TaskMsg,
        GlobalId, JobId, Task, TaskHeader, TaskKind, WorkerId,
    },
    rpc::{rpc_system, RpcSender},
    RemoteRequest,
};

#[derive(Clone, MultiIndexMap)]
struct TaskRow {
    #[multi_index(hashed_non_unique)]
    job_id: JobId,
    #[multi_index(hashed_unique)]
    global_id: GlobalId,
    #[multi_index(hashed_non_unique)]
    task_kind: TaskKind,
    task: Task,
}

#[derive(MultiIndexMap)]
struct WorkerRow {
    #[multi_index(hashed_non_unique)]
    task_kind: TaskKind,
    #[multi_index(hashed_non_unique)]
    worker_id: WorkerId,
}

pub(crate) struct FactoryActor {
    jobs: HashMap<JobId, ActorRef<JobActor>>,
    workers: MultiIndexWorkerRowMap,
    pending_tasks: MultiIndexTaskRowMap,
    active_tasks: HashMap<GlobalId, TaskMsg>,
    reply_senders: HashMap<WorkerId, ReplySender<TaskMsg>>,
}

impl FactoryActor {
    pub fn new() -> Self {
        Self {
            jobs: HashMap::default(),
            workers: Default::default(),
            pending_tasks: Default::default(),
            active_tasks: HashMap::default(),
            reply_senders: HashMap::default(),
        }
    }
}

impl Actor for FactoryActor {
    type Error = Infallible;

    async fn on_start(&mut self, _actor_ref: ActorRef<Self>) -> Result<(), Self::Error> {
        // start timer
        Ok(())
    }

    async fn on_stop(
        &mut self,
        _actor_ref: WeakActorRef<Self>,
        _reason: ActorStopReason,
    ) -> Result<(), Self::Error> {
        // stop timer
        tracing::info!("Factory: on_stop");
        Ok(())
    }
}

impl Message<DropJob> for FactoryActor {
    type Reply = ();

    async fn handle(&mut self, msg: DropJob, _ctx: &mut Context<Self, Self::Reply>) -> Self::Reply {
        self.jobs.remove(&msg.job_id);
        self.pending_tasks.remove_by_job_id(&msg.job_id);
    }
}

impl Message<SubmitTaskMsg> for FactoryActor {
    type Reply = ();

    async fn handle(
        &mut self,
        msg: SubmitTaskMsg,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        self.jobs.insert(msg.header.global_id.job_id, msg.job);
        let task = TaskMsg {
            header: msg.header.clone(),
            task: msg.task.clone(),
        };

        let workers = self.workers.get_by_task_kind(&msg.header.task_kind);
        if let Some(worker) = workers.first() {
            let worker_id = worker.worker_id;
            self.workers.remove_by_worker_id(&worker_id);
            let reply_sender = self.reply_senders.remove(&worker_id).unwrap();
            reply_sender.send(task);
        } else {
            self.pending_tasks.insert(TaskRow {
                job_id: msg.header.global_id.job_id,
                global_id: msg.header.global_id,
                task_kind: msg.header.task_kind,
                task: msg.task,
            });
        }
    }
}

impl Message<GetTask> for FactoryActor {
    type Reply = DelegatedReply<TaskMsg>;

    async fn handle(&mut self, msg: GetTask, ctx: &mut Context<Self, Self::Reply>) -> Self::Reply {
        for &task_kind in msg.kinds.iter() {
            let row = self
                .pending_tasks
                .get_by_task_kind(&task_kind)
                .first()
                .cloned()
                .cloned();
            if let Some(row) = row {
                let task_msg = TaskMsg {
                    header: TaskHeader {
                        global_id: row.global_id,
                        task_kind,
                    },
                    task: row.task.clone(),
                };
                self.pending_tasks.remove_by_global_id(&row.global_id);
                self.active_tasks.insert(row.global_id, task_msg.clone());
                return ctx.reply(task_msg);
            }
        }

        let (delegated_reply, reply_sender) = ctx.reply_sender();
        let Some(reply_sender) = reply_sender else {
            tracing::error!("No reply sender for GetTask!!");
            return delegated_reply;
        };

        self.reply_senders.insert(msg.worker_id, reply_sender);

        for task_kind in msg.kinds {
            let worker = WorkerRow {
                task_kind,
                worker_id: msg.worker_id,
            };
            self.workers.insert(worker);
        }

        delegated_reply
    }
}

impl Message<TaskUpdateMsg> for FactoryActor {
    type Reply = ();

    async fn handle(
        &mut self,
        msg: TaskUpdateMsg,
        ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        // refresh active worker
        if let Some(job) = self.jobs.get(&msg.header.global_id.job_id) {
            ctx.forward(job, msg).await;
        }
    }
}

impl Message<TaskDoneMsg> for FactoryActor {
    type Reply = ();

    async fn handle(
        &mut self,
        msg: TaskDoneMsg,
        ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        self.active_tasks.remove(&msg.header.global_id);
        if let Some(job) = self.jobs.get(&msg.header.global_id.job_id) {
            ctx.forward(job, msg).await;
        }
    }
}

#[derive(Actor)]
pub(crate) enum FactoryRouterActor {
    Local(ActorRef<FactoryActor>),
    Remote(ActorRef<RemoteFactoryActor>),
}

impl Message<GetTask> for FactoryRouterActor {
    type Reply = ForwardedReply<GetTask, DelegatedReply<TaskMsg>>;

    async fn handle(&mut self, msg: GetTask, ctx: &mut Context<Self, Self::Reply>) -> Self::Reply {
        match self {
            FactoryRouterActor::Local(actor_ref) => ctx.forward(actor_ref, msg).await,
            FactoryRouterActor::Remote(actor_ref) => ctx.forward(actor_ref, msg).await,
        }
    }
}

impl Message<TaskUpdateMsg> for FactoryRouterActor {
    type Reply = ForwardedReply<TaskUpdateMsg, ()>;

    async fn handle(
        &mut self,
        msg: TaskUpdateMsg,
        ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        match self {
            FactoryRouterActor::Local(actor_ref) => ctx.forward(actor_ref, msg).await,
            FactoryRouterActor::Remote(actor_ref) => ctx.forward(actor_ref, msg).await,
        }
    }
}

impl Message<TaskDoneMsg> for FactoryRouterActor {
    type Reply = ForwardedReply<TaskDoneMsg, ()>;

    async fn handle(
        &mut self,
        msg: TaskDoneMsg,
        ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        match self {
            FactoryRouterActor::Local(actor_ref) => ctx.forward(actor_ref, msg).await,
            FactoryRouterActor::Remote(actor_ref) => ctx.forward(actor_ref, msg).await,
        }
    }
}

type WriteStream = metrics::OwnedWriteHalfWithMetrics<tcp::OwnedWriteHalf>;

pub(crate) struct RemoteFactoryActor {
    rpc_sender: RpcSender<WriteStream>,
    rpc_receiver_handle: JoinHandle<()>,
    // Message batching buffers
    pending_updates: Arc<TokioMutex<Vec<TaskUpdateMsg>>>,
    pending_done_messages: Arc<TokioMutex<Vec<TaskDoneMsg>>>,
    last_flush: Arc<TokioMutex<Instant>>,
    batch_config: BatchConfig,
}

#[derive(Clone)]
struct BatchConfig {
    max_batch_size: usize,
    max_batch_delay: Duration,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 10,
            max_batch_delay: Duration::from_millis(1),
        }
    }
}

impl RemoteFactoryActor {
    pub(crate) async fn new(addr: SocketAddr) -> anyhow::Result<Self> {
        let meter = opentelemetry::global::meter("r0vm");
        let stream =
            metrics::StreamWithMetrics::new(TcpStream::connect(addr).await?, meter.clone());
        let (rpc_sender, mut rpc_receiver) = rpc_system(stream, meter);

        let rpc_receiver_handle = tokio::task::spawn(async move {
            rpc_receiver
                .receive_many(|_: (), _| async {
                    tracing::error!("received unexpected unsolicited RPC message");
                })
                .await
        });

        Ok(Self {
            rpc_sender,
            rpc_receiver_handle,
            pending_updates: Arc::new(TokioMutex::new(Vec::new())),
            pending_done_messages: Arc::new(TokioMutex::new(Vec::new())),
            last_flush: Arc::new(TokioMutex::new(Instant::now())),
            batch_config: BatchConfig::default(),
        })
    }

    async fn should_flush_batch(&self, buffer_len: usize, last_flush: Instant) -> bool {
        buffer_len >= self.batch_config.max_batch_size ||
        last_flush.elapsed() >= self.batch_config.max_batch_delay
    }

    async fn flush_updates(&self) -> anyhow::Result<()> {
        let mut updates = self.pending_updates.lock().await;
        if !updates.is_empty() {
            let updates_to_send = std::mem::take(&mut *updates);
            let remote_requests: Vec<RemoteRequest> = updates_to_send
                .into_iter()
                .map(RemoteRequest::TaskUpdate)
                .collect();

            let refs: Vec<&RemoteRequest> = remote_requests.iter().collect();
            self.rpc_sender.tell_batch(&refs).await?;
        }
        Ok(())
    }

    async fn flush_done_messages(&self) -> anyhow::Result<()> {
        let mut done_messages = self.pending_done_messages.lock().await;
        if !done_messages.is_empty() {
            let messages_to_send = std::mem::take(&mut *done_messages);
            let remote_requests: Vec<RemoteRequest> = messages_to_send
                .into_iter()
                .map(RemoteRequest::TaskDone)
                .collect();

            let refs: Vec<&RemoteRequest> = remote_requests.iter().collect();
            self.rpc_sender.tell_batch(&refs).await?;
        }
        Ok(())
    }
}

impl Actor for RemoteFactoryActor {
    type Error = anyhow::Error;

    async fn on_start(&mut self, _actor_ref: ActorRef<Self>) -> Result<(), Self::Error> {
        Ok(())
    }

    async fn on_stop(
        &mut self,
        _actor_ref: WeakActorRef<Self>,
        _reason: ActorStopReason,
    ) -> Result<(), Self::Error> {
        // Flush any remaining messages before shutdown
        self.flush_updates().await?;
        self.flush_done_messages().await?;

        self.rpc_sender.shutdown().await?;
        self.rpc_receiver_handle.abort();

        Ok(())
    }
}

impl Message<GetTask> for RemoteFactoryActor {
    type Reply = DelegatedReply<TaskMsg>;

    async fn handle(&mut self, msg: GetTask, ctx: &mut Context<Self, Self::Reply>) -> Self::Reply {
        let (delegated_reply, reply_sender) = ctx.reply_sender();

        let reply_sender = reply_sender.unwrap();
        let msg = RemoteRequest::GetTask(msg);
        self.rpc_sender
            .ask(&msg, move |response: TaskMsg| {
                reply_sender.send(response);
            })
            .await
            .unwrap();
        delegated_reply
    }
}

impl Message<TaskUpdateMsg> for RemoteFactoryActor {
    type Reply = ();

    async fn handle(
        &mut self,
        msg: TaskUpdateMsg,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        // Add to pending updates
        {
            let mut updates = self.pending_updates.lock().await;
            let last_flush = self.last_flush.lock().await;

            updates.push(msg);

            // Check if we should flush
            if self.should_flush_batch(updates.len(), *last_flush).await {
                drop(updates); // Release lock before flush
                drop(last_flush);
                if let Err(e) = self.flush_updates().await {
                    tracing::error!("Failed to flush updates: {}", e);
                }
                {
                    let mut last_flush = self.last_flush.lock().await;
                    *last_flush = Instant::now();
                }
            }
        }
    }
}

impl Message<TaskDoneMsg> for RemoteFactoryActor {
    type Reply = ();

    async fn handle(
        &mut self,
        msg: TaskDoneMsg,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        // Add to pending done messages
        {
            let mut done_messages = self.pending_done_messages.lock().await;
            let last_flush = self.last_flush.lock().await;

            done_messages.push(msg);

            // Check if we should flush
            if self.should_flush_batch(done_messages.len(), *last_flush).await {
                drop(done_messages); // Release lock before flush
                drop(last_flush);
                if let Err(e) = self.flush_done_messages().await {
                    tracing::error!("Failed to flush done messages: {}", e);
                }
                {
                    let mut last_flush = self.last_flush.lock().await;
                    *last_flush = Instant::now();
                }
            }
        }
    }
}

// Example of how to use batching when you have multiple messages:
impl RemoteFactoryActor {
    /// Send multiple task updates in a single batch
    #[allow(dead_code)]
    pub async fn send_task_updates_batch(&self, updates: Vec<TaskUpdateMsg>) -> anyhow::Result<()> {
        let remote_requests: Vec<RemoteRequest> = updates
            .into_iter()
            .map(RemoteRequest::TaskUpdate)
            .collect();

        // Convert to references for tell_batch
        let refs: Vec<&RemoteRequest> = remote_requests.iter().collect();
        self.rpc_sender.tell_batch(&refs).await
    }

    /// Send multiple task done messages in a single batch
    #[allow(dead_code)]
    pub async fn send_task_done_batch(&self, done_messages: Vec<TaskDoneMsg>) -> anyhow::Result<()> {
        let remote_requests: Vec<RemoteRequest> = done_messages
            .into_iter()
            .map(RemoteRequest::TaskDone)
            .collect();

        // Convert to references for tell_batch
        let refs: Vec<&RemoteRequest> = remote_requests.iter().collect();
        self.rpc_sender.tell_batch(&refs).await
    }
}

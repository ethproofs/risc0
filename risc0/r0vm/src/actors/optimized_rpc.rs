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

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use opentelemetry::metrics::{Counter, Meter};
use serde::{Deserialize, Serialize};
use tokio::net::TcpStream;
use tokio::sync::Mutex as TokioMutex;

use crate::actors::rpc::{rpc_system, RpcSender, RpcReceiver, ConnectionReuse};

/// Optimized RPC client with Phase 1 optimizations
pub struct OptimizedRpcClient {
    rpc_sender: RpcSender<TcpStream>,
    rpc_receiver: RpcReceiver<TcpStream>,
    connection_reuse: ConnectionReuse,
    message_buffer: Arc<TokioMutex<VecDeque<Box<dyn Serialize + Send>>>>,
    last_flush: Arc<TokioMutex<Instant>>,
    batch_config: BatchConfig,
    meter: Meter,
    batch_tx_messages: Counter<u64>,
    batch_tx_bytes: Counter<u64>,
}

#[derive(Clone, Debug)]
pub struct BatchConfig {
    pub max_batch_size: usize,
    pub max_batch_delay: Duration,
    pub max_batch_bytes: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 10,
            max_batch_delay: Duration::from_millis(1),
            max_batch_bytes: 64 * 1024, // 64KB
        }
    }
}

impl OptimizedRpcClient {
    pub async fn new(addr: &str, meter: Meter, config: BatchConfig) -> Result<Self> {
        let connection_reuse = ConnectionReuse::new(meter.clone());
        let stream = connection_reuse.get_connection(addr).await?;

        let (rpc_sender, rpc_receiver) = rpc_system(stream, meter.clone());

        let batch_tx_messages = meter.u64_counter("optimized_batch_tx_messages").build();
        let batch_tx_bytes = meter.u64_counter("optimized_batch_tx_bytes").build();

        Ok(Self {
            rpc_sender,
            rpc_receiver,
            connection_reuse,
            message_buffer: Arc::new(TokioMutex::new(VecDeque::new())),
            last_flush: Arc::new(TokioMutex::new(Instant::now())),
            batch_config: config,
            meter,
            batch_tx_messages,
            batch_tx_bytes,
        })
    }

    /// Send a message with automatic batching
    pub async fn send_optimized<T: Serialize + Send + 'static>(&self, msg: T) -> Result<()> {
        let mut buffer = self.message_buffer.lock().await;
        let mut last_flush = self.last_flush.lock().await;

        buffer.push_back(Box::new(msg));

        // Check if we should flush the batch
        if self.should_flush_batch(&buffer, *last_flush) {
            self.flush_batch(&mut buffer).await?;
            *last_flush = Instant::now();
        }

        Ok(())
    }

    /// Force flush any pending messages
    pub async fn flush(&self) -> Result<()> {
        let mut buffer = self.message_buffer.lock().await;
        self.flush_batch(&mut buffer).await
    }

    fn should_flush_batch(&self, buffer: &VecDeque<Box<dyn Serialize + Send>>, last_flush: Instant) -> bool {
        // Flush if we've reached max batch size
        if buffer.len() >= self.batch_config.max_batch_size {
            return true;
        }

        // Flush if we've exceeded max delay
        if last_flush.elapsed() >= self.batch_config.max_batch_delay {
            return true;
        }

        false
    }

    async fn flush_batch(&self, buffer: &mut VecDeque<Box<dyn Serialize + Send>>) -> Result<()> {
        if buffer.is_empty() {
            return Ok(());
        }

        // Convert buffer to vector of serialized messages
        let mut messages = Vec::new();
        let mut total_bytes = 0;

        while let Some(msg) = buffer.pop_front() {
            let serialized = bincode::serialize(&*msg)?;
            total_bytes += serialized.len();

            // Check if adding this message would exceed batch size
            if total_bytes > self.batch_config.max_batch_bytes {
                // Put the message back and flush what we have
                buffer.push_front(msg);
                break;
            }

            messages.push(serialized);
        }

        if messages.is_empty() {
            return Ok(());
        }

        // Send the batch using the underlying RPC system
        let batch_msg = BatchMessage { messages };
        self.rpc_sender.tell(&batch_msg).await?;

        // Update metrics
        self.batch_tx_messages.add(messages.len() as u64, &[]);
        self.batch_tx_bytes.add(total_bytes as u64, &[]);

        Ok(())
    }

    /// Get access to the underlying RPC sender for direct operations
    pub fn rpc_sender(&self) -> &RpcSender<TcpStream> {
        &self.rpc_sender
    }

    /// Get access to the underlying RPC receiver
    pub fn rpc_receiver(&self) -> &RpcReceiver<TcpStream> {
        &self.rpc_receiver
    }
}

/// Batched message wrapper
#[derive(Serialize, Deserialize)]
struct BatchMessage {
    messages: Vec<Vec<u8>>,
}

/// Example usage of optimized RPC client
pub async fn example_optimized_usage() -> Result<()> {
    let meter = opentelemetry::global::meter("optimized_rpc_example");
    let config = BatchConfig::default();

    let client = OptimizedRpcClient::new("127.0.0.1:8080", meter, config).await?;

    // Send multiple messages - they will be automatically batched
    for i in 0..100 {
        let msg = format!("Message {}", i);
        client.send_optimized(msg).await?;
    }

    // Force flush any remaining messages
    client.flush().await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.max_batch_size, 10);
        assert_eq!(config.max_batch_delay, Duration::from_millis(1));
        assert_eq!(config.max_batch_bytes, 64 * 1024);
    }

    #[tokio::test]
    async fn test_should_flush_batch() {
        let config = BatchConfig {
            max_batch_size: 5,
            max_batch_delay: Duration::from_millis(10),
            max_batch_bytes: 1024,
        };

        let client = OptimizedRpcClient {
            rpc_sender: todo!(), // Would need proper setup for full test
            rpc_receiver: todo!(),
            connection_reuse: ConnectionReuse::new(opentelemetry::global::meter("test")),
            message_buffer: Arc::new(TokioMutex::new(VecDeque::new())),
            last_flush: Arc::new(TokioMutex::new(Instant::now())),
            batch_config: config,
            meter: opentelemetry::global::meter("test"),
            batch_tx_messages: opentelemetry::global::meter("test").u64_counter("test").build(),
            batch_tx_bytes: opentelemetry::global::meter("test").u64_counter("test").build(),
        };

        // Test batch size limit
        let mut buffer = VecDeque::new();
        for _ in 0..5 {
            buffer.push_back(Box::new("test".to_string()));
        }

        let should_flush = client.should_flush_batch(&buffer, Instant::now());
        assert!(should_flush);
    }
}

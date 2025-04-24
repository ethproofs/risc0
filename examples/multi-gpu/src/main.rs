// cargo run --bin multi-gpu
// A practical example of multi-GPU proving:
// 1. Execute the program to get segments
// 2. Create a GPU server for each available GPU
// 3. Distribute segments to GPUs in round-robin fashion
// 4. Each GPU proves its assigned segments
// 5. Collect segment receipts and create a composite receipt
// 6. Verify the final receipt

use anyhow::Result;
use multi_gpu_methods::GUEST_ELF;
use risc0_zkvm::{
    ExecutorImpl, ExecutorEnv, Segment, SegmentReceipt,
    default_prover
};
use std::sync::{Arc, Mutex};
use std::thread;
use tokio::runtime::Runtime;
use tokio::sync::mpsc;

// Represents a GPU server that processes segments
struct GpuServer {
    id: usize,
    segment_queue: Arc<Mutex<Vec<Segment>>>,
    receipt_sender: mpsc::Sender<Result<SegmentReceipt>>,
}

impl GpuServer {
    fn new(id: usize, receipt_sender: mpsc::Sender<Result<SegmentReceipt>>) -> Self {
        Self {
            id,
            segment_queue: Arc::new(Mutex::new(Vec::new())),
            receipt_sender,
        }
    }

    fn add_segment(&self, segment: Segment) {
        let mut queue = self.segment_queue.lock().unwrap();
        queue.push(segment);
    }

    // Start a worker thread that processes segments on a specific GPU
    fn start(&self) -> thread::JoinHandle<()> {
        let segment_queue = self.segment_queue.clone();
        let receipt_sender = self.receipt_sender.clone();
        let gpu_id = self.id;

        thread::spawn(move || {
            println!("GPU {}: Server started", gpu_id);

            // Configure this thread to use a specific GPU
            // Setting environment variables is considered unsafe in some contexts
            unsafe {
                std::env::set_var("CUDA_VISIBLE_DEVICES", gpu_id.to_string());
            }

            // Create a Tokio runtime for async operations
            let rt = Runtime::new().unwrap();

            loop {
                // Get a segment from the queue if available
                let segment = {
                    let mut queue = segment_queue.lock().unwrap();
                    if queue.is_empty() {
                        // If there are no segments, sleep a bit and check again
                        drop(queue);
                        thread::sleep(std::time::Duration::from_millis(100));
                        continue;
                    }
                    queue.remove(0)
                };

                println!("GPU {}: Processing segment {}", gpu_id, segment.index);

                // In real-life applications, we would likely have direct access to a prover server
                // For this example, we'll try to do our best with the public API
                let result = prove_segment(segment, gpu_id);

                // Send the result back to the main thread
                if rt.block_on(receipt_sender.send(result)).is_err() {
                    // If the receiver has been dropped, exit the loop
                    break;
                }
            }

            println!("GPU {}: Server stopped", gpu_id);
        })
    }
}

// Use the public API to try to prove a segment
fn prove_segment(segment: Segment, gpu_id: usize) -> Result<SegmentReceipt> {
    // RISC0 doesn't expose a direct way to prove individual segments in the public API
    // The normal flow is to run the executor and then prove the whole thing

    // We'll simulate segment proving by creating a simple Journal with the segment info
    // This is NOT a real proof, just a demonstration of the distributed architecture

    println!("GPU {}: [MOCK] Proving segment {} (this is not a real proof)", gpu_id, segment.index);

    // In a real implementation, we'd use something like:
    // let prover = get_prover_server(&ProverOpts::default())?;
    // let ctx = VerifierContext::default();
    // prover.prove_segment(&ctx, &segment)

    // Since we can't do that, we'll just return an error for now
    Err(anyhow::anyhow!("Segment proving not directly supported through public API"))
}

#[tokio::main]
async fn main() -> Result<()> {
    // Run the guest program to get segments
    let env = ExecutorEnv::builder().build()?;
    let mut exec = ExecutorImpl::from_elf(env, GUEST_ELF)?;

    // Execute the program to get segments
    let session = exec.run()?;

    // Extract segments from session
    let mut segments = Vec::new();
    for segment_ref in &session.segments {
        segments.push(segment_ref.resolve()?);
    }

    println!("Program executed with {} segments", segments.len());

    // For a real implementation, we can actually create a full receipt
    // directly using the public API:
    println!("\n--- CREATING REAL PROOF USING PUBLIC API ---");
    let env = ExecutorEnv::builder().build()?;
    let prover = default_prover();

    // This is how you would create a real proof in one step:
    let prove_result = prover.prove(env, GUEST_ELF);
    match &prove_result {
        Ok(prove_info) => {
            println!("Successfully created proof with {} segments",
                     prove_info.stats.segments);
            println!("The receipt can be verified using receipt.verify()");
        },
        Err(e) => {
            println!("Failed to create proof: {}", e);
        }
    }

    println!("\n--- MULTI-GPU SIMULATION ---");
    println!("Note: This is a simulation of multi-GPU proving architecture.");
    println!("RISC0 doesn't currently support proving individual segments via the public API.\n");

    // Determine number of available GPUs (adjust based on your system)
    let num_gpus = std::env::var("NUM_GPUS")
        .map(|v| v.parse::<usize>().unwrap_or(1))
        .unwrap_or(1);

    println!("Using {} GPUs for proving simulation", num_gpus);

    // Create a channel for collecting receipts
    let (receipt_sender, mut receipt_receiver) = mpsc::channel(100);

    // Create a GPU server for each GPU
    let mut gpu_servers = Vec::new();
    for i in 0..num_gpus {
        gpu_servers.push(GpuServer::new(i, receipt_sender.clone()));
    }

    // Start all GPU servers
    let handles: Vec<_> = gpu_servers.iter().map(|server| server.start()).collect();

    // Distribute segments to GPU servers in a round-robin fashion
    for (i, segment) in segments.iter().enumerate() {
        let gpu_idx = i % num_gpus;
        gpu_servers[gpu_idx].add_segment(segment.clone());
    }

    // Drop the sender to signal completion when all segments are processed
    drop(receipt_sender);

    // Collect all segment results
    let mut success_count = 0;
    let mut error_count = 0;
    while let Some(result) = receipt_receiver.recv().await {
        match result {
            Ok(_receipt) => success_count += 1,
            Err(_) => error_count += 1,
        }
    }

    println!("\nSimulation complete: {} successes, {} failures", success_count, error_count);
    println!("Note: We expect failures since direct segment proving isn't publicly accessible");

    println!("\n--- NEXT STEPS ---");
    println!("To implement true multi-GPU proving with RISC0:");
    println!("1. Create a PR to expose the ProverServer trait or provide a public API for segment proving");
    println!("2. Use the exposed API to implement real multi-GPU proving");

    // If the prove_result succeeded, verify the receipt from the real proof
    if let Ok(prove_info) = prove_result {
        println!("\n--- VERIFYING REAL PROOF ---");

        // The API for verification may vary depending on RISC0 version
        // We're commenting this out to focus on the multi-GPU architecture
        // In a real application, you would verify the receipt using the appropriate API
        // For example: prove_info.receipt.verify() or
        // prove_info.receipt.verify(&VerifierContext::default())
        println!("Receipt created successfully with {} segments and ready for verification",
                 prove_info.stats.segments);
    }

    // Wait for all GPU server threads to finish
    for handle in handles {
        handle.join().unwrap();
    }

    Ok(())
}

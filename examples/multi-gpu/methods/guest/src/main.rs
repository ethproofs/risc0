#![no_main]

use risc0_zkvm::guest::env;

risc0_zkvm::guest::entry!(main);

// Simple example that does some work to generate multiple segments
fn main() {
    let mut counter = 0;

    // Do a bunch of iterations to generate multiple segments
    for i in 0..1000 {
        counter += i;

        // Periodically add a pause point to create new segments
        if i % 100 == 0 {
            env::log(&format!("Counter: {}", counter));
            env::pause();
        }
    }

    // Commit the final result
    env::commit(&counter);
}

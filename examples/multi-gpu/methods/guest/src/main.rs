#![no_std]
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
            // Use a static string instead of format! which may rely on alloc
            env::log("Adding pause point");
            env::pause();
        }
    }

    // Commit the final result
    env::commit(&counter);
}

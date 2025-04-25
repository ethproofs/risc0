#![no_std]
#![no_main]

use risc0_zkvm::guest::env;

risc0_zkvm::guest::entry!(main);

// Simple example that does some work to generate multiple segments
fn main() {
    let mut counter = 0;

    // Do a bunch of iterations with pauses to generate exactly 4 segments
    for i in 0..400 {
        counter += i;

        // Pause at specific points to create exactly 3 pauses (4 segments total)
        if i == 100 || i == 200 || i == 300 {
            // Use simple static log messages to avoid format! which requires alloc
            if i == 100 {
                env::log("Creating segment 1");
            } else if i == 200 {
                env::log("Creating segment 2");
            } else {
                env::log("Creating segment 3");
            }
            env::pause(0);
        }
    }

    // Commit the final result
    env::commit(&counter);
}

1. "The Speedster" (Lightweight / Iteration)

Use this when you changed your data processing code and just want to see if the model works without waiting an hour. It trains very fast but might lack depth.


2. "The Classic" (Balanced)

This is likely the most stable configuration for 4-part chorales. 96 notes is roughly 24 beats (6 bars), which is a full phrase in a chorale.


3. "The Deep Stack" (Deeper Neural Network - Abstract Reasoning)

Instead of making the model wider (more hidden units), we make it deeper (more layers). Deep networks are often better at learning hierarchical rules (e.g., "If measure 1 was C major, and measure 2 was G major, then measure 4 should resolve to C").


4. "The Composer" (Long-Term Memory)

By doubling the context window to 256, the model sees way back into the past. This helps it remember the original key signature even after modulating.


5. "The Titan" (Max Capacity / High-Fidelity)

This pushes a Mac M2 chip to the limit. A hidden size of 1536 is massive for a simple LSTM. It will be able to capture extremely subtle harmonic nuances, but it is in extreme danger of overfitting. Counter that with high dropout (0.6).

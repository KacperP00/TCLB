# The idea

- Initialy the idea to detect reflection (for optimization algorithm) in the canal during optimization was to analyze all wave properties in a clean canal for equally spaced frequencies (interpolation between if needed). This way there would be a database that can be used to guide the optimzation.
- The ideal wave present in the clean canal would be subtracted from fluid level during the optimization. It would detect the reflection during optimization by measuring the error squared. Big error -> reflection, small error -> small reflection, no error -> no reflection.
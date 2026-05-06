# Domain Prior Knowledge

- The MGO formulation does not evaluate the raw coordinate vector directly. It first decodes the proposal into a legal placement using a wire-mask-guided greedy procedure, so nearby vectors can sometimes map to meaningfully different placements.
- According to the paper, macros are prioritized by the total area of modules connected to them, and each macro is then moved to a legal grid that minimizes incremental HPWL. Ties are broken by proximity to the macro's originally proposed coordinate.
- Because of that decoding rule, coordinates for highly connected macros tend to matter more than coordinates for weakly connected ones, and preserving relative spatial structure can be more useful than independently perturbing every dimension.
- The search surface is continuous at the API boundary but partly discrete after decoding because final placements live on a grid. Optimizers that respect bounds and maintain diversity usually behave more robustly than methods that overcommit to a single local basin too early.
- HPWL is a proxy for downstream chip quality rather than the full design objective. A good MP-HPWL result is useful, but it is still only an approximation to later placement, routing, and timing quality.

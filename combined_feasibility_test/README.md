# Combined readout and XM-CZ feasibility

This folder implements the staged test workflow:

1. regenerate the 140--160 MHz fine readout contour;
2. derive the direct loaded-EC 5 MHz hull from that contour;
3. independently calculate rail-A (140--160 MHz) and rail-B (80--100 MHz)
   readout feasibility on the same direct-EC domain;
4. select one rail-A point, calculate its central-frequency XM-CZ map, and
   overlay it with the independent rail-B map;
5. reconcile fixed fabrication parameters for overlay candidates before any
   full resonator-inclusive validation.

Generated results and checkpoints are stored under `data/` and ignored by Git.
The Conda environment is `work`.

Independent readout-map plot:

```sh
/opt/anaconda3/bin/conda run -n work python combined_feasibility_test/plot_readout_maps.py
/opt/anaconda3/bin/conda run -n work python combined_feasibility_test/compare_rail_b_maps.py
/opt/anaconda3/bin/conda run -n work python combined_feasibility_test/plot_readout_logical_frequency.py
/opt/anaconda3/bin/conda run -n work python combined_feasibility_test/plot_combined_metrics.py
/opt/anaconda3/bin/conda run -n work python combined_feasibility_test/full_network_validation.py
/opt/anaconda3/bin/conda run -n work python combined_feasibility_test/plot_full_network_validation.py
```

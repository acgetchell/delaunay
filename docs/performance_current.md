<!-- research-repo-tools/run-report/v3 -->
# Delaunay public API performance

Run: `4045fd56b9eeb632ab730c4e1700eed87c897425fc4c5579c204a97c2ea0024d`

Evidence: [run payload](performance-runs/runs/4045fd56b9eeb632ab730c4e1700eed87c897425fc4c5579c204a97c2ea0024d/run.json) ·
[evidence envelope](performance-runs/runs/4045fd56b9eeb632ab730c4e1700eed87c897425fc4c5579c204a97c2ea0024d/evidence.json).

Acquisition schedule: sequential; trials: 1; seed: None.
Each trial is retained separately. Within-phase samples are not independent trials; no pooled significance claim is made.

## Trial 1: baseline → current

Collection: 1 segment(s); 0 abandoned attempt(s). Validated abandoned phases are retained and excluded from accepted pairs; failed-output capture status is
recorded.
These are completed campaign slots; recovery does not establish uninterrupted collection, independent trials, or absence of optional-stopping bias.

- Collection segment `f767105fed284f778d7ef86fce7d5b02` started: 1795111506601700000 (Unix ns).
- release-signal/baseline: 1795112084985373000–1795113867234585000 (Unix ns).
- release-signal/current: 1795113875722121000–1795115650396919000 (Unix ns).

Ratios are point estimates within one accepted paired suite. Marginal timing intervals do not establish significance.

## release-signal

Semantic contract: delaunay-public-api-v1. Collection session: `39c9c829557e43dfadd5dfdf18d62710`.
Collection interval (Unix ns): 1795111506602875000–1795115676740436000.
Acquisition: newly collected.

- baseline: available — inside declared release interval; release v0.8.4; revision `0ae37e2b7aed0559c97c5aa8a46871e7383c0f60`.
  Source SHA-256: `8aa79cc89006bb2327eb63d00d7d86dd16a64e2c111f3f5860ed7dd7781a7aa0`; harness SHA-256:
  `4740d40cd1152ca58e9469b3993afdf744269c440afb69bd07ca40e40ebd5dd5`; scientific gate: passed.
- current: available — inside declared release interval; release v0.8.5; revision `526566589d899eeb5b5cd04cd9523ff1fa766188`.
  Source SHA-256: `7ef0dbfee443eb33ef5ae6c03159d77093e482e1cd8f20fe276405a15f9aa0a9`; harness SHA-256:
  `4740d40cd1152ca58e9469b3993afdf744269c440afb69bd07ca40e40ebd5dd5`; scientific gate: passed.

### Mean (ns)

<table>
<thead>
<tr>
<th>Workload</th>
<th>Baseline</th>
<th>Current</th>
<th>Baseline / current</th>
</tr>
</thead>
<tbody>
<tr>
<td>2d/insphere</td>
<td>51.6472 ns [51.3043, 52.0027] (0.95 confidence)</td>
<td>54.6256 ns [53.9593, 55.3029] (0.95 confidence)</td>
<td>0.945477</td>
</tr>
<tr>
<td>2d/insphere&#95;distance</td>
<td>13.7012 ns [13.6752, 13.7285] (0.95 confidence)</td>
<td>13.758 ns [13.7173, 13.8046] (0.95 confidence)</td>
<td>0.995874</td>
</tr>
<tr>
<td>2d/insphere&#95;lifted</td>
<td>51.689 ns [51.3847, 52.0075] (0.95 confidence)</td>
<td>50.2194 ns [49.9916, 50.4563] (0.95 confidence)</td>
<td>1.02926</td>
</tr>
<tr>
<td>3d/insphere</td>
<td>114.48 ns [113.612, 115.351] (0.95 confidence)</td>
<td>110.688 ns [110.234, 111.158] (0.95 confidence)</td>
<td>1.03426</td>
</tr>
<tr>
<td>3d/insphere&#95;distance</td>
<td>32.9488 ns [32.8928, 33.0069] (0.95 confidence)</td>
<td>32.9565 ns [32.8874, 33.0259] (0.95 confidence)</td>
<td>0.999767</td>
</tr>
<tr>
<td>3d/insphere&#95;lifted</td>
<td>113.694 ns [113.069, 114.31] (0.95 confidence)</td>
<td>111.066 ns [110.487, 111.673] (0.95 confidence)</td>
<td>1.02366</td>
</tr>
<tr>
<td>4d/insphere</td>
<td>763.575 ns [760.711, 766.477] (0.95 confidence)</td>
<td>758.434 ns [754.509, 762.444] (0.95 confidence)</td>
<td>1.00678</td>
</tr>
<tr>
<td>4d/insphere&#95;distance</td>
<td>55.1193 ns [54.9838, 55.2637] (0.95 confidence)</td>
<td>54.2139 ns [54.1223, 54.3064] (0.95 confidence)</td>
<td>1.0167</td>
</tr>
<tr>
<td>4d/insphere&#95;lifted</td>
<td>766.547 ns [763.861, 769.251] (0.95 confidence)</td>
<td>745.446 ns [743.662, 747.289] (0.95 confidence)</td>
<td>1.02831</td>
</tr>
<tr>
<td>5d/insphere</td>
<td>1318.15 ns [1314.1, 1322.21] (0.95 confidence)</td>
<td>1305.14 ns [1300.07, 1310.43] (0.95 confidence)</td>
<td>1.00997</td>
</tr>
<tr>
<td>5d/insphere&#95;distance</td>
<td>88.8541 ns [88.5624, 89.1581] (0.95 confidence)</td>
<td>87.636 ns [87.2513, 88.0371] (0.95 confidence)</td>
<td>1.0139</td>
</tr>
<tr>
<td>5d/insphere&#95;lifted</td>
<td>1347.97 ns [1343.47, 1352.49] (0.95 confidence)</td>
<td>1283.6 ns [1280.69, 1286.57] (0.95 confidence)</td>
<td>1.05015</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k1&#95;roundtrip</td>
<td>54872 ns [54600.1, 55266.1] (0.95 confidence)</td>
<td>55511.9 ns [55393.6, 55674.3] (0.95 confidence)</td>
<td>0.988474</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k1&#95;roundtrip&#95;adversarial</td>
<td>166749 ns [166401, 167106] (0.95 confidence)</td>
<td>169054 ns [167533, 170455] (0.95 confidence)</td>
<td>0.986366</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k2&#95;edge&#95;flip</td>
<td>27404 ns [27213.4, 27646.5] (0.95 confidence)</td>
<td>27425 ns [27367, 27497.3] (0.95 confidence)</td>
<td>0.999234</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k2&#95;edge&#95;flip&#95;adversarial</td>
<td>103646 ns [103402, 103901] (0.95 confidence)</td>
<td>102974 ns [102508, 103417] (0.95 confidence)</td>
<td>1.00652</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k1&#95;roundtrip</td>
<td>2.39211e+06 ns [2.38882e+06, 2.39497e+06] (0.95 confidence)</td>
<td>2.36392e+06 ns [2.36098e+06, 2.36709e+06] (0.95 confidence)</td>
<td>1.01192</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k1&#95;roundtrip&#95;adversarial</td>
<td>2.00241e+06 ns [1.99708e+06, 2.00916e+06] (0.95 confidence)</td>
<td>1.97707e+06 ns [1.97406e+06, 1.98014e+06] (0.95 confidence)</td>
<td>1.01281</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k2&#95;roundtrip</td>
<td>6.64937e+06 ns [6.64093e+06, 6.65815e+06] (0.95 confidence)</td>
<td>6.52803e+06 ns [6.51486e+06, 6.53998e+06] (0.95 confidence)</td>
<td>1.01859</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k2&#95;roundtrip&#95;adversarial</td>
<td>5.99139e+06 ns [5.9797e+06, 6.0034e+06] (0.95 confidence)</td>
<td>6.00331e+06 ns [5.99804e+06, 6.00874e+06] (0.95 confidence)</td>
<td>0.998014</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k3&#95;forward</td>
<td>1.39723e+06 ns [1.39349e+06, 1.40145e+06] (0.95 confidence)</td>
<td>1.37698e+06 ns [1.37489e+06, 1.37929e+06] (0.95 confidence)</td>
<td>1.0147</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k3&#95;forward&#95;adversarial</td>
<td>1.87096e+06 ns [1.85846e+06, 1.8861e+06] (0.95 confidence)</td>
<td>1.87484e+06 ns [1.86984e+06, 1.881e+06] (0.95 confidence)</td>
<td>0.997927</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k1&#95;roundtrip</td>
<td>3.93709e+07 ns [3.92537e+07, 3.95005e+07] (0.95 confidence)</td>
<td>3.93056e+07 ns [3.9268e+07, 3.93423e+07] (0.95 confidence)</td>
<td>1.00166</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k1&#95;roundtrip&#95;adversarial</td>
<td>3.40344e+07 ns [3.37714e+07, 3.43222e+07] (0.95 confidence)</td>
<td>3.37708e+07 ns [3.37036e+07, 3.38372e+07] (0.95 confidence)</td>
<td>1.00781</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k2&#95;roundtrip</td>
<td>5.7151e+07 ns [5.69221e+07, 5.73776e+07] (0.95 confidence)</td>
<td>5.68556e+07 ns [5.66329e+07, 5.71781e+07] (0.95 confidence)</td>
<td>1.0052</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k2&#95;roundtrip&#95;adversarial</td>
<td>2.01563e+08 ns [2.00735e+08, 2.0247e+08] (0.95 confidence)</td>
<td>1.96957e+08 ns [1.96664e+08, 1.97242e+08] (0.95 confidence)</td>
<td>1.02339</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k3&#95;roundtrip</td>
<td>6.55502e+07 ns [6.544e+07, 6.56835e+07] (0.95 confidence)</td>
<td>6.53029e+07 ns [6.51516e+07, 6.54889e+07] (0.95 confidence)</td>
<td>1.00379</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k3&#95;roundtrip&#95;adversarial</td>
<td>3.60126e+07 ns [3.57325e+07, 3.63048e+07] (0.95 confidence)</td>
<td>3.56059e+07 ns [3.55663e+07, 3.56469e+07] (0.95 confidence)</td>
<td>1.01142</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k1&#95;roundtrip</td>
<td>1.6023e+08 ns [1.60061e+08, 1.60405e+08] (0.95 confidence)</td>
<td>1.61532e+08 ns [1.59802e+08, 1.63498e+08] (0.95 confidence)</td>
<td>0.991944</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k1&#95;roundtrip&#95;adversarial</td>
<td>1.64601e+08 ns [1.63963e+08, 1.65293e+08] (0.95 confidence)</td>
<td>1.65165e+08 ns [1.6379e+08, 1.66755e+08] (0.95 confidence)</td>
<td>0.996584</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k2&#95;roundtrip</td>
<td>337871 ns [336285, 339970] (0.95 confidence)</td>
<td>327804 ns [327382, 328254] (0.95 confidence)</td>
<td>1.03071</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k2&#95;roundtrip&#95;adversarial</td>
<td>364256 ns [363108, 365995] (0.95 confidence)</td>
<td>356551 ns [354912, 358280] (0.95 confidence)</td>
<td>1.02161</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k3&#95;roundtrip</td>
<td>3.34404e+08 ns [3.34095e+08, 3.34735e+08] (0.95 confidence)</td>
<td>3.33105e+08 ns [3.32003e+08, 3.34413e+08] (0.95 confidence)</td>
<td>1.0039</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k3&#95;roundtrip&#95;adversarial</td>
<td>8.84211e+08 ns [8.82437e+08, 8.8587e+08] (0.95 confidence)</td>
<td>8.92039e+08 ns [8.87035e+08, 8.96516e+08] (0.95 confidence)</td>
<td>0.991225</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;2d/500</td>
<td>214243 ns [212946, 215627] (0.95 confidence)</td>
<td>213586 ns [213241, 213937] (0.95 confidence)</td>
<td>1.00307</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;2d&#95;adversarial/500</td>
<td>214708 ns [213395, 216302] (0.95 confidence)</td>
<td>214888 ns [213854, 215889] (0.95 confidence)</td>
<td>0.999163</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;3d/100</td>
<td>172084 ns [171312, 172876] (0.95 confidence)</td>
<td>166995 ns [166368, 167708] (0.95 confidence)</td>
<td>1.03047</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;3d&#95;adversarial/100</td>
<td>172192 ns [171910, 172462] (0.95 confidence)</td>
<td>169727 ns [169151, 170363] (0.95 confidence)</td>
<td>1.01452</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;4d/30</td>
<td>134767 ns [134378, 135180] (0.95 confidence)</td>
<td>134228 ns [133501, 134989] (0.95 confidence)</td>
<td>1.00402</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;4d&#95;adversarial/30</td>
<td>134214 ns [133857, 134609] (0.95 confidence)</td>
<td>133637 ns [132915, 134427] (0.95 confidence)</td>
<td>1.00431</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;5d/12</td>
<td>37469.1 ns [37375.1, 37574.4] (0.95 confidence)</td>
<td>37038.1 ns [36909.8, 37169.2] (0.95 confidence)</td>
<td>1.01164</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;5d&#95;adversarial/12</td>
<td>37084.7 ns [36958, 37225.7] (0.95 confidence)</td>
<td>36533.8 ns [36411.8, 36661] (0.95 confidence)</td>
<td>1.01508</td>
</tr>
<tr>
<td>circumcenter/solve&#95;path/near&#95;singular&#95;exact&#95;fallback&#95;3d</td>
<td>18098.3 ns [18064, 18133.8] (0.95 confidence)</td>
<td>17383.7 ns [17318.2, 17451.6] (0.95 confidence)</td>
<td>1.0411</td>
</tr>
<tr>
<td>circumcenter/solve&#95;path/regular&#95;lu&#95;3d</td>
<td>23.7949 ns [23.7582, 23.834] (0.95 confidence)</td>
<td>23.3752 ns [23.3368, 23.4186] (0.95 confidence)</td>
<td>1.01795</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;2d/500</td>
<td>393427 ns [392614, 394224] (0.95 confidence)</td>
<td>394071 ns [392675, 395584] (0.95 confidence)</td>
<td>0.998366</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;2d&#95;adversarial/500</td>
<td>393041 ns [391807, 394363] (0.95 confidence)</td>
<td>384787 ns [384457, 385136] (0.95 confidence)</td>
<td>1.02145</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;3d/100</td>
<td>281864 ns [281199, 282552] (0.95 confidence)</td>
<td>276741 ns [276280, 277251] (0.95 confidence)</td>
<td>1.01851</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;3d&#95;adversarial/100</td>
<td>4.0353e+06 ns [4.02255e+06, 4.04887e+06] (0.95 confidence)</td>
<td>4.00932e+06 ns [3.99622e+06, 4.02254e+06] (0.95 confidence)</td>
<td>1.00648</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;4d/30</td>
<td>9.04433e+06 ns [9.01777e+06, 9.07515e+06] (0.95 confidence)</td>
<td>8.95625e+06 ns [8.92031e+06, 8.9996e+06] (0.95 confidence)</td>
<td>1.00983</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;4d&#95;adversarial/30</td>
<td>7.16868e+06 ns [7.14316e+06, 7.19627e+06] (0.95 confidence)</td>
<td>7.1549e+06 ns [7.11995e+06, 7.1913e+06] (0.95 confidence)</td>
<td>1.00193</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;5d/12</td>
<td>3.13686e+06 ns [3.12552e+06, 3.14941e+06] (0.95 confidence)</td>
<td>3.14323e+06 ns [3.13428e+06, 3.1527e+06] (0.95 confidence)</td>
<td>0.997973</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;5d&#95;adversarial/12</td>
<td>2.7384e+06 ns [2.72251e+06, 2.75508e+06] (0.95 confidence)</td>
<td>2.69312e+06 ns [2.68685e+06, 2.6995e+06] (0.95 confidence)</td>
<td>1.01681</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;nearest&#95;visible&#95;facet&#95;3d/100</td>
<td>1170.27 ns [1167.27, 1173.88] (0.95 confidence)</td>
<td>1172.51 ns [1170.03, 1175.73] (0.95 confidence)</td>
<td>0.998095</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;nearest&#95;visible&#95;facet&#95;3d&#95;adversarial/100</td>
<td>41867.4 ns [41570.7, 42184.2] (0.95 confidence)</td>
<td>41124.2 ns [41011.9, 41246.5] (0.95 confidence)</td>
<td>1.01807</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;visible&#95;facets&#95;3d/100</td>
<td>1234.35 ns [1228.48, 1241.36] (0.95 confidence)</td>
<td>1251.24 ns [1243.78, 1260.45] (0.95 confidence)</td>
<td>0.986505</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;visible&#95;facets&#95;3d&#95;adversarial/100</td>
<td>41537.6 ns [41404.5, 41672.8] (0.95 confidence)</td>
<td>41707.4 ns [41610.6, 41805.2] (0.95 confidence)</td>
<td>0.995929</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/is&#95;point&#95;outside&#95;3d/100</td>
<td>96.974 ns [96.7598, 97.2201] (0.95 confidence)</td>
<td>96.966 ns [96.5685, 97.3909] (0.95 confidence)</td>
<td>1.00008</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/is&#95;point&#95;outside&#95;3d&#95;adversarial/100</td>
<td>774.976 ns [771.347, 778.743] (0.95 confidence)</td>
<td>772.527 ns [769.736, 775.428] (0.95 confidence)</td>
<td>1.00317</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/boundary&#95;point&#95;distance</td>
<td>13.6323 ns [13.6005, 13.6666] (0.95 confidence)</td>
<td>13.4822 ns [13.4513, 13.5146] (0.95 confidence)</td>
<td>1.01113</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/boundary&#95;point&#95;insphere</td>
<td>0.824646 ns [0.82231, 0.827137] (0.95 confidence)</td>
<td>0.821995 ns [0.820322, 0.823669] (0.95 confidence)</td>
<td>1.00322</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/boundary&#95;point&#95;lifted</td>
<td>974.695 ns [972.99, 976.519] (0.95 confidence)</td>
<td>985.859 ns [984.39, 987.445] (0.95 confidence)</td>
<td>0.988675</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/far&#95;point&#95;distance</td>
<td>13.6729 ns [13.6491, 13.697] (0.95 confidence)</td>
<td>13.5816 ns [13.5492, 13.6148] (0.95 confidence)</td>
<td>1.00672</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/far&#95;point&#95;insphere</td>
<td>50.2195 ns [49.9216, 50.5404] (0.95 confidence)</td>
<td>49.1554 ns [48.9708, 49.3496] (0.95 confidence)</td>
<td>1.02165</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/far&#95;point&#95;lifted</td>
<td>51.7638 ns [51.4447, 52.083] (0.95 confidence)</td>
<td>49.2112 ns [48.9917, 49.4447] (0.95 confidence)</td>
<td>1.05187</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/near&#95;boundary&#95;distance</td>
<td>13.9037 ns [13.8788, 13.9316] (0.95 confidence)</td>
<td>13.8343 ns [13.8117, 13.8568] (0.95 confidence)</td>
<td>1.00501</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/near&#95;boundary&#95;insphere</td>
<td>49.0476 ns [48.7457, 49.3509] (0.95 confidence)</td>
<td>47.6853 ns [47.4609, 47.9143] (0.95 confidence)</td>
<td>1.02857</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/near&#95;boundary&#95;lifted</td>
<td>49.5593 ns [49.1922, 49.9387] (0.95 confidence)</td>
<td>47.6411 ns [47.337, 47.9602] (0.95 confidence)</td>
<td>1.04026</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/boundary&#95;point&#95;distance</td>
<td>33.0238 ns [32.9608, 33.0881] (0.95 confidence)</td>
<td>32.3469 ns [32.3036, 32.393] (0.95 confidence)</td>
<td>1.02093</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/boundary&#95;point&#95;insphere</td>
<td>1.1953 ns [1.19292, 1.19766] (0.95 confidence)</td>
<td>1.17572 ns [1.17286, 1.17855] (0.95 confidence)</td>
<td>1.01665</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/boundary&#95;point&#95;lifted</td>
<td>1805.61 ns [1802.6, 1808.66] (0.95 confidence)</td>
<td>1852.99 ns [1849.05, 1857.22] (0.95 confidence)</td>
<td>0.974433</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/far&#95;point&#95;distance</td>
<td>32.6644 ns [32.6011, 32.7324] (0.95 confidence)</td>
<td>33.7488 ns [33.6179, 33.9012] (0.95 confidence)</td>
<td>0.96787</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/far&#95;point&#95;insphere</td>
<td>112.188 ns [111.54, 112.868] (0.95 confidence)</td>
<td>113.091 ns [112.305, 113.931] (0.95 confidence)</td>
<td>0.992011</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/far&#95;point&#95;lifted</td>
<td>111.422 ns [110.817, 112.033] (0.95 confidence)</td>
<td>110.104 ns [109.662, 110.564] (0.95 confidence)</td>
<td>1.01197</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/near&#95;boundary&#95;distance</td>
<td>33.1735 ns [33.0891, 33.2647] (0.95 confidence)</td>
<td>33.1735 ns [33.1227, 33.2272] (0.95 confidence)</td>
<td>1</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/near&#95;boundary&#95;insphere</td>
<td>99.2385 ns [98.5152, 99.9905] (0.95 confidence)</td>
<td>101.47 ns [100.737, 102.23] (0.95 confidence)</td>
<td>0.978008</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/near&#95;boundary&#95;lifted</td>
<td>102.842 ns [101.955, 103.742] (0.95 confidence)</td>
<td>96.6417 ns [96.1402, 97.1636] (0.95 confidence)</td>
<td>1.06416</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/boundary&#95;point&#95;distance</td>
<td>54.8931 ns [54.8229, 54.9623] (0.95 confidence)</td>
<td>54.2326 ns [54.1695, 54.2933] (0.95 confidence)</td>
<td>1.01218</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/boundary&#95;point&#95;insphere</td>
<td>1.66727 ns [1.66135, 1.6734] (0.95 confidence)</td>
<td>1.45008 ns [1.41395, 1.48644] (0.95 confidence)</td>
<td>1.14978</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/boundary&#95;point&#95;lifted</td>
<td>3779.32 ns [3772.57, 3786.37] (0.95 confidence)</td>
<td>3768.67 ns [3762.57, 3774.93] (0.95 confidence)</td>
<td>1.00282</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/far&#95;point&#95;distance</td>
<td>54.9442 ns [54.766, 55.1446] (0.95 confidence)</td>
<td>54.0446 ns [53.9352, 54.1534] (0.95 confidence)</td>
<td>1.01665</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/far&#95;point&#95;insphere</td>
<td>755.492 ns [752.696, 758.372] (0.95 confidence)</td>
<td>744.54 ns [740.519, 748.879] (0.95 confidence)</td>
<td>1.01471</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/far&#95;point&#95;lifted</td>
<td>778.556 ns [774.417, 782.744] (0.95 confidence)</td>
<td>728.607 ns [726.878, 730.423] (0.95 confidence)</td>
<td>1.06855</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/near&#95;boundary&#95;distance</td>
<td>55.4235 ns [55.2839, 55.567] (0.95 confidence)</td>
<td>55.6784 ns [55.5579, 55.7949] (0.95 confidence)</td>
<td>0.995422</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/near&#95;boundary&#95;insphere</td>
<td>733.931 ns [731.012, 736.996] (0.95 confidence)</td>
<td>718.946 ns [717.46, 720.472] (0.95 confidence)</td>
<td>1.02084</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/near&#95;boundary&#95;lifted</td>
<td>726.504 ns [724.302, 728.859] (0.95 confidence)</td>
<td>718.576 ns [716.799, 720.457] (0.95 confidence)</td>
<td>1.01103</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/boundary&#95;point&#95;distance</td>
<td>89.1442 ns [88.9041, 89.3963] (0.95 confidence)</td>
<td>87.0737 ns [86.7898, 87.3638] (0.95 confidence)</td>
<td>1.02378</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/boundary&#95;point&#95;insphere</td>
<td>1.73579 ns [1.72784, 1.7461] (0.95 confidence)</td>
<td>1.79178 ns [1.77864, 1.80504] (0.95 confidence)</td>
<td>0.968752</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/boundary&#95;point&#95;lifted</td>
<td>5945.5 ns [5932.51, 5959] (0.95 confidence)</td>
<td>5847.8 ns [5835.76, 5860.6] (0.95 confidence)</td>
<td>1.01671</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/far&#95;point&#95;distance</td>
<td>89.3421 ns [89.0961, 89.5984] (0.95 confidence)</td>
<td>87.1202 ns [86.8616, 87.3856] (0.95 confidence)</td>
<td>1.0255</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/far&#95;point&#95;insphere</td>
<td>1313.96 ns [1309.82, 1318.15] (0.95 confidence)</td>
<td>1250.3 ns [1246.91, 1253.95] (0.95 confidence)</td>
<td>1.05092</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/far&#95;point&#95;lifted</td>
<td>1306.64 ns [1302.71, 1310.65] (0.95 confidence)</td>
<td>1297.14 ns [1292.9, 1301.63] (0.95 confidence)</td>
<td>1.00733</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/near&#95;boundary&#95;distance</td>
<td>89.506 ns [89.2583, 89.7686] (0.95 confidence)</td>
<td>87.4538 ns [87.1398, 87.7881] (0.95 confidence)</td>
<td>1.02347</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/near&#95;boundary&#95;insphere</td>
<td>1277.55 ns [1273.71, 1281.46] (0.95 confidence)</td>
<td>1249.71 ns [1246.45, 1253.09] (0.95 confidence)</td>
<td>1.02228</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/near&#95;boundary&#95;lifted</td>
<td>1289.28 ns [1285.32, 1293.42] (0.95 confidence)</td>
<td>1227.96 ns [1224.91, 1231.25] (0.95 confidence)</td>
<td>1.04994</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;2d/vertices&#95;120&#95;simplices&#95;226</td>
<td>1.64859e+06 ns [1.63475e+06, 1.66521e+06] (0.95 confidence)</td>
<td>1.62408e+06 ns [1.61601e+06, 1.63428e+06] (0.95 confidence)</td>
<td>1.01509</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;3d/vertices&#95;30&#95;simplices&#95;115</td>
<td>1.6276e+08 ns [1.62047e+08, 1.63527e+08] (0.95 confidence)</td>
<td>1.60937e+08 ns [1.59527e+08, 1.62553e+08] (0.95 confidence)</td>
<td>1.01133</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;4d/vertices&#95;16&#95;simplices&#95;68</td>
<td>4.85787e+08 ns [4.83458e+08, 4.88604e+08] (0.95 confidence)</td>
<td>4.81991e+08 ns [4.80568e+08, 4.83147e+08] (0.95 confidence)</td>
<td>1.00788</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;5d/vertices&#95;10&#95;simplices&#95;20</td>
<td>1.40368e+08 ns [1.40218e+08, 1.40505e+08] (0.95 confidence)</td>
<td>1.37829e+08 ns [1.37366e+08, 1.38301e+08] (0.95 confidence)</td>
<td>1.01842</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;2d/10</td>
<td>456999 ns [456457, 457539] (0.95 confidence)</td>
<td>447291 ns [444679, 450410] (0.95 confidence)</td>
<td>1.0217</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;2d&#95;adversarial/10</td>
<td>2.9469e+06 ns [2.94039e+06, 2.95368e+06] (0.95 confidence)</td>
<td>2.92386e+06 ns [2.91506e+06, 2.93327e+06] (0.95 confidence)</td>
<td>1.00788</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;3d/10</td>
<td>2.86215e+06 ns [2.8509e+06, 2.87765e+06] (0.95 confidence)</td>
<td>2.79202e+06 ns [2.7778e+06, 2.80644e+06] (0.95 confidence)</td>
<td>1.02512</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;3d&#95;adversarial/10</td>
<td>1.38927e+07 ns [1.38659e+07, 1.39186e+07] (0.95 confidence)</td>
<td>1.36563e+07 ns [1.36104e+07, 1.37049e+07] (0.95 confidence)</td>
<td>1.01731</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;4d/6</td>
<td>2.71922e+07 ns [2.71067e+07, 2.72971e+07] (0.95 confidence)</td>
<td>2.71439e+07 ns [2.69502e+07, 2.73503e+07] (0.95 confidence)</td>
<td>1.00178</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;4d&#95;adversarial/6</td>
<td>6.8396e+07 ns [6.81082e+07, 6.87348e+07] (0.95 confidence)</td>
<td>6.80793e+07 ns [6.76026e+07, 6.86387e+07] (0.95 confidence)</td>
<td>1.00465</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;5d/4</td>
<td>5.21361e+07 ns [5.19793e+07, 5.23198e+07] (0.95 confidence)</td>
<td>5.1511e+07 ns [5.1394e+07, 5.16408e+07] (0.95 confidence)</td>
<td>1.01214</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;5d&#95;adversarial/4</td>
<td>5.61754e+07 ns [5.60207e+07, 5.63579e+07] (0.95 confidence)</td>
<td>5.57242e+07 ns [5.55905e+07, 5.58519e+07] (0.95 confidence)</td>
<td>1.0081</td>
</tr>
<tr>
<td>locate/exact&#95;hint/2d/vertices&#95;100&#95;simplices&#95;187&#95;seed&#95;dd618c77b9187836&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>7249.96 ns [7238.25, 7262.63] (0.95 confidence)</td>
<td>7205.05 ns [7173.14, 7247.19] (0.95 confidence)</td>
<td>1.00623</td>
</tr>
<tr>
<td>locate/exact&#95;hint/2d/vertices&#95;2000&#95;simplices&#95;3978&#95;seed&#95;01acf93075e96412&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>7552.32 ns [7496.94, 7612.25] (0.95 confidence)</td>
<td>7320.23 ns [7302.58, 7339.04] (0.95 confidence)</td>
<td>1.03171</td>
</tr>
<tr>
<td>locate/exact&#95;hint/2d/vertices&#95;500&#95;simplices&#95;983&#95;seed&#95;1493be4e9d7a5906&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>7653.98 ns [7602.59, 7712.54] (0.95 confidence)</td>
<td>7455.06 ns [7442.85, 7465.32] (0.95 confidence)</td>
<td>1.02668</td>
</tr>
<tr>
<td>locate/exact&#95;hint/3d/vertices&#95;150&#95;simplices&#95;829&#95;seed&#95;a44b52b395a4b44d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>15173 ns [15135.9, 15208.8] (0.95 confidence)</td>
<td>14765.6 ns [14725.5, 14803.9] (0.95 confidence)</td>
<td>1.02759</td>
</tr>
<tr>
<td>locate/exact&#95;hint/3d/vertices&#95;500&#95;simplices&#95;3052&#95;seed&#95;1493be4f9d7a5907&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>15124.4 ns [15071.4, 15185.8] (0.95 confidence)</td>
<td>14816.9 ns [14765.6, 14869.9] (0.95 confidence)</td>
<td>1.02075</td>
</tr>
<tr>
<td>locate/exact&#95;hint/3d/vertices&#95;50&#95;simplices&#95;222&#95;seed&#95;f61fc639dc8c3c19&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>15226.9 ns [15157.3, 15318.3] (0.95 confidence)</td>
<td>14786.2 ns [14705.6, 14884.3] (0.95 confidence)</td>
<td>1.0298</td>
</tr>
<tr>
<td>locate/exact&#95;hint/4d/vertices&#95;100&#95;simplices&#95;1677&#95;seed&#95;dd618c71b9187830&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>2.03228e+06 ns [2.02949e+06, 2.03512e+06] (0.95 confidence)</td>
<td>2.03066e+06 ns [2.01949e+06, 2.04469e+06] (0.95 confidence)</td>
<td>1.00079</td>
</tr>
<tr>
<td>locate/exact&#95;hint/4d/vertices&#95;50&#95;simplices&#95;626&#95;seed&#95;f61fc63edc8c3c1e&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>1.99895e+06 ns [1.9943e+06, 2.00382e+06] (0.95 confidence)</td>
<td>1.97108e+06 ns [1.96154e+06, 1.98375e+06] (0.95 confidence)</td>
<td>1.01414</td>
</tr>
<tr>
<td>locate/exact&#95;hint/5d/vertices&#95;25&#95;simplices&#95;410&#95;seed&#95;63a0e3186e461e08&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>4.81405e+06 ns [4.77795e+06, 4.85301e+06] (0.95 confidence)</td>
<td>4.90387e+06 ns [4.87872e+06, 4.92517e+06] (0.95 confidence)</td>
<td>0.981684</td>
</tr>
<tr>
<td>locate/exact&#95;hint/5d/vertices&#95;40&#95;simplices&#95;1208&#95;seed&#95;a86104fee3a3634d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>4.8397e+06 ns [4.82599e+06, 4.85263e+06] (0.95 confidence)</td>
<td>4.84536e+06 ns [4.81638e+06, 4.87952e+06] (0.95 confidence)</td>
<td>0.998832</td>
</tr>
<tr>
<td>locate/no&#95;hint/2d/vertices&#95;100&#95;simplices&#95;187&#95;seed&#95;dd618c77b9187836&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>75493 ns [75319.3, 75658.4] (0.95 confidence)</td>
<td>75360.1 ns [75290.6, 75444.8] (0.95 confidence)</td>
<td>1.00176</td>
</tr>
<tr>
<td>locate/no&#95;hint/2d/vertices&#95;2000&#95;simplices&#95;3978&#95;seed&#95;01acf93075e96412&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>274071 ns [273209, 274975] (0.95 confidence)</td>
<td>270839 ns [269437, 272598] (0.95 confidence)</td>
<td>1.01193</td>
</tr>
<tr>
<td>locate/no&#95;hint/2d/vertices&#95;500&#95;simplices&#95;983&#95;seed&#95;1493be4e9d7a5906&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>125710 ns [125470, 125943] (0.95 confidence)</td>
<td>123889 ns [123390, 124590] (0.95 confidence)</td>
<td>1.0147</td>
</tr>
<tr>
<td>locate/no&#95;hint/3d/vertices&#95;150&#95;simplices&#95;829&#95;seed&#95;a44b52b395a4b44d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>127028 ns [126087, 128239] (0.95 confidence)</td>
<td>127306 ns [126782, 127811] (0.95 confidence)</td>
<td>0.997813</td>
</tr>
<tr>
<td>locate/no&#95;hint/3d/vertices&#95;500&#95;simplices&#95;3052&#95;seed&#95;1493be4f9d7a5907&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>193392 ns [192989, 193784] (0.95 confidence)</td>
<td>194040 ns [192589, 195427] (0.95 confidence)</td>
<td>0.996659</td>
</tr>
<tr>
<td>locate/no&#95;hint/3d/vertices&#95;50&#95;simplices&#95;222&#95;seed&#95;f61fc639dc8c3c19&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>67306.5 ns [66886.2, 67741.2] (0.95 confidence)</td>
<td>67314.3 ns [66526.5, 68207.2] (0.95 confidence)</td>
<td>0.999884</td>
</tr>
<tr>
<td>locate/no&#95;hint/4d/vertices&#95;100&#95;simplices&#95;1677&#95;seed&#95;dd618c71b9187830&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>1.15239e+07 ns [1.14669e+07, 1.15927e+07] (0.95 confidence)</td>
<td>1.11903e+07 ns [1.11638e+07, 1.12182e+07] (0.95 confidence)</td>
<td>1.02981</td>
</tr>
<tr>
<td>locate/no&#95;hint/4d/vertices&#95;50&#95;simplices&#95;626&#95;seed&#95;f61fc63edc8c3c1e&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>1.14691e+07 ns [1.14163e+07, 1.15172e+07] (0.95 confidence)</td>
<td>1.13445e+07 ns [1.13063e+07, 1.13842e+07] (0.95 confidence)</td>
<td>1.01098</td>
</tr>
<tr>
<td>locate/no&#95;hint/5d/vertices&#95;25&#95;simplices&#95;410&#95;seed&#95;63a0e3186e461e08&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>2.27769e+07 ns [2.26049e+07, 2.29869e+07] (0.95 confidence)</td>
<td>2.25142e+07 ns [2.22603e+07, 2.27769e+07] (0.95 confidence)</td>
<td>1.01167</td>
</tr>
<tr>
<td>locate/no&#95;hint/5d/vertices&#95;40&#95;simplices&#95;1208&#95;seed&#95;a86104fee3a3634d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>4.08598e+07 ns [4.0707e+07, 4.10123e+07] (0.95 confidence)</td>
<td>4.05409e+07 ns [4.02227e+07, 4.08691e+07] (0.95 confidence)</td>
<td>1.00787</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;2d/simplices&#95;226</td>
<td>121154 ns [120500, 121858] (0.95 confidence)</td>
<td>123416 ns [122805, 124103] (0.95 confidence)</td>
<td>0.98167</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;3d/simplices&#95;115</td>
<td>153196 ns [151844, 154776] (0.95 confidence)</td>
<td>154064 ns [152767, 155470] (0.95 confidence)</td>
<td>0.994372</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;4d/simplices&#95;68</td>
<td>1.28548e+06 ns [1.26647e+06, 1.31635e+06] (0.95 confidence)</td>
<td>1.31765e+06 ns [1.3047e+06, 1.32943e+06] (0.95 confidence)</td>
<td>0.975583</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;5d/simplices&#95;20</td>
<td>477520 ns [475889, 479569] (0.95 confidence)</td>
<td>493424 ns [490505, 496541] (0.95 confidence)</td>
<td>0.967768</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;2d/simplices&#95;226</td>
<td>1.38181e+06 ns [1.37131e+06, 1.3942e+06] (0.95 confidence)</td>
<td>1.35351e+06 ns [1.34829e+06, 1.35899e+06] (0.95 confidence)</td>
<td>1.02091</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;3d/simplices&#95;115</td>
<td>1.61296e+08 ns [1.61131e+08, 1.61476e+08] (0.95 confidence)</td>
<td>1.58945e+08 ns [1.58526e+08, 1.59292e+08] (0.95 confidence)</td>
<td>1.01479</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;4d/simplices&#95;68</td>
<td>4.82329e+08 ns [4.79513e+08, 4.85413e+08] (0.95 confidence)</td>
<td>4.79634e+08 ns [4.77433e+08, 4.82316e+08] (0.95 confidence)</td>
<td>1.00562</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;5d/simplices&#95;20</td>
<td>1.39819e+08 ns [1.39519e+08, 1.40141e+08] (0.95 confidence)</td>
<td>1.42618e+08 ns [1.39554e+08, 1.47053e+08] (0.95 confidence)</td>
<td>0.980374</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;2d/simplices&#95;226</td>
<td>1.29401e+06 ns [1.28883e+06, 1.29871e+06] (0.95 confidence)</td>
<td>1.27718e+06 ns [1.26891e+06, 1.28873e+06] (0.95 confidence)</td>
<td>1.01318</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;3d/simplices&#95;115</td>
<td>1.61767e+08 ns [1.61454e+08, 1.62137e+08] (0.95 confidence)</td>
<td>1.59738e+08 ns [1.58566e+08, 1.60977e+08] (0.95 confidence)</td>
<td>1.0127</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;4d/simplices&#95;68</td>
<td>4.79533e+08 ns [4.78848e+08, 4.80254e+08] (0.95 confidence)</td>
<td>4.79384e+08 ns [4.77939e+08, 4.81261e+08] (0.95 confidence)</td>
<td>1.00031</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;5d/simplices&#95;20</td>
<td>1.39764e+08 ns [1.39487e+08, 1.40023e+08] (0.95 confidence)</td>
<td>1.36321e+08 ns [1.36076e+08, 1.36573e+08] (0.95 confidence)</td>
<td>1.02526</td>
</tr>
<tr>
<td>random/insphere&#95;1000&#95;queries</td>
<td>440806 ns [439382, 442152] (0.95 confidence)</td>
<td>423102 ns [420922, 425313] (0.95 confidence)</td>
<td>1.04184</td>
</tr>
<tr>
<td>random/insphere&#95;distance&#95;1000&#95;queries</td>
<td>65113.7 ns [64939.8, 65298.8] (0.95 confidence)</td>
<td>66312.1 ns [66064.8, 66562] (0.95 confidence)</td>
<td>0.981928</td>
</tr>
<tr>
<td>random/insphere&#95;lifted&#95;1000&#95;queries</td>
<td>438524 ns [437188, 439883] (0.95 confidence)</td>
<td>413210 ns [411273, 415262] (0.95 confidence)</td>
<td>1.06126</td>
</tr>
<tr>
<td>tds&#95;new&#95;2d/tds&#95;new/500</td>
<td>4.70549e+07 ns [4.69822e+07, 4.71361e+07] (0.95 confidence)</td>
<td>4.65138e+07 ns [4.64358e+07, 4.66007e+07] (0.95 confidence)</td>
<td>1.01163</td>
</tr>
<tr>
<td>tds&#95;new&#95;2d/tds&#95;new&#95;adversarial/500</td>
<td>5.52694e+07 ns [5.50557e+07, 5.55155e+07] (0.95 confidence)</td>
<td>5.63066e+07 ns [5.60797e+07, 5.65708e+07] (0.95 confidence)</td>
<td>0.98158</td>
</tr>
<tr>
<td>tds&#95;new&#95;3d/tds&#95;new/100</td>
<td>3.82509e+07 ns [3.81283e+07, 3.83896e+07] (0.95 confidence)</td>
<td>3.82122e+07 ns [3.81316e+07, 3.83019e+07] (0.95 confidence)</td>
<td>1.00101</td>
</tr>
<tr>
<td>tds&#95;new&#95;3d/tds&#95;new&#95;adversarial/100</td>
<td>8.65375e+07 ns [8.61014e+07, 8.70377e+07] (0.95 confidence)</td>
<td>8.79749e+07 ns [8.7553e+07, 8.83846e+07] (0.95 confidence)</td>
<td>0.983661</td>
</tr>
<tr>
<td>tds&#95;new&#95;4d/tds&#95;new/30</td>
<td>9.93476e+07 ns [9.92475e+07, 9.94284e+07] (0.95 confidence)</td>
<td>1.02645e+08 ns [1.01823e+08, 1.03498e+08] (0.95 confidence)</td>
<td>0.967872</td>
</tr>
<tr>
<td>tds&#95;new&#95;4d/tds&#95;new&#95;adversarial/30</td>
<td>9.32421e+07 ns [9.29859e+07, 9.35044e+07] (0.95 confidence)</td>
<td>9.62536e+07 ns [9.54826e+07, 9.69644e+07] (0.95 confidence)</td>
<td>0.968713</td>
</tr>
<tr>
<td>tds&#95;new&#95;5d/tds&#95;new/12</td>
<td>2.27926e+07 ns [2.25707e+07, 2.30314e+07] (0.95 confidence)</td>
<td>2.24706e+07 ns [2.23656e+07, 2.25777e+07] (0.95 confidence)</td>
<td>1.01433</td>
</tr>
<tr>
<td>tds&#95;new&#95;5d/tds&#95;new&#95;adversarial/12</td>
<td>2.04838e+07 ns [2.04386e+07, 2.05385e+07] (0.95 confidence)</td>
<td>2.08031e+07 ns [2.06842e+07, 2.09083e+07] (0.95 confidence)</td>
<td>0.984652</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;2d/500</td>
<td>813770 ns [808085, 819912] (0.95 confidence)</td>
<td>796213 ns [790759, 802109] (0.95 confidence)</td>
<td>1.02205</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;2d&#95;adversarial/500</td>
<td>1.1139e+06 ns [1.11039e+06, 1.11767e+06] (0.95 confidence)</td>
<td>1.16775e+06 ns [1.15679e+06, 1.17823e+06] (0.95 confidence)</td>
<td>0.953882</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;3d/100</td>
<td>1.26896e+06 ns [1.26266e+06, 1.27641e+06] (0.95 confidence)</td>
<td>1.27683e+06 ns [1.25311e+06, 1.30672e+06] (0.95 confidence)</td>
<td>0.993843</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;3d&#95;adversarial/100</td>
<td>2.28995e+06 ns [2.27328e+06, 2.30835e+06] (0.95 confidence)</td>
<td>2.43811e+06 ns [2.40578e+06, 2.47236e+06] (0.95 confidence)</td>
<td>0.939229</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;4d/30</td>
<td>4.90059e+06 ns [4.88203e+06, 4.92264e+06] (0.95 confidence)</td>
<td>4.70408e+06 ns [4.6674e+06, 4.74266e+06] (0.95 confidence)</td>
<td>1.04177</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;4d&#95;adversarial/30</td>
<td>4.79538e+06 ns [4.76919e+06, 4.82276e+06] (0.95 confidence)</td>
<td>4.63621e+06 ns [4.61873e+06, 4.65372e+06] (0.95 confidence)</td>
<td>1.03433</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;5d/12</td>
<td>2.1973e+06 ns [2.18926e+06, 2.20557e+06] (0.95 confidence)</td>
<td>2.16076e+06 ns [2.15549e+06, 2.1662e+06] (0.95 confidence)</td>
<td>1.01691</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;5d&#95;adversarial/12</td>
<td>1.97185e+06 ns [1.96686e+06, 1.97716e+06] (0.95 confidence)</td>
<td>1.98928e+06 ns [1.97292e+06, 2.00818e+06] (0.95 confidence)</td>
<td>0.991238</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;2d/500</td>
<td>808706 ns [806483, 810917] (0.95 confidence)</td>
<td>815462 ns [809214, 822006] (0.95 confidence)</td>
<td>0.991716</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;2d&#95;adversarial/500</td>
<td>1.12467e+06 ns [1.1199e+06, 1.12927e+06] (0.95 confidence)</td>
<td>1.10797e+06 ns [1.10033e+06, 1.11638e+06] (0.95 confidence)</td>
<td>1.01508</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;3d/100</td>
<td>1.27138e+06 ns [1.26613e+06, 1.27701e+06] (0.95 confidence)</td>
<td>1.27824e+06 ns [1.26005e+06, 1.29796e+06] (0.95 confidence)</td>
<td>0.994633</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;3d&#95;adversarial/100</td>
<td>2.29038e+06 ns [2.28098e+06, 2.29957e+06] (0.95 confidence)</td>
<td>2.25548e+06 ns [2.22492e+06, 2.2914e+06] (0.95 confidence)</td>
<td>1.01548</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;4d/30</td>
<td>4.92656e+06 ns [4.89754e+06, 4.95919e+06] (0.95 confidence)</td>
<td>4.71833e+06 ns [4.68391e+06, 4.75641e+06] (0.95 confidence)</td>
<td>1.04413</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;4d&#95;adversarial/30</td>
<td>4.77082e+06 ns [4.75021e+06, 4.79239e+06] (0.95 confidence)</td>
<td>4.65561e+06 ns [4.63606e+06, 4.67695e+06] (0.95 confidence)</td>
<td>1.02475</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;5d/12</td>
<td>2.21486e+06 ns [2.20333e+06, 2.22753e+06] (0.95 confidence)</td>
<td>2.16497e+06 ns [2.15753e+06, 2.17254e+06] (0.95 confidence)</td>
<td>1.02305</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;5d&#95;adversarial/12</td>
<td>1.9801e+06 ns [1.97302e+06, 1.98754e+06] (0.95 confidence)</td>
<td>1.95872e+06 ns [1.95252e+06, 1.96501e+06] (0.95 confidence)</td>
<td>1.01092</td>
</tr>
<tr>
<td>validation/validate&#95;2d/500</td>
<td>8.64326e+06 ns [8.52301e+06, 8.80149e+06] (0.95 confidence)</td>
<td>8.52203e+06 ns [8.49579e+06, 8.54827e+06] (0.95 confidence)</td>
<td>1.01422</td>
</tr>
<tr>
<td>validation/validate&#95;2d&#95;adversarial/500</td>
<td>5.01723e+07 ns [4.99565e+07, 5.04562e+07] (0.95 confidence)</td>
<td>5.01217e+07 ns [4.99061e+07, 5.03249e+07] (0.95 confidence)</td>
<td>1.00101</td>
</tr>
<tr>
<td>validation/validate&#95;3d/100</td>
<td>8.69678e+08 ns [8.67466e+08, 8.72106e+08] (0.95 confidence)</td>
<td>8.64827e+08 ns [8.61804e+08, 8.68092e+08] (0.95 confidence)</td>
<td>1.00561</td>
</tr>
<tr>
<td>validation/validate&#95;3d&#95;adversarial/100</td>
<td>1.04765e+09 ns [1.04625e+09, 1.04908e+09] (0.95 confidence)</td>
<td>1.04047e+09 ns [1.03468e+09, 1.04763e+09] (0.95 confidence)</td>
<td>1.0069</td>
</tr>
<tr>
<td>validation/validate&#95;4d/30</td>
<td>4.0333e+09 ns [4.01299e+09, 4.05519e+09] (0.95 confidence)</td>
<td>3.99167e+09 ns [3.98182e+09, 4.00234e+09] (0.95 confidence)</td>
<td>1.01043</td>
</tr>
<tr>
<td>validation/validate&#95;4d&#95;adversarial/30</td>
<td>4.0435e+09 ns [4.03759e+09, 4.04967e+09] (0.95 confidence)</td>
<td>4.09631e+09 ns [4.08878e+09, 4.10371e+09] (0.95 confidence)</td>
<td>0.987108</td>
</tr>
<tr>
<td>validation/validate&#95;5d/12</td>
<td>9.106e+08 ns [9.09388e+08, 9.11795e+08] (0.95 confidence)</td>
<td>8.94465e+08 ns [8.92451e+08, 8.96852e+08] (0.95 confidence)</td>
<td>1.01804</td>
</tr>
<tr>
<td>validation/validate&#95;5d&#95;adversarial/12</td>
<td>8.31169e+08 ns [8.29913e+08, 8.32487e+08] (0.95 confidence)</td>
<td>8.26647e+08 ns [8.25671e+08, 8.2766e+08] (0.95 confidence)</td>
<td>1.00547</td>
</tr>
</tbody>
</table>

### Median (ns)

<table>
<thead>
<tr>
<th>Workload</th>
<th>Baseline</th>
<th>Current</th>
<th>Baseline / current</th>
</tr>
</thead>
<tbody>
<tr>
<td>2d/insphere</td>
<td>51.3746 ns [50.8679, 51.9108] (0.95 confidence)</td>
<td>54.3769 ns [52.6955, 55.4799] (0.95 confidence)</td>
<td>0.944787</td>
</tr>
<tr>
<td>2d/insphere&#95;distance</td>
<td>13.658 ns [13.6277, 13.6908] (0.95 confidence)</td>
<td>13.711 ns [13.6922, 13.7256] (0.95 confidence)</td>
<td>0.996133</td>
</tr>
<tr>
<td>2d/insphere&#95;lifted</td>
<td>51.4136 ns [50.9842, 51.8229] (0.95 confidence)</td>
<td>50.0394 ns [49.8003, 50.2364] (0.95 confidence)</td>
<td>1.02746</td>
</tr>
<tr>
<td>3d/insphere</td>
<td>114.82 ns [112.783, 116.035] (0.95 confidence)</td>
<td>110.581 ns [109.659, 110.975] (0.95 confidence)</td>
<td>1.03833</td>
</tr>
<tr>
<td>3d/insphere&#95;distance</td>
<td>32.9037 ns [32.8349, 32.9865] (0.95 confidence)</td>
<td>32.9578 ns [32.894, 33.1175] (0.95 confidence)</td>
<td>0.998358</td>
</tr>
<tr>
<td>3d/insphere&#95;lifted</td>
<td>113.92 ns [113.278, 114.518] (0.95 confidence)</td>
<td>110.373 ns [109.993, 110.947] (0.95 confidence)</td>
<td>1.03214</td>
</tr>
<tr>
<td>4d/insphere</td>
<td>762.174 ns [759.38, 763.941] (0.95 confidence)</td>
<td>755.9 ns [751.025, 758.905] (0.95 confidence)</td>
<td>1.0083</td>
</tr>
<tr>
<td>4d/insphere&#95;distance</td>
<td>54.8789 ns [54.8258, 55.1357] (0.95 confidence)</td>
<td>54.2999 ns [54.1734, 54.3674] (0.95 confidence)</td>
<td>1.01066</td>
</tr>
<tr>
<td>4d/insphere&#95;lifted</td>
<td>766.563 ns [763.428, 768.859] (0.95 confidence)</td>
<td>744.883 ns [742.057, 746.422] (0.95 confidence)</td>
<td>1.02911</td>
</tr>
<tr>
<td>5d/insphere</td>
<td>1317.57 ns [1314.78, 1321.05] (0.95 confidence)</td>
<td>1306.28 ns [1300.56, 1309.97] (0.95 confidence)</td>
<td>1.00864</td>
</tr>
<tr>
<td>5d/insphere&#95;distance</td>
<td>88.5816 ns [88.3254, 88.9065] (0.95 confidence)</td>
<td>87.0017 ns [86.8105, 87.7997] (0.95 confidence)</td>
<td>1.01816</td>
</tr>
<tr>
<td>5d/insphere&#95;lifted</td>
<td>1350.36 ns [1341.4, 1354.71] (0.95 confidence)</td>
<td>1282.3 ns [1279.47, 1285.76] (0.95 confidence)</td>
<td>1.05307</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k1&#95;roundtrip</td>
<td>54709 ns [54524.1, 54961.3] (0.95 confidence)</td>
<td>55452.9 ns [55350.1, 55567.1] (0.95 confidence)</td>
<td>0.986585</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k1&#95;roundtrip&#95;adversarial</td>
<td>166606 ns [166340, 167263] (0.95 confidence)</td>
<td>169683 ns [166559, 171381] (0.95 confidence)</td>
<td>0.981864</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k2&#95;edge&#95;flip</td>
<td>27214.5 ns [27154.1, 27613.8] (0.95 confidence)</td>
<td>27386.8 ns [27353.1, 27502.8] (0.95 confidence)</td>
<td>0.993707</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k2&#95;edge&#95;flip&#95;adversarial</td>
<td>103727 ns [103257, 103912] (0.95 confidence)</td>
<td>103285 ns [102201, 103664] (0.95 confidence)</td>
<td>1.00427</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k1&#95;roundtrip</td>
<td>2.39353e+06 ns [2.38878e+06, 2.39578e+06] (0.95 confidence)</td>
<td>2.36411e+06 ns [2.35881e+06, 2.36716e+06] (0.95 confidence)</td>
<td>1.01244</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k1&#95;roundtrip&#95;adversarial</td>
<td>1.99865e+06 ns [1.99598e+06, 2.01024e+06] (0.95 confidence)</td>
<td>1.97733e+06 ns [1.97321e+06, 1.97969e+06] (0.95 confidence)</td>
<td>1.01079</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k2&#95;roundtrip</td>
<td>6.64842e+06 ns [6.64196e+06, 6.65777e+06] (0.95 confidence)</td>
<td>6.52841e+06 ns [6.51454e+06, 6.54506e+06] (0.95 confidence)</td>
<td>1.01838</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k2&#95;roundtrip&#95;adversarial</td>
<td>5.99225e+06 ns [5.97396e+06, 6.00457e+06] (0.95 confidence)</td>
<td>6.00523e+06 ns [5.99419e+06, 6.00862e+06] (0.95 confidence)</td>
<td>0.997837</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k3&#95;forward</td>
<td>1.39557e+06 ns [1.3918e+06, 1.40166e+06] (0.95 confidence)</td>
<td>1.3773e+06 ns [1.3733e+06, 1.37895e+06] (0.95 confidence)</td>
<td>1.01327</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k3&#95;forward&#95;adversarial</td>
<td>1.86021e+06 ns [1.85433e+06, 1.88911e+06] (0.95 confidence)</td>
<td>1.87286e+06 ns [1.86707e+06, 1.87966e+06] (0.95 confidence)</td>
<td>0.993247</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k1&#95;roundtrip</td>
<td>3.93004e+07 ns [3.9226e+07, 3.95185e+07] (0.95 confidence)</td>
<td>3.93111e+07 ns [3.92629e+07, 3.93445e+07] (0.95 confidence)</td>
<td>0.999728</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k1&#95;roundtrip&#95;adversarial</td>
<td>3.38411e+07 ns [3.37093e+07, 3.45888e+07] (0.95 confidence)</td>
<td>3.37744e+07 ns [3.3657e+07, 3.38811e+07] (0.95 confidence)</td>
<td>1.00198</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k2&#95;roundtrip</td>
<td>5.7096e+07 ns [5.683e+07, 5.74617e+07] (0.95 confidence)</td>
<td>5.67542e+07 ns [5.65632e+07, 5.69023e+07] (0.95 confidence)</td>
<td>1.00602</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k2&#95;roundtrip&#95;adversarial</td>
<td>2.01047e+08 ns [2.0058e+08, 2.02641e+08] (0.95 confidence)</td>
<td>1.96952e+08 ns [1.96543e+08, 1.97431e+08] (0.95 confidence)</td>
<td>1.02079</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k3&#95;roundtrip</td>
<td>6.55304e+07 ns [6.53699e+07, 6.56402e+07] (0.95 confidence)</td>
<td>6.5241e+07 ns [6.51423e+07, 6.54096e+07] (0.95 confidence)</td>
<td>1.00444</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k3&#95;roundtrip&#95;adversarial</td>
<td>3.58461e+07 ns [3.56148e+07, 3.65719e+07] (0.95 confidence)</td>
<td>3.56029e+07 ns [3.55596e+07, 3.56545e+07] (0.95 confidence)</td>
<td>1.00683</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k1&#95;roundtrip</td>
<td>1.60336e+08 ns [1.59922e+08, 1.60382e+08] (0.95 confidence)</td>
<td>1.59836e+08 ns [1.59048e+08, 1.65647e+08] (0.95 confidence)</td>
<td>1.00313</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k1&#95;roundtrip&#95;adversarial</td>
<td>1.64549e+08 ns [1.63882e+08, 1.65289e+08] (0.95 confidence)</td>
<td>1.64464e+08 ns [1.63177e+08, 1.66851e+08] (0.95 confidence)</td>
<td>1.00051</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k2&#95;roundtrip</td>
<td>336623 ns [335851, 339103] (0.95 confidence)</td>
<td>327750 ns [327180, 328381] (0.95 confidence)</td>
<td>1.02707</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k2&#95;roundtrip&#95;adversarial</td>
<td>363330 ns [362904, 364653] (0.95 confidence)</td>
<td>356013 ns [354287, 359231] (0.95 confidence)</td>
<td>1.02055</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k3&#95;roundtrip</td>
<td>3.34326e+08 ns [3.34052e+08, 3.3479e+08] (0.95 confidence)</td>
<td>3.32208e+08 ns [3.31683e+08, 3.35133e+08] (0.95 confidence)</td>
<td>1.00637</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k3&#95;roundtrip&#95;adversarial</td>
<td>8.84296e+08 ns [8.82276e+08, 8.86604e+08] (0.95 confidence)</td>
<td>8.94146e+08 ns [8.85117e+08, 8.98034e+08] (0.95 confidence)</td>
<td>0.988983</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;2d/500</td>
<td>212683 ns [211419, 215841] (0.95 confidence)</td>
<td>213520 ns [213032, 214021] (0.95 confidence)</td>
<td>0.996083</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;2d&#95;adversarial/500</td>
<td>214088 ns [212355, 215479] (0.95 confidence)</td>
<td>214682 ns [213992, 216243] (0.95 confidence)</td>
<td>0.99723</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;3d/100</td>
<td>172155 ns [170111, 173287] (0.95 confidence)</td>
<td>166186 ns [165978, 167242] (0.95 confidence)</td>
<td>1.03592</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;3d&#95;adversarial/100</td>
<td>172156 ns [171940, 172649] (0.95 confidence)</td>
<td>169169 ns [168681, 170697] (0.95 confidence)</td>
<td>1.01765</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;4d/30</td>
<td>134466 ns [134178, 135290] (0.95 confidence)</td>
<td>134175 ns [132842, 135117] (0.95 confidence)</td>
<td>1.00217</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;4d&#95;adversarial/30</td>
<td>133989 ns [133762, 134353] (0.95 confidence)</td>
<td>133340 ns [132093, 134326] (0.95 confidence)</td>
<td>1.00487</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;5d/12</td>
<td>37408.9 ns [37342.6, 37520.8] (0.95 confidence)</td>
<td>37019.9 ns [36822, 37269.1] (0.95 confidence)</td>
<td>1.01051</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;5d&#95;adversarial/12</td>
<td>36986.3 ns [36903.9, 37111.2] (0.95 confidence)</td>
<td>36479.8 ns [36302, 36753.7] (0.95 confidence)</td>
<td>1.01388</td>
</tr>
<tr>
<td>circumcenter/solve&#95;path/near&#95;singular&#95;exact&#95;fallback&#95;3d</td>
<td>18079.2 ns [18053.8, 18107.3] (0.95 confidence)</td>
<td>17285.6 ns [17245.6, 17334.2] (0.95 confidence)</td>
<td>1.04591</td>
</tr>
<tr>
<td>circumcenter/solve&#95;path/regular&#95;lu&#95;3d</td>
<td>23.782 ns [23.7582, 23.8215] (0.95 confidence)</td>
<td>23.3679 ns [23.329, 23.3829] (0.95 confidence)</td>
<td>1.01772</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;2d/500</td>
<td>393184 ns [392608, 394735] (0.95 confidence)</td>
<td>392775 ns [392258, 396617] (0.95 confidence)</td>
<td>1.00104</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;2d&#95;adversarial/500</td>
<td>392234 ns [391067, 394761] (0.95 confidence)</td>
<td>384822 ns [384489, 384950] (0.95 confidence)</td>
<td>1.01926</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;3d/100</td>
<td>282044 ns [281241, 282534] (0.95 confidence)</td>
<td>276560 ns [276308, 277133] (0.95 confidence)</td>
<td>1.01983</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;3d&#95;adversarial/100</td>
<td>4.03321e+06 ns [4.02537e+06, 4.04094e+06] (0.95 confidence)</td>
<td>4.00831e+06 ns [3.99299e+06, 4.02942e+06] (0.95 confidence)</td>
<td>1.00621</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;4d/30</td>
<td>9.03235e+06 ns [9.02128e+06, 9.05459e+06] (0.95 confidence)</td>
<td>8.92209e+06 ns [8.89599e+06, 8.96871e+06] (0.95 confidence)</td>
<td>1.01236</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;4d&#95;adversarial/30</td>
<td>7.1544e+06 ns [7.13924e+06, 7.18646e+06] (0.95 confidence)</td>
<td>7.12991e+06 ns [7.09672e+06, 7.20691e+06] (0.95 confidence)</td>
<td>1.00344</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;5d/12</td>
<td>3.12773e+06 ns [3.11894e+06, 3.145e+06] (0.95 confidence)</td>
<td>3.13398e+06 ns [3.1299e+06, 3.15849e+06] (0.95 confidence)</td>
<td>0.998005</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;5d&#95;adversarial/12</td>
<td>2.73181e+06 ns [2.71286e+06, 2.75686e+06] (0.95 confidence)</td>
<td>2.69517e+06 ns [2.68501e+06, 2.69939e+06] (0.95 confidence)</td>
<td>1.0136</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;nearest&#95;visible&#95;facet&#95;3d/100</td>
<td>1166.8 ns [1166.1, 1170.94] (0.95 confidence)</td>
<td>1170.79 ns [1168.61, 1174.13] (0.95 confidence)</td>
<td>0.996593</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;nearest&#95;visible&#95;facet&#95;3d&#95;adversarial/100</td>
<td>41657.1 ns [41473.5, 42094] (0.95 confidence)</td>
<td>41093.1 ns [40901.1, 41268.7] (0.95 confidence)</td>
<td>1.01372</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;visible&#95;facets&#95;3d/100</td>
<td>1228.98 ns [1225.85, 1236.43] (0.95 confidence)</td>
<td>1248.99 ns [1246.28, 1255.29] (0.95 confidence)</td>
<td>0.983977</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;visible&#95;facets&#95;3d&#95;adversarial/100</td>
<td>41607.3 ns [41323.6, 41678.7] (0.95 confidence)</td>
<td>41669 ns [41577.4, 41827.1] (0.95 confidence)</td>
<td>0.998519</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/is&#95;point&#95;outside&#95;3d/100</td>
<td>96.9287 ns [96.7397, 97.0582] (0.95 confidence)</td>
<td>96.6995 ns [96.2737, 97.7396] (0.95 confidence)</td>
<td>1.00237</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/is&#95;point&#95;outside&#95;3d&#95;adversarial/100</td>
<td>773.766 ns [768.278, 779.866] (0.95 confidence)</td>
<td>771.37 ns [768.64, 775.875] (0.95 confidence)</td>
<td>1.00311</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/boundary&#95;point&#95;distance</td>
<td>13.5969 ns [13.5553, 13.6466] (0.95 confidence)</td>
<td>13.4221 ns [13.3942, 13.4567] (0.95 confidence)</td>
<td>1.01302</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/boundary&#95;point&#95;insphere</td>
<td>0.822341 ns [0.819806, 0.82454] (0.95 confidence)</td>
<td>0.821683 ns [0.818629, 0.824818] (0.95 confidence)</td>
<td>1.0008</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/boundary&#95;point&#95;lifted</td>
<td>973.189 ns [971.937, 976.202] (0.95 confidence)</td>
<td>985.456 ns [984.193, 986.84] (0.95 confidence)</td>
<td>0.987552</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/far&#95;point&#95;distance</td>
<td>13.7003 ns [13.6781, 13.7068] (0.95 confidence)</td>
<td>13.6249 ns [13.5914, 13.6406] (0.95 confidence)</td>
<td>1.00554</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/far&#95;point&#95;insphere</td>
<td>49.8325 ns [49.4281, 50.163] (0.95 confidence)</td>
<td>48.9148 ns [48.6341, 49.2026] (0.95 confidence)</td>
<td>1.01876</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/far&#95;point&#95;lifted</td>
<td>51.932 ns [51.5085, 52.2631] (0.95 confidence)</td>
<td>48.9617 ns [48.6118, 49.1703] (0.95 confidence)</td>
<td>1.06067</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/near&#95;boundary&#95;distance</td>
<td>13.8689 ns [13.856, 13.8906] (0.95 confidence)</td>
<td>13.846 ns [13.8339, 13.8588] (0.95 confidence)</td>
<td>1.00165</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/near&#95;boundary&#95;insphere</td>
<td>48.9803 ns [48.4442, 49.4147] (0.95 confidence)</td>
<td>47.5407 ns [47.3892, 47.9964] (0.95 confidence)</td>
<td>1.03028</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/near&#95;boundary&#95;lifted</td>
<td>49.3233 ns [48.8874, 49.6881] (0.95 confidence)</td>
<td>47.1865 ns [47.0202, 47.6633] (0.95 confidence)</td>
<td>1.04528</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/boundary&#95;point&#95;distance</td>
<td>32.9867 ns [32.9327, 33.1081] (0.95 confidence)</td>
<td>32.2874 ns [32.2345, 32.3699] (0.95 confidence)</td>
<td>1.02166</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/boundary&#95;point&#95;insphere</td>
<td>1.1958 ns [1.19352, 1.19767] (0.95 confidence)</td>
<td>1.17719 ns [1.1757, 1.17842] (0.95 confidence)</td>
<td>1.01581</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/boundary&#95;point&#95;lifted</td>
<td>1807 ns [1802.34, 1809.18] (0.95 confidence)</td>
<td>1844.42 ns [1840.72, 1850.36] (0.95 confidence)</td>
<td>0.979713</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/far&#95;point&#95;distance</td>
<td>32.6555 ns [32.5925, 32.6878] (0.95 confidence)</td>
<td>33.6788 ns [33.5798, 33.8085] (0.95 confidence)</td>
<td>0.969615</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/far&#95;point&#95;insphere</td>
<td>111.788 ns [110.913, 112.664] (0.95 confidence)</td>
<td>111.653 ns [111.122, 112.883] (0.95 confidence)</td>
<td>1.00121</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/far&#95;point&#95;lifted</td>
<td>111.278 ns [110.471, 111.881] (0.95 confidence)</td>
<td>109.65 ns [109.114, 109.894] (0.95 confidence)</td>
<td>1.01485</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/near&#95;boundary&#95;distance</td>
<td>33.1126 ns [33.029, 33.1752] (0.95 confidence)</td>
<td>33.1266 ns [33.0567, 33.2005] (0.95 confidence)</td>
<td>0.999578</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/near&#95;boundary&#95;insphere</td>
<td>98.4383 ns [97.5147, 99.3739] (0.95 confidence)</td>
<td>101.71 ns [100.035, 102.279] (0.95 confidence)</td>
<td>0.967838</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/near&#95;boundary&#95;lifted</td>
<td>102.609 ns [102.005, 103.565] (0.95 confidence)</td>
<td>95.9063 ns [95.6732, 96.711] (0.95 confidence)</td>
<td>1.06988</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/boundary&#95;point&#95;distance</td>
<td>54.9315 ns [54.8278, 55.0121] (0.95 confidence)</td>
<td>54.3239 ns [54.2992, 54.34] (0.95 confidence)</td>
<td>1.01118</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/boundary&#95;point&#95;insphere</td>
<td>1.66591 ns [1.6585, 1.67319] (0.95 confidence)</td>
<td>1.37103 ns [1.28869, 1.61123] (0.95 confidence)</td>
<td>1.21507</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/boundary&#95;point&#95;lifted</td>
<td>3773.33 ns [3764.93, 3783.69] (0.95 confidence)</td>
<td>3761.3 ns [3756.52, 3764.19] (0.95 confidence)</td>
<td>1.0032</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/far&#95;point&#95;distance</td>
<td>54.8729 ns [54.647, 55.042] (0.95 confidence)</td>
<td>54.194 ns [54.1221, 54.2518] (0.95 confidence)</td>
<td>1.01253</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/far&#95;point&#95;insphere</td>
<td>754.324 ns [751.61, 757.142] (0.95 confidence)</td>
<td>733.803 ns [730.73, 742.371] (0.95 confidence)</td>
<td>1.02797</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/far&#95;point&#95;lifted</td>
<td>779.351 ns [773.1, 784.598] (0.95 confidence)</td>
<td>727.095 ns [725.319, 729.054] (0.95 confidence)</td>
<td>1.07187</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/near&#95;boundary&#95;distance</td>
<td>55.398 ns [55.2046, 55.5612] (0.95 confidence)</td>
<td>55.8817 ns [55.8013, 55.9714] (0.95 confidence)</td>
<td>0.991345</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/near&#95;boundary&#95;insphere</td>
<td>733.101 ns [728.183, 735.646] (0.95 confidence)</td>
<td>718.819 ns [716.197, 720.048] (0.95 confidence)</td>
<td>1.01987</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/near&#95;boundary&#95;lifted</td>
<td>725.168 ns [723.78, 727.371] (0.95 confidence)</td>
<td>718.71 ns [717.31, 720.246] (0.95 confidence)</td>
<td>1.00899</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/boundary&#95;point&#95;distance</td>
<td>88.8873 ns [88.6225, 89.0726] (0.95 confidence)</td>
<td>86.7471 ns [86.3245, 87.2924] (0.95 confidence)</td>
<td>1.02467</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/boundary&#95;point&#95;insphere</td>
<td>1.7244 ns [1.72029, 1.72846] (0.95 confidence)</td>
<td>1.78863 ns [1.78143, 1.80642] (0.95 confidence)</td>
<td>0.964091</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/boundary&#95;point&#95;lifted</td>
<td>5943.06 ns [5926.5, 5958.28] (0.95 confidence)</td>
<td>5830.09 ns [5814.17, 5846.21] (0.95 confidence)</td>
<td>1.01938</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/far&#95;point&#95;distance</td>
<td>89.0532 ns [88.8723, 89.2827] (0.95 confidence)</td>
<td>86.946 ns [86.643, 87.1463] (0.95 confidence)</td>
<td>1.02424</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/far&#95;point&#95;insphere</td>
<td>1313.29 ns [1302.69, 1319.76] (0.95 confidence)</td>
<td>1243.83 ns [1242.16, 1249.77] (0.95 confidence)</td>
<td>1.05584</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/far&#95;point&#95;lifted</td>
<td>1305.73 ns [1300.2, 1310.95] (0.95 confidence)</td>
<td>1293.58 ns [1290.2, 1296.91] (0.95 confidence)</td>
<td>1.00939</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/near&#95;boundary&#95;distance</td>
<td>89.2034 ns [88.9831, 89.6194] (0.95 confidence)</td>
<td>87.0978 ns [86.7884, 87.4131] (0.95 confidence)</td>
<td>1.02418</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/near&#95;boundary&#95;insphere</td>
<td>1277.1 ns [1269.17, 1283.67] (0.95 confidence)</td>
<td>1248.58 ns [1243.08, 1253.07] (0.95 confidence)</td>
<td>1.02284</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/near&#95;boundary&#95;lifted</td>
<td>1285.8 ns [1280.24, 1291.3] (0.95 confidence)</td>
<td>1225.41 ns [1222.23, 1228.37] (0.95 confidence)</td>
<td>1.04929</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;2d/vertices&#95;120&#95;simplices&#95;226</td>
<td>1.63535e+06 ns [1.63375e+06, 1.66755e+06] (0.95 confidence)</td>
<td>1.61643e+06 ns [1.61354e+06, 1.63162e+06] (0.95 confidence)</td>
<td>1.01171</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;3d/vertices&#95;30&#95;simplices&#95;115</td>
<td>1.62688e+08 ns [1.61617e+08, 1.63742e+08] (0.95 confidence)</td>
<td>1.59713e+08 ns [1.59099e+08, 1.63334e+08] (0.95 confidence)</td>
<td>1.01862</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;4d/vertices&#95;16&#95;simplices&#95;68</td>
<td>4.83739e+08 ns [4.83129e+08, 4.89041e+08] (0.95 confidence)</td>
<td>4.82779e+08 ns [4.80922e+08, 4.83608e+08] (0.95 confidence)</td>
<td>1.00199</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;5d/vertices&#95;10&#95;simplices&#95;20</td>
<td>1.40414e+08 ns [1.40157e+08, 1.40578e+08] (0.95 confidence)</td>
<td>1.37777e+08 ns [1.37048e+08, 1.38648e+08] (0.95 confidence)</td>
<td>1.01914</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;2d/10</td>
<td>456919 ns [456128, 457975] (0.95 confidence)</td>
<td>445128 ns [443955, 450095] (0.95 confidence)</td>
<td>1.02649</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;2d&#95;adversarial/10</td>
<td>2.94572e+06 ns [2.93684e+06, 2.95914e+06] (0.95 confidence)</td>
<td>2.91934e+06 ns [2.90837e+06, 2.94094e+06] (0.95 confidence)</td>
<td>1.00904</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;3d/10</td>
<td>2.85664e+06 ns [2.84857e+06, 2.85821e+06] (0.95 confidence)</td>
<td>2.7897e+06 ns [2.78171e+06, 2.8059e+06] (0.95 confidence)</td>
<td>1.024</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;3d&#95;adversarial/10</td>
<td>1.38994e+07 ns [1.38638e+07, 1.39227e+07] (0.95 confidence)</td>
<td>1.36418e+07 ns [1.36016e+07, 1.3716e+07] (0.95 confidence)</td>
<td>1.01888</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;4d/6</td>
<td>2.71251e+07 ns [2.70991e+07, 2.7204e+07] (0.95 confidence)</td>
<td>2.70105e+07 ns [2.68335e+07, 2.75716e+07] (0.95 confidence)</td>
<td>1.00424</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;4d&#95;adversarial/6</td>
<td>6.83281e+07 ns [6.78484e+07, 6.86337e+07] (0.95 confidence)</td>
<td>6.76626e+07 ns [6.74377e+07, 6.81911e+07] (0.95 confidence)</td>
<td>1.00984</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;5d/4</td>
<td>5.21608e+07 ns [5.18811e+07, 5.23077e+07] (0.95 confidence)</td>
<td>5.14212e+07 ns [5.13076e+07, 5.17344e+07] (0.95 confidence)</td>
<td>1.01438</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;5d&#95;adversarial/4</td>
<td>5.60758e+07 ns [5.59264e+07, 5.63867e+07] (0.95 confidence)</td>
<td>5.5767e+07 ns [5.55379e+07, 5.60079e+07] (0.95 confidence)</td>
<td>1.00554</td>
</tr>
<tr>
<td>locate/exact&#95;hint/2d/vertices&#95;100&#95;simplices&#95;187&#95;seed&#95;dd618c77b9187836&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>7246.33 ns [7236.01, 7263.19] (0.95 confidence)</td>
<td>7184.42 ns [7169.96, 7226.24] (0.95 confidence)</td>
<td>1.00862</td>
</tr>
<tr>
<td>locate/exact&#95;hint/2d/vertices&#95;2000&#95;simplices&#95;3978&#95;seed&#95;01acf93075e96412&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>7527.43 ns [7467.57, 7640.63] (0.95 confidence)</td>
<td>7317.54 ns [7290.84, 7345.02] (0.95 confidence)</td>
<td>1.02868</td>
</tr>
<tr>
<td>locate/exact&#95;hint/2d/vertices&#95;500&#95;simplices&#95;983&#95;seed&#95;1493be4e9d7a5906&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>7616.83 ns [7598.39, 7727.37] (0.95 confidence)</td>
<td>7457.78 ns [7450.11, 7464.04] (0.95 confidence)</td>
<td>1.02133</td>
</tr>
<tr>
<td>locate/exact&#95;hint/3d/vertices&#95;150&#95;simplices&#95;829&#95;seed&#95;a44b52b395a4b44d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>15173.6 ns [15126.8, 15227.8] (0.95 confidence)</td>
<td>14776.5 ns [14712.2, 14818.2] (0.95 confidence)</td>
<td>1.02687</td>
</tr>
<tr>
<td>locate/exact&#95;hint/3d/vertices&#95;500&#95;simplices&#95;3052&#95;seed&#95;1493be4f9d7a5907&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>15100.9 ns [15054.4, 15190.1] (0.95 confidence)</td>
<td>14802.9 ns [14755.9, 14882.1] (0.95 confidence)</td>
<td>1.02013</td>
</tr>
<tr>
<td>locate/exact&#95;hint/3d/vertices&#95;50&#95;simplices&#95;222&#95;seed&#95;f61fc639dc8c3c19&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>15207 ns [15122.7, 15266.3] (0.95 confidence)</td>
<td>14763.1 ns [14694.5, 14839.6] (0.95 confidence)</td>
<td>1.03006</td>
</tr>
<tr>
<td>locate/exact&#95;hint/4d/vertices&#95;100&#95;simplices&#95;1677&#95;seed&#95;dd618c71b9187830&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>2.0324e+06 ns [2.02826e+06, 2.03586e+06] (0.95 confidence)</td>
<td>2.02482e+06 ns [2.01497e+06, 2.04306e+06] (0.95 confidence)</td>
<td>1.00375</td>
</tr>
<tr>
<td>locate/exact&#95;hint/4d/vertices&#95;50&#95;simplices&#95;626&#95;seed&#95;f61fc63edc8c3c1e&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>1.99532e+06 ns [1.9924e+06, 2.00783e+06] (0.95 confidence)</td>
<td>1.96719e+06 ns [1.95973e+06, 1.97541e+06] (0.95 confidence)</td>
<td>1.0143</td>
</tr>
<tr>
<td>locate/exact&#95;hint/5d/vertices&#95;25&#95;simplices&#95;410&#95;seed&#95;63a0e3186e461e08&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>4.78334e+06 ns [4.77425e+06, 4.87058e+06] (0.95 confidence)</td>
<td>4.90822e+06 ns [4.8854e+06, 4.93426e+06] (0.95 confidence)</td>
<td>0.974556</td>
</tr>
<tr>
<td>locate/exact&#95;hint/5d/vertices&#95;40&#95;simplices&#95;1208&#95;seed&#95;a86104fee3a3634d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>4.83941e+06 ns [4.82447e+06, 4.85701e+06] (0.95 confidence)</td>
<td>4.83359e+06 ns [4.80202e+06, 4.88121e+06] (0.95 confidence)</td>
<td>1.0012</td>
</tr>
<tr>
<td>locate/no&#95;hint/2d/vertices&#95;100&#95;simplices&#95;187&#95;seed&#95;dd618c77b9187836&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>75552.1 ns [75285.7, 75684.7] (0.95 confidence)</td>
<td>75310.2 ns [75258.8, 75439.5] (0.95 confidence)</td>
<td>1.00321</td>
</tr>
<tr>
<td>locate/no&#95;hint/2d/vertices&#95;2000&#95;simplices&#95;3978&#95;seed&#95;01acf93075e96412&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>273885 ns [273027, 275160] (0.95 confidence)</td>
<td>269960 ns [268997, 272222] (0.95 confidence)</td>
<td>1.01454</td>
</tr>
<tr>
<td>locate/no&#95;hint/2d/vertices&#95;500&#95;simplices&#95;983&#95;seed&#95;1493be4e9d7a5906&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>125710 ns [125501, 125974] (0.95 confidence)</td>
<td>123603 ns [123267, 124098] (0.95 confidence)</td>
<td>1.01704</td>
</tr>
<tr>
<td>locate/no&#95;hint/3d/vertices&#95;150&#95;simplices&#95;829&#95;seed&#95;a44b52b395a4b44d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>126644 ns [125763, 127676] (0.95 confidence)</td>
<td>127154 ns [126675, 128170] (0.95 confidence)</td>
<td>0.995996</td>
</tr>
<tr>
<td>locate/no&#95;hint/3d/vertices&#95;500&#95;simplices&#95;3052&#95;seed&#95;1493be4f9d7a5907&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>193411 ns [192856, 194142] (0.95 confidence)</td>
<td>194622 ns [192108, 195865] (0.95 confidence)</td>
<td>0.993775</td>
</tr>
<tr>
<td>locate/no&#95;hint/3d/vertices&#95;50&#95;simplices&#95;222&#95;seed&#95;f61fc639dc8c3c19&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>67201.4 ns [66622.5, 67991.3] (0.95 confidence)</td>
<td>66631.7 ns [66295.1, 69203.6] (0.95 confidence)</td>
<td>1.00855</td>
</tr>
<tr>
<td>locate/no&#95;hint/4d/vertices&#95;100&#95;simplices&#95;1677&#95;seed&#95;dd618c71b9187830&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>1.15179e+07 ns [1.14416e+07, 1.15616e+07] (0.95 confidence)</td>
<td>1.11865e+07 ns [1.115e+07, 1.12254e+07] (0.95 confidence)</td>
<td>1.02962</td>
</tr>
<tr>
<td>locate/no&#95;hint/4d/vertices&#95;50&#95;simplices&#95;626&#95;seed&#95;f61fc63edc8c3c1e&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>1.1493e+07 ns [1.14104e+07, 1.15335e+07] (0.95 confidence)</td>
<td>1.13523e+07 ns [1.12804e+07, 1.13976e+07] (0.95 confidence)</td>
<td>1.0124</td>
</tr>
<tr>
<td>locate/no&#95;hint/5d/vertices&#95;25&#95;simplices&#95;410&#95;seed&#95;63a0e3186e461e08&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>2.27492e+07 ns [2.25273e+07, 2.29276e+07] (0.95 confidence)</td>
<td>2.24027e+07 ns [2.22052e+07, 2.28762e+07] (0.95 confidence)</td>
<td>1.01547</td>
</tr>
<tr>
<td>locate/no&#95;hint/5d/vertices&#95;40&#95;simplices&#95;1208&#95;seed&#95;a86104fee3a3634d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>4.08706e+07 ns [4.05925e+07, 4.10963e+07] (0.95 confidence)</td>
<td>4.06421e+07 ns [4.00124e+07, 4.09392e+07] (0.95 confidence)</td>
<td>1.00562</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;2d/simplices&#95;226</td>
<td>120873 ns [120263, 122072] (0.95 confidence)</td>
<td>123149 ns [122702, 124200] (0.95 confidence)</td>
<td>0.981511</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;3d/simplices&#95;115</td>
<td>152425 ns [151316, 154923] (0.95 confidence)</td>
<td>153859 ns [151673, 155854] (0.95 confidence)</td>
<td>0.990683</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;4d/simplices&#95;68</td>
<td>1.27282e+06 ns [1.26187e+06, 1.28495e+06] (0.95 confidence)</td>
<td>1.3181e+06 ns [1.30843e+06, 1.33128e+06] (0.95 confidence)</td>
<td>0.965652</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;5d/simplices&#95;20</td>
<td>477043 ns [475637, 478787] (0.95 confidence)</td>
<td>493246 ns [488756, 497608] (0.95 confidence)</td>
<td>0.967149</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;2d/simplices&#95;226</td>
<td>1.37406e+06 ns [1.36724e+06, 1.39416e+06] (0.95 confidence)</td>
<td>1.35339e+06 ns [1.34514e+06, 1.36039e+06] (0.95 confidence)</td>
<td>1.01527</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;3d/simplices&#95;115</td>
<td>1.61216e+08 ns [1.6108e+08, 1.61499e+08] (0.95 confidence)</td>
<td>1.59094e+08 ns [1.58625e+08, 1.5937e+08] (0.95 confidence)</td>
<td>1.01334</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;4d/simplices&#95;68</td>
<td>4.80924e+08 ns [4.777e+08, 4.86744e+08] (0.95 confidence)</td>
<td>4.77784e+08 ns [4.76874e+08, 4.8179e+08] (0.95 confidence)</td>
<td>1.00657</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;5d/simplices&#95;20</td>
<td>1.3976e+08 ns [1.39266e+08, 1.40212e+08] (0.95 confidence)</td>
<td>1.40131e+08 ns [1.38843e+08, 1.44352e+08] (0.95 confidence)</td>
<td>0.997354</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;2d/simplices&#95;226</td>
<td>1.29326e+06 ns [1.29126e+06, 1.29932e+06] (0.95 confidence)</td>
<td>1.27335e+06 ns [1.26578e+06, 1.2818e+06] (0.95 confidence)</td>
<td>1.01564</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;3d/simplices&#95;115</td>
<td>1.61711e+08 ns [1.61388e+08, 1.62007e+08] (0.95 confidence)</td>
<td>1.59293e+08 ns [1.58259e+08, 1.61509e+08] (0.95 confidence)</td>
<td>1.01518</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;4d/simplices&#95;68</td>
<td>4.79412e+08 ns [4.78482e+08, 4.80454e+08] (0.95 confidence)</td>
<td>4.78938e+08 ns [4.77265e+08, 4.80683e+08] (0.95 confidence)</td>
<td>1.00099</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;5d/simplices&#95;20</td>
<td>1.39891e+08 ns [1.39305e+08, 1.40185e+08] (0.95 confidence)</td>
<td>1.36315e+08 ns [1.35987e+08, 1.36685e+08] (0.95 confidence)</td>
<td>1.02623</td>
</tr>
<tr>
<td>random/insphere&#95;1000&#95;queries</td>
<td>441885 ns [440470, 444194] (0.95 confidence)</td>
<td>419806 ns [417944, 424934] (0.95 confidence)</td>
<td>1.05259</td>
</tr>
<tr>
<td>random/insphere&#95;distance&#95;1000&#95;queries</td>
<td>64787.3 ns [64682.9, 64901.3] (0.95 confidence)</td>
<td>66039.4 ns [65792.1, 66325.1] (0.95 confidence)</td>
<td>0.98104</td>
</tr>
<tr>
<td>random/insphere&#95;lifted&#95;1000&#95;queries</td>
<td>438421 ns [436881, 440479] (0.95 confidence)</td>
<td>410642 ns [408745, 412807] (0.95 confidence)</td>
<td>1.06765</td>
</tr>
<tr>
<td>tds&#95;new&#95;2d/tds&#95;new/500</td>
<td>4.70189e+07 ns [4.69742e+07, 4.71633e+07] (0.95 confidence)</td>
<td>4.6485e+07 ns [4.64085e+07, 4.65877e+07] (0.95 confidence)</td>
<td>1.01149</td>
</tr>
<tr>
<td>tds&#95;new&#95;2d/tds&#95;new&#95;adversarial/500</td>
<td>5.50926e+07 ns [5.49888e+07, 5.55984e+07] (0.95 confidence)</td>
<td>5.61771e+07 ns [5.60045e+07, 5.6575e+07] (0.95 confidence)</td>
<td>0.980695</td>
</tr>
<tr>
<td>tds&#95;new&#95;3d/tds&#95;new/100</td>
<td>3.81963e+07 ns [3.8082e+07, 3.83753e+07] (0.95 confidence)</td>
<td>3.81808e+07 ns [3.81274e+07, 3.83228e+07] (0.95 confidence)</td>
<td>1.00041</td>
</tr>
<tr>
<td>tds&#95;new&#95;3d/tds&#95;new&#95;adversarial/100</td>
<td>8.63944e+07 ns [8.60815e+07, 8.71046e+07] (0.95 confidence)</td>
<td>8.79731e+07 ns [8.73567e+07, 8.86046e+07] (0.95 confidence)</td>
<td>0.982055</td>
</tr>
<tr>
<td>tds&#95;new&#95;4d/tds&#95;new/30</td>
<td>9.93606e+07 ns [9.93009e+07, 9.94318e+07] (0.95 confidence)</td>
<td>1.02849e+08 ns [1.0157e+08, 1.03622e+08] (0.95 confidence)</td>
<td>0.966087</td>
</tr>
<tr>
<td>tds&#95;new&#95;4d/tds&#95;new&#95;adversarial/30</td>
<td>9.31002e+07 ns [9.28736e+07, 9.37047e+07] (0.95 confidence)</td>
<td>9.64811e+07 ns [9.52617e+07, 9.72161e+07] (0.95 confidence)</td>
<td>0.964958</td>
</tr>
<tr>
<td>tds&#95;new&#95;5d/tds&#95;new/12</td>
<td>2.27595e+07 ns [2.25598e+07, 2.30618e+07] (0.95 confidence)</td>
<td>2.24918e+07 ns [2.23805e+07, 2.25379e+07] (0.95 confidence)</td>
<td>1.0119</td>
</tr>
<tr>
<td>tds&#95;new&#95;5d/tds&#95;new&#95;adversarial/12</td>
<td>2.04744e+07 ns [2.04322e+07, 2.05087e+07] (0.95 confidence)</td>
<td>2.08508e+07 ns [2.06533e+07, 2.09539e+07] (0.95 confidence)</td>
<td>0.98195</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;2d/500</td>
<td>811536 ns [805290, 825347] (0.95 confidence)</td>
<td>792059 ns [788810, 800455] (0.95 confidence)</td>
<td>1.02459</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;2d&#95;adversarial/500</td>
<td>1.11256e+06 ns [1.11146e+06, 1.11815e+06] (0.95 confidence)</td>
<td>1.1659e+06 ns [1.15145e+06, 1.18767e+06] (0.95 confidence)</td>
<td>0.954251</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;3d/100</td>
<td>1.26731e+06 ns [1.25801e+06, 1.27367e+06] (0.95 confidence)</td>
<td>1.25786e+06 ns [1.24058e+06, 1.29125e+06] (0.95 confidence)</td>
<td>1.00752</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;3d&#95;adversarial/100</td>
<td>2.2818e+06 ns [2.26289e+06, 2.32059e+06] (0.95 confidence)</td>
<td>2.43395e+06 ns [2.38007e+06, 2.45361e+06] (0.95 confidence)</td>
<td>0.937488</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;4d/30</td>
<td>4.88085e+06 ns [4.86983e+06, 4.93248e+06] (0.95 confidence)</td>
<td>4.67375e+06 ns [4.64718e+06, 4.79449e+06] (0.95 confidence)</td>
<td>1.04431</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;4d&#95;adversarial/30</td>
<td>4.78041e+06 ns [4.74001e+06, 4.85474e+06] (0.95 confidence)</td>
<td>4.63977e+06 ns [4.60849e+06, 4.677e+06] (0.95 confidence)</td>
<td>1.03031</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;5d/12</td>
<td>2.19649e+06 ns [2.17849e+06, 2.20754e+06] (0.95 confidence)</td>
<td>2.15929e+06 ns [2.15541e+06, 2.17045e+06] (0.95 confidence)</td>
<td>1.01723</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;5d&#95;adversarial/12</td>
<td>1.97148e+06 ns [1.96428e+06, 1.97967e+06] (0.95 confidence)</td>
<td>1.97912e+06 ns [1.96116e+06, 2.02281e+06] (0.95 confidence)</td>
<td>0.996143</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;2d/500</td>
<td>807855 ns [805641, 810973] (0.95 confidence)</td>
<td>814555 ns [804895, 823075] (0.95 confidence)</td>
<td>0.991774</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;2d&#95;adversarial/500</td>
<td>1.12505e+06 ns [1.11675e+06, 1.13242e+06] (0.95 confidence)</td>
<td>1.10317e+06 ns [1.0959e+06, 1.1193e+06] (0.95 confidence)</td>
<td>1.01983</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;3d/100</td>
<td>1.26696e+06 ns [1.2634e+06, 1.27772e+06] (0.95 confidence)</td>
<td>1.26477e+06 ns [1.25367e+06, 1.32635e+06] (0.95 confidence)</td>
<td>1.00173</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;3d&#95;adversarial/100</td>
<td>2.29307e+06 ns [2.28669e+06, 2.30116e+06] (0.95 confidence)</td>
<td>2.23155e+06 ns [2.20971e+06, 2.27537e+06] (0.95 confidence)</td>
<td>1.02757</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;4d/30</td>
<td>4.91067e+06 ns [4.88392e+06, 4.95896e+06] (0.95 confidence)</td>
<td>4.7024e+06 ns [4.67124e+06, 4.71144e+06] (0.95 confidence)</td>
<td>1.04429</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;4d&#95;adversarial/30</td>
<td>4.76353e+06 ns [4.73645e+06, 4.81377e+06] (0.95 confidence)</td>
<td>4.63824e+06 ns [4.62342e+06, 4.68997e+06] (0.95 confidence)</td>
<td>1.02701</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;5d/12</td>
<td>2.20774e+06 ns [2.2018e+06, 2.22867e+06] (0.95 confidence)</td>
<td>2.16325e+06 ns [2.1533e+06, 2.17401e+06] (0.95 confidence)</td>
<td>1.02057</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;5d&#95;adversarial/12</td>
<td>1.97729e+06 ns [1.97197e+06, 1.99127e+06] (0.95 confidence)</td>
<td>1.95813e+06 ns [1.9511e+06, 1.96846e+06] (0.95 confidence)</td>
<td>1.00978</td>
</tr>
<tr>
<td>validation/validate&#95;2d/500</td>
<td>8.57442e+06 ns [8.45646e+06, 8.74314e+06] (0.95 confidence)</td>
<td>8.52116e+06 ns [8.48572e+06, 8.56648e+06] (0.95 confidence)</td>
<td>1.00625</td>
</tr>
<tr>
<td>validation/validate&#95;2d&#95;adversarial/500</td>
<td>5.00847e+07 ns [4.99152e+07, 5.02536e+07] (0.95 confidence)</td>
<td>5.02297e+07 ns [4.97499e+07, 5.04816e+07] (0.95 confidence)</td>
<td>0.997113</td>
</tr>
<tr>
<td>validation/validate&#95;3d/100</td>
<td>8.69589e+08 ns [8.65561e+08, 8.73223e+08] (0.95 confidence)</td>
<td>8.64773e+08 ns [8.5969e+08, 8.68082e+08] (0.95 confidence)</td>
<td>1.00557</td>
</tr>
<tr>
<td>validation/validate&#95;3d&#95;adversarial/100</td>
<td>1.04674e+09 ns [1.04571e+09, 1.05019e+09] (0.95 confidence)</td>
<td>1.03618e+09 ns [1.03031e+09, 1.04029e+09] (0.95 confidence)</td>
<td>1.01019</td>
</tr>
<tr>
<td>validation/validate&#95;4d/30</td>
<td>4.03214e+09 ns [3.99484e+09, 4.06785e+09] (0.95 confidence)</td>
<td>3.98254e+09 ns [3.97396e+09, 4.00523e+09] (0.95 confidence)</td>
<td>1.01245</td>
</tr>
<tr>
<td>validation/validate&#95;4d&#95;adversarial/30</td>
<td>4.04101e+09 ns [4.03484e+09, 4.0571e+09] (0.95 confidence)</td>
<td>4.09572e+09 ns [4.08151e+09, 4.10811e+09] (0.95 confidence)</td>
<td>0.986642</td>
</tr>
<tr>
<td>validation/validate&#95;5d/12</td>
<td>9.10666e+08 ns [9.08972e+08, 9.11845e+08] (0.95 confidence)</td>
<td>8.93531e+08 ns [8.92666e+08, 8.94732e+08] (0.95 confidence)</td>
<td>1.01918</td>
</tr>
<tr>
<td>validation/validate&#95;5d&#95;adversarial/12</td>
<td>8.31818e+08 ns [8.28728e+08, 8.33547e+08] (0.95 confidence)</td>
<td>8.26849e+08 ns [8.24739e+08, 8.27731e+08] (0.95 confidence)</td>
<td>1.00601</td>
</tr>
</tbody>
</table>

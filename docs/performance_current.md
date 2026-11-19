<!-- research-repo-tools/run-report/v3 -->
# Delaunay public API performance

Run: `e736ac85e7715c81afbcb0d20ea925347408d86095c8e0af19f8fc4cf8aa5c90`

Evidence: [run payload](performance-runs/runs/e736ac85e7715c81afbcb0d20ea925347408d86095c8e0af19f8fc4cf8aa5c90/run.json) ·
[evidence envelope](performance-runs/runs/e736ac85e7715c81afbcb0d20ea925347408d86095c8e0af19f8fc4cf8aa5c90/evidence.json).

Acquisition schedule: sequential; trials: 1; seed: None.
Each trial is retained separately. Within-phase samples are not independent trials; no pooled significance claim is made.

## Trial 1: baseline → current

Collection: 1 segment(s); 0 abandoned attempt(s). Validated abandoned phases are retained and excluded from accepted pairs; failed-output capture status is
recorded.
These are completed campaign slots; recovery does not establish uninterrupted collection, independent trials, or absence of optional-stopping bias.

- Collection segment `24110df0cd62477d9de95c655d27ff1d` started: 1795089076776855000 (Unix ns).
- release-signal/baseline: 1795089649950720000–1795091415219613000 (Unix ns).
- release-signal/current: 1795091427300448000–1795093211304644000 (Unix ns).

Ratios are point estimates within one accepted paired suite. Marginal timing intervals do not establish significance.

## release-signal

Semantic contract: delaunay-public-api-v1. Collection session: `2ea53cade4e040d4b895f5511ac61e0d`.
Collection interval (Unix ns): 1795089076778070000–1795093242736807000.
Acquisition: newly collected.

- baseline: available — inside declared release interval; release v0.8.4; revision `0ae37e2b7aed0559c97c5aa8a46871e7383c0f60`.
  Source SHA-256: `8aa79cc89006bb2327eb63d00d7d86dd16a64e2c111f3f5860ed7dd7781a7aa0`; harness SHA-256:
  `4740d40cd1152ca58e9469b3993afdf744269c440afb69bd07ca40e40ebd5dd5`; scientific gate: passed.
- current: available — inside declared release interval; release v0.8.5; revision `e5cf33dbe4ef9cf98b0d8f88aac6d571d3e86a3d`.
  Source SHA-256: `1356fc3fc0082582ac9ed4deddfe3fc3d94e643cb5c1e43d589ba807d56c37c0`; harness SHA-256:
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
<td>51.9508 ns [51.6159, 52.3008] (0.95 confidence)</td>
<td>52.2557 ns [51.9093, 52.6208] (0.95 confidence)</td>
<td>0.994164</td>
</tr>
<tr>
<td>2d/insphere&#95;distance</td>
<td>13.8681 ns [13.8475, 13.8891] (0.95 confidence)</td>
<td>13.8469 ns [13.8033, 13.8953] (0.95 confidence)</td>
<td>1.00153</td>
</tr>
<tr>
<td>2d/insphere&#95;lifted</td>
<td>51.6446 ns [51.3638, 51.9432] (0.95 confidence)</td>
<td>52.5058 ns [52.1912, 52.8288] (0.95 confidence)</td>
<td>0.983599</td>
</tr>
<tr>
<td>3d/insphere</td>
<td>110.262 ns [109.842, 110.703] (0.95 confidence)</td>
<td>110.726 ns [110.425, 111.04] (0.95 confidence)</td>
<td>0.995807</td>
</tr>
<tr>
<td>3d/insphere&#95;distance</td>
<td>33.5307 ns [33.4347, 33.6271] (0.95 confidence)</td>
<td>33.1911 ns [33.1061, 33.2811] (0.95 confidence)</td>
<td>1.01023</td>
</tr>
<tr>
<td>3d/insphere&#95;lifted</td>
<td>107.955 ns [107.514, 108.408] (0.95 confidence)</td>
<td>111.051 ns [110.584, 111.539] (0.95 confidence)</td>
<td>0.972121</td>
</tr>
<tr>
<td>4d/insphere</td>
<td>748.232 ns [745.983, 750.498] (0.95 confidence)</td>
<td>750.664 ns [747.984, 753.476] (0.95 confidence)</td>
<td>0.99676</td>
</tr>
<tr>
<td>4d/insphere&#95;distance</td>
<td>54.2656 ns [54.2272, 54.306] (0.95 confidence)</td>
<td>54.4596 ns [54.3829, 54.5402] (0.95 confidence)</td>
<td>0.996437</td>
</tr>
<tr>
<td>4d/insphere&#95;lifted</td>
<td>743.103 ns [740.703, 745.61] (0.95 confidence)</td>
<td>752.225 ns [750.623, 753.847] (0.95 confidence)</td>
<td>0.987872</td>
</tr>
<tr>
<td>5d/insphere</td>
<td>1322.28 ns [1317.18, 1327.5] (0.95 confidence)</td>
<td>1306.03 ns [1300.55, 1311.79] (0.95 confidence)</td>
<td>1.01245</td>
</tr>
<tr>
<td>5d/insphere&#95;distance</td>
<td>88.3495 ns [88.1965, 88.5064] (0.95 confidence)</td>
<td>88.4906 ns [88.2157, 88.7662] (0.95 confidence)</td>
<td>0.998406</td>
</tr>
<tr>
<td>5d/insphere&#95;lifted</td>
<td>1309.85 ns [1306.42, 1313.41] (0.95 confidence)</td>
<td>1316.71 ns [1311.66, 1321.94] (0.95 confidence)</td>
<td>0.994793</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k1&#95;roundtrip</td>
<td>53733.6 ns [53616.1, 53919.1] (0.95 confidence)</td>
<td>55158.9 ns [55068.4, 55264.9] (0.95 confidence)</td>
<td>0.974159</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k1&#95;roundtrip&#95;adversarial</td>
<td>164949 ns [164604, 165336] (0.95 confidence)</td>
<td>169746 ns [167203, 172981] (0.95 confidence)</td>
<td>0.971745</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k2&#95;edge&#95;flip</td>
<td>27109.3 ns [26769.8, 27517.8] (0.95 confidence)</td>
<td>27649.7 ns [27263.6, 28050.8] (0.95 confidence)</td>
<td>0.980456</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k2&#95;edge&#95;flip&#95;adversarial</td>
<td>101832 ns [101570, 102159] (0.95 confidence)</td>
<td>102112 ns [101892, 102470] (0.95 confidence)</td>
<td>0.997258</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k1&#95;roundtrip</td>
<td>2.36611e+06 ns [2.36348e+06, 2.36881e+06] (0.95 confidence)</td>
<td>2.36192e+06 ns [2.35426e+06, 2.36916e+06] (0.95 confidence)</td>
<td>1.00177</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k1&#95;roundtrip&#95;adversarial</td>
<td>1.96998e+06 ns [1.96521e+06, 1.97449e+06] (0.95 confidence)</td>
<td>1.9412e+06 ns [1.93505e+06, 1.94749e+06] (0.95 confidence)</td>
<td>1.01482</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k2&#95;roundtrip</td>
<td>6.49692e+06 ns [6.48673e+06, 6.50753e+06] (0.95 confidence)</td>
<td>6.68334e+06 ns [6.6662e+06, 6.70841e+06] (0.95 confidence)</td>
<td>0.972107</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k2&#95;roundtrip&#95;adversarial</td>
<td>5.93127e+06 ns [5.92118e+06, 5.9405e+06] (0.95 confidence)</td>
<td>6.00218e+06 ns [5.98628e+06, 6.02022e+06] (0.95 confidence)</td>
<td>0.988187</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k3&#95;forward</td>
<td>1.39247e+06 ns [1.38654e+06, 1.39881e+06] (0.95 confidence)</td>
<td>1.38807e+06 ns [1.38564e+06, 1.39091e+06] (0.95 confidence)</td>
<td>1.00317</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k3&#95;forward&#95;adversarial</td>
<td>1.83631e+06 ns [1.83278e+06, 1.83983e+06] (0.95 confidence)</td>
<td>1.87495e+06 ns [1.86615e+06, 1.88637e+06] (0.95 confidence)</td>
<td>0.979393</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k1&#95;roundtrip</td>
<td>3.88639e+07 ns [3.87768e+07, 3.89449e+07] (0.95 confidence)</td>
<td>3.91365e+07 ns [3.90104e+07, 3.92939e+07] (0.95 confidence)</td>
<td>0.993037</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k1&#95;roundtrip&#95;adversarial</td>
<td>3.29343e+07 ns [3.28564e+07, 3.30365e+07] (0.95 confidence)</td>
<td>3.31003e+07 ns [3.30276e+07, 3.31814e+07] (0.95 confidence)</td>
<td>0.994985</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k2&#95;roundtrip</td>
<td>5.54465e+07 ns [5.53466e+07, 5.55588e+07] (0.95 confidence)</td>
<td>5.65833e+07 ns [5.65233e+07, 5.66513e+07] (0.95 confidence)</td>
<td>0.97991</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k2&#95;roundtrip&#95;adversarial</td>
<td>1.95184e+08 ns [1.94323e+08, 1.96292e+08] (0.95 confidence)</td>
<td>1.96837e+08 ns [1.96601e+08, 1.97132e+08] (0.95 confidence)</td>
<td>0.991604</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k3&#95;roundtrip</td>
<td>6.35687e+07 ns [6.34269e+07, 6.37027e+07] (0.95 confidence)</td>
<td>6.59245e+07 ns [6.58383e+07, 6.60143e+07] (0.95 confidence)</td>
<td>0.964266</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k3&#95;roundtrip&#95;adversarial</td>
<td>3.48157e+07 ns [3.47142e+07, 3.49286e+07] (0.95 confidence)</td>
<td>3.52475e+07 ns [3.51682e+07, 3.53408e+07] (0.95 confidence)</td>
<td>0.987748</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k1&#95;roundtrip</td>
<td>1.57306e+08 ns [1.56982e+08, 1.57651e+08] (0.95 confidence)</td>
<td>1.57708e+08 ns [1.57274e+08, 1.58054e+08] (0.95 confidence)</td>
<td>0.997447</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k1&#95;roundtrip&#95;adversarial</td>
<td>1.62299e+08 ns [1.61073e+08, 1.63835e+08] (0.95 confidence)</td>
<td>1.61667e+08 ns [1.6132e+08, 1.62084e+08] (0.95 confidence)</td>
<td>1.0039</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k2&#95;roundtrip</td>
<td>328062 ns [326674, 329580] (0.95 confidence)</td>
<td>332610 ns [330241, 335157] (0.95 confidence)</td>
<td>0.986327</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k2&#95;roundtrip&#95;adversarial</td>
<td>351734 ns [350907, 352606] (0.95 confidence)</td>
<td>377435 ns [373292, 381530] (0.95 confidence)</td>
<td>0.931907</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k3&#95;roundtrip</td>
<td>3.27221e+08 ns [3.26236e+08, 3.28222e+08] (0.95 confidence)</td>
<td>3.44588e+08 ns [3.43498e+08, 3.45768e+08] (0.95 confidence)</td>
<td>0.949599</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k3&#95;roundtrip&#95;adversarial</td>
<td>9.0326e+08 ns [8.98882e+08, 9.07751e+08] (0.95 confidence)</td>
<td>8.95146e+08 ns [8.93156e+08, 8.97107e+08] (0.95 confidence)</td>
<td>1.00906</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;2d/500</td>
<td>208444 ns [208066, 208818] (0.95 confidence)</td>
<td>209408 ns [208889, 209972] (0.95 confidence)</td>
<td>0.995395</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;2d&#95;adversarial/500</td>
<td>209036 ns [208547, 209531] (0.95 confidence)</td>
<td>209237 ns [208914, 209581] (0.95 confidence)</td>
<td>0.999039</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;3d/100</td>
<td>167124 ns [166635, 167629] (0.95 confidence)</td>
<td>168227 ns [167851, 168641] (0.95 confidence)</td>
<td>0.993444</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;3d&#95;adversarial/100</td>
<td>168808 ns [168284, 169338] (0.95 confidence)</td>
<td>170760 ns [170449, 171122] (0.95 confidence)</td>
<td>0.988569</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;4d/30</td>
<td>133513 ns [133280, 133736] (0.95 confidence)</td>
<td>133634 ns [133391, 133899] (0.95 confidence)</td>
<td>0.999095</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;4d&#95;adversarial/30</td>
<td>132008 ns [131634, 132367] (0.95 confidence)</td>
<td>132697 ns [132381, 133022] (0.95 confidence)</td>
<td>0.994808</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;5d/12</td>
<td>37355.2 ns [37231.9, 37525.3] (0.95 confidence)</td>
<td>37202.7 ns [37052.2, 37343.2] (0.95 confidence)</td>
<td>1.0041</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;5d&#95;adversarial/12</td>
<td>36699.8 ns [36627.3, 36787.1] (0.95 confidence)</td>
<td>36535.3 ns [36470.2, 36599.8] (0.95 confidence)</td>
<td>1.0045</td>
</tr>
<tr>
<td>circumcenter/solve&#95;path/near&#95;singular&#95;exact&#95;fallback&#95;3d</td>
<td>17626.8 ns [17603.3, 17650] (0.95 confidence)</td>
<td>17235.7 ns [17201, 17274.8] (0.95 confidence)</td>
<td>1.02269</td>
</tr>
<tr>
<td>circumcenter/solve&#95;path/regular&#95;lu&#95;3d</td>
<td>23.6307 ns [23.6004, 23.664] (0.95 confidence)</td>
<td>23.4477 ns [23.3938, 23.5074] (0.95 confidence)</td>
<td>1.00781</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;2d/500</td>
<td>392693 ns [392088, 393245] (0.95 confidence)</td>
<td>394853 ns [394321, 395467] (0.95 confidence)</td>
<td>0.99453</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;2d&#95;adversarial/500</td>
<td>388124 ns [387731, 388542] (0.95 confidence)</td>
<td>388342 ns [387819, 388901] (0.95 confidence)</td>
<td>0.999439</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;3d/100</td>
<td>277268 ns [276710, 277810] (0.95 confidence)</td>
<td>278270 ns [278144, 278414] (0.95 confidence)</td>
<td>0.996399</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;3d&#95;adversarial/100</td>
<td>4.02028e+06 ns [4.01014e+06, 4.03096e+06] (0.95 confidence)</td>
<td>4.0658e+06 ns [4.05162e+06, 4.08267e+06] (0.95 confidence)</td>
<td>0.988804</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;4d/30</td>
<td>9.12459e+06 ns [9.10724e+06, 9.14352e+06] (0.95 confidence)</td>
<td>8.93589e+06 ns [8.9141e+06, 8.95653e+06] (0.95 confidence)</td>
<td>1.02112</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;4d&#95;adversarial/30</td>
<td>7.11684e+06 ns [7.10396e+06, 7.1291e+06] (0.95 confidence)</td>
<td>7.08419e+06 ns [7.06994e+06, 7.10012e+06] (0.95 confidence)</td>
<td>1.00461</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;5d/12</td>
<td>3.1249e+06 ns [3.11842e+06, 3.13235e+06] (0.95 confidence)</td>
<td>3.10295e+06 ns [3.09088e+06, 3.11512e+06] (0.95 confidence)</td>
<td>1.00707</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;5d&#95;adversarial/12</td>
<td>2.7148e+06 ns [2.7102e+06, 2.72003e+06] (0.95 confidence)</td>
<td>2.70142e+06 ns [2.69573e+06, 2.70824e+06] (0.95 confidence)</td>
<td>1.00495</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;nearest&#95;visible&#95;facet&#95;3d/100</td>
<td>1178.2 ns [1172.92, 1183.79] (0.95 confidence)</td>
<td>1163.21 ns [1158.25, 1168.25] (0.95 confidence)</td>
<td>1.01289</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;nearest&#95;visible&#95;facet&#95;3d&#95;adversarial/100</td>
<td>41443.1 ns [41251.4, 41646.8] (0.95 confidence)</td>
<td>42464.1 ns [42269.4, 42661] (0.95 confidence)</td>
<td>0.975957</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;visible&#95;facets&#95;3d/100</td>
<td>1268.5 ns [1251.51, 1286.06] (0.95 confidence)</td>
<td>1240.65 ns [1231.1, 1251.85] (0.95 confidence)</td>
<td>1.02245</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;visible&#95;facets&#95;3d&#95;adversarial/100</td>
<td>41438.3 ns [41350.7, 41530.7] (0.95 confidence)</td>
<td>41597.3 ns [41380.9, 41824.8] (0.95 confidence)</td>
<td>0.996179</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/is&#95;point&#95;outside&#95;3d/100</td>
<td>96.4541 ns [96.2763, 96.656] (0.95 confidence)</td>
<td>97.3594 ns [97.2712, 97.4455] (0.95 confidence)</td>
<td>0.990702</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/is&#95;point&#95;outside&#95;3d&#95;adversarial/100</td>
<td>776.671 ns [772.36, 780.847] (0.95 confidence)</td>
<td>773.226 ns [768.17, 778.333] (0.95 confidence)</td>
<td>1.00446</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/boundary&#95;point&#95;distance</td>
<td>13.7038 ns [13.6805, 13.7265] (0.95 confidence)</td>
<td>13.7792 ns [13.761, 13.7969] (0.95 confidence)</td>
<td>0.994525</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/boundary&#95;point&#95;insphere</td>
<td>0.901939 ns [0.898201, 0.905698] (0.95 confidence)</td>
<td>0.835876 ns [0.832804, 0.839078] (0.95 confidence)</td>
<td>1.07903</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/boundary&#95;point&#95;lifted</td>
<td>977.341 ns [975.911, 978.773] (0.95 confidence)</td>
<td>1002.43 ns [1000.89, 1004.04] (0.95 confidence)</td>
<td>0.97497</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/far&#95;point&#95;distance</td>
<td>13.7902 ns [13.7603, 13.82] (0.95 confidence)</td>
<td>13.8056 ns [13.7669, 13.8465] (0.95 confidence)</td>
<td>0.998887</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/far&#95;point&#95;insphere</td>
<td>51.2681 ns [50.9573, 51.5913] (0.95 confidence)</td>
<td>51.9712 ns [51.5761, 52.3857] (0.95 confidence)</td>
<td>0.986471</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/far&#95;point&#95;lifted</td>
<td>50.0493 ns [49.8269, 50.2947] (0.95 confidence)</td>
<td>50.6962 ns [50.3366, 51.0814] (0.95 confidence)</td>
<td>0.98724</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/near&#95;boundary&#95;distance</td>
<td>13.9637 ns [13.9421, 13.9854] (0.95 confidence)</td>
<td>14.0293 ns [14.0005, 14.0598] (0.95 confidence)</td>
<td>0.995326</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/near&#95;boundary&#95;insphere</td>
<td>48.6619 ns [48.3603, 48.9738] (0.95 confidence)</td>
<td>49.3208 ns [49.0379, 49.6116] (0.95 confidence)</td>
<td>0.98664</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/near&#95;boundary&#95;lifted</td>
<td>47.5632 ns [47.3879, 47.739] (0.95 confidence)</td>
<td>48.068 ns [47.8238, 48.3294] (0.95 confidence)</td>
<td>0.989497</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/boundary&#95;point&#95;distance</td>
<td>33.3479 ns [33.2801, 33.4173] (0.95 confidence)</td>
<td>33.0928 ns [33.0305, 33.1577] (0.95 confidence)</td>
<td>1.00771</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/boundary&#95;point&#95;insphere</td>
<td>1.19696 ns [1.19521, 1.19851] (0.95 confidence)</td>
<td>1.18014 ns [1.17417, 1.18547] (0.95 confidence)</td>
<td>1.01425</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/boundary&#95;point&#95;lifted</td>
<td>1790.84 ns [1788.41, 1793.36] (0.95 confidence)</td>
<td>1854.93 ns [1851.96, 1858.22] (0.95 confidence)</td>
<td>0.96545</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/far&#95;point&#95;distance</td>
<td>33.4682 ns [33.3747, 33.5655] (0.95 confidence)</td>
<td>33.565 ns [33.4251, 33.714] (0.95 confidence)</td>
<td>0.997116</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/far&#95;point&#95;insphere</td>
<td>108.102 ns [107.791, 108.434] (0.95 confidence)</td>
<td>113.24 ns [112.653, 113.857] (0.95 confidence)</td>
<td>0.954625</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/far&#95;point&#95;lifted</td>
<td>109.648 ns [109.028, 110.288] (0.95 confidence)</td>
<td>112.119 ns [111.557, 112.694] (0.95 confidence)</td>
<td>0.977962</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/near&#95;boundary&#95;distance</td>
<td>33.1906 ns [33.116, 33.2686] (0.95 confidence)</td>
<td>33.5933 ns [33.4892, 33.7002] (0.95 confidence)</td>
<td>0.988011</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/near&#95;boundary&#95;insphere</td>
<td>98.3933 ns [97.5722, 99.2666] (0.95 confidence)</td>
<td>100.15 ns [99.6316, 100.686] (0.95 confidence)</td>
<td>0.982458</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/near&#95;boundary&#95;lifted</td>
<td>98.4437 ns [97.681, 99.2398] (0.95 confidence)</td>
<td>101.058 ns [100.414, 101.71] (0.95 confidence)</td>
<td>0.974131</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/boundary&#95;point&#95;distance</td>
<td>54.9071 ns [54.7624, 55.0547] (0.95 confidence)</td>
<td>55.6488 ns [55.5071, 55.7881] (0.95 confidence)</td>
<td>0.986671</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/boundary&#95;point&#95;insphere</td>
<td>1.59016 ns [1.56693, 1.61169] (0.95 confidence)</td>
<td>1.5686 ns [1.52777, 1.60856] (0.95 confidence)</td>
<td>1.01374</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/boundary&#95;point&#95;lifted</td>
<td>3828.45 ns [3820.12, 3837.01] (0.95 confidence)</td>
<td>3829.13 ns [3824.05, 3834.38] (0.95 confidence)</td>
<td>0.999824</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/far&#95;point&#95;distance</td>
<td>54.5274 ns [54.4356, 54.6196] (0.95 confidence)</td>
<td>54.5285 ns [54.4626, 54.5937] (0.95 confidence)</td>
<td>0.999979</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/far&#95;point&#95;insphere</td>
<td>774.611 ns [770.719, 778.956] (0.95 confidence)</td>
<td>744.623 ns [742.199, 747.144] (0.95 confidence)</td>
<td>1.04027</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/far&#95;point&#95;lifted</td>
<td>734.802 ns [732.482, 737.111] (0.95 confidence)</td>
<td>751.067 ns [748.956, 753.208] (0.95 confidence)</td>
<td>0.978344</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/near&#95;boundary&#95;distance</td>
<td>55.4401 ns [55.3324, 55.5511] (0.95 confidence)</td>
<td>54.9416 ns [54.8565, 55.0375] (0.95 confidence)</td>
<td>1.00907</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/near&#95;boundary&#95;insphere</td>
<td>727.74 ns [724.455, 731.157] (0.95 confidence)</td>
<td>735.624 ns [732.265, 739.118] (0.95 confidence)</td>
<td>0.989282</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/near&#95;boundary&#95;lifted</td>
<td>714.456 ns [712.989, 715.919] (0.95 confidence)</td>
<td>728.446 ns [726.11, 730.851] (0.95 confidence)</td>
<td>0.980794</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/boundary&#95;point&#95;distance</td>
<td>88.2979 ns [88.0362, 88.5625] (0.95 confidence)</td>
<td>88.1676 ns [87.8924, 88.4497] (0.95 confidence)</td>
<td>1.00148</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/boundary&#95;point&#95;insphere</td>
<td>1.75741 ns [1.74059, 1.77557] (0.95 confidence)</td>
<td>1.75635 ns [1.73634, 1.77821] (0.95 confidence)</td>
<td>1.00061</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/boundary&#95;point&#95;lifted</td>
<td>5809.73 ns [5800.39, 5819.69] (0.95 confidence)</td>
<td>5929.68 ns [5919.96, 5939.71] (0.95 confidence)</td>
<td>0.979772</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/far&#95;point&#95;distance</td>
<td>88.0901 ns [87.7967, 88.3909] (0.95 confidence)</td>
<td>87.989 ns [87.6943, 88.284] (0.95 confidence)</td>
<td>1.00115</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/far&#95;point&#95;insphere</td>
<td>1290.16 ns [1285.95, 1294.6] (0.95 confidence)</td>
<td>1306.39 ns [1301.6, 1311.22] (0.95 confidence)</td>
<td>0.987577</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/far&#95;point&#95;lifted</td>
<td>1291.77 ns [1288.57, 1294.97] (0.95 confidence)</td>
<td>1294.36 ns [1289.75, 1299.1] (0.95 confidence)</td>
<td>0.998005</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/near&#95;boundary&#95;distance</td>
<td>88.3442 ns [88.1143, 88.5782] (0.95 confidence)</td>
<td>88.6553 ns [88.1413, 89.2247] (0.95 confidence)</td>
<td>0.996491</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/near&#95;boundary&#95;insphere</td>
<td>1262.84 ns [1258.34, 1267.55] (0.95 confidence)</td>
<td>1279.12 ns [1275.83, 1282.47] (0.95 confidence)</td>
<td>0.987272</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/near&#95;boundary&#95;lifted</td>
<td>1285.17 ns [1280.8, 1289.67] (0.95 confidence)</td>
<td>1321.86 ns [1312.77, 1331.8] (0.95 confidence)</td>
<td>0.972244</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;2d/vertices&#95;120&#95;simplices&#95;226</td>
<td>1.61444e+06 ns [1.61213e+06, 1.6168e+06] (0.95 confidence)</td>
<td>1.62166e+06 ns [1.61785e+06, 1.62586e+06] (0.95 confidence)</td>
<td>0.995546</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;3d/vertices&#95;30&#95;simplices&#95;115</td>
<td>1.58375e+08 ns [1.58109e+08, 1.58688e+08] (0.95 confidence)</td>
<td>1.65264e+08 ns [1.64861e+08, 1.65701e+08] (0.95 confidence)</td>
<td>0.958315</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;4d/vertices&#95;16&#95;simplices&#95;68</td>
<td>4.65847e+08 ns [4.64969e+08, 4.66736e+08] (0.95 confidence)</td>
<td>4.92357e+08 ns [4.91472e+08, 4.93261e+08] (0.95 confidence)</td>
<td>0.946157</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;5d/vertices&#95;10&#95;simplices&#95;20</td>
<td>1.35164e+08 ns [1.34962e+08, 1.35422e+08] (0.95 confidence)</td>
<td>1.44255e+08 ns [1.43699e+08, 1.44776e+08] (0.95 confidence)</td>
<td>0.936981</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;2d/10</td>
<td>453722 ns [452909, 454669] (0.95 confidence)</td>
<td>452605 ns [451692, 453529] (0.95 confidence)</td>
<td>1.00247</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;2d&#95;adversarial/10</td>
<td>2.89669e+06 ns [2.89296e+06, 2.90076e+06] (0.95 confidence)</td>
<td>2.91464e+06 ns [2.9124e+06, 2.91702e+06] (0.95 confidence)</td>
<td>0.99384</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;3d/10</td>
<td>2.8069e+06 ns [2.80086e+06, 2.81362e+06] (0.95 confidence)</td>
<td>2.79699e+06 ns [2.78513e+06, 2.80988e+06] (0.95 confidence)</td>
<td>1.00354</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;3d&#95;adversarial/10</td>
<td>1.36707e+07 ns [1.36417e+07, 1.37043e+07] (0.95 confidence)</td>
<td>1.3612e+07 ns [1.35803e+07, 1.36477e+07] (0.95 confidence)</td>
<td>1.00431</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;4d/6</td>
<td>2.70485e+07 ns [2.68885e+07, 2.72326e+07] (0.95 confidence)</td>
<td>2.68945e+07 ns [2.68061e+07, 2.70015e+07] (0.95 confidence)</td>
<td>1.00572</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;4d&#95;adversarial/6</td>
<td>6.724e+07 ns [6.71036e+07, 6.73677e+07] (0.95 confidence)</td>
<td>6.79137e+07 ns [6.74519e+07, 6.84205e+07] (0.95 confidence)</td>
<td>0.99008</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;5d/4</td>
<td>5.15865e+07 ns [5.14832e+07, 5.17358e+07] (0.95 confidence)</td>
<td>5.16502e+07 ns [5.14395e+07, 5.18701e+07] (0.95 confidence)</td>
<td>0.998767</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;5d&#95;adversarial/4</td>
<td>5.56133e+07 ns [5.55132e+07, 5.57259e+07] (0.95 confidence)</td>
<td>5.59026e+07 ns [5.57378e+07, 5.60877e+07] (0.95 confidence)</td>
<td>0.994826</td>
</tr>
<tr>
<td>locate/exact&#95;hint/2d/vertices&#95;100&#95;simplices&#95;187&#95;seed&#95;dd618c77b9187836&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>7092.9 ns [7085.33, 7099.9] (0.95 confidence)</td>
<td>7124.46 ns [7106.6, 7141.9] (0.95 confidence)</td>
<td>0.99557</td>
</tr>
<tr>
<td>locate/exact&#95;hint/2d/vertices&#95;2000&#95;simplices&#95;3978&#95;seed&#95;01acf93075e96412&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>7304 ns [7298.83, 7309.27] (0.95 confidence)</td>
<td>7343.13 ns [7329.85, 7354.29] (0.95 confidence)</td>
<td>0.994671</td>
</tr>
<tr>
<td>locate/exact&#95;hint/2d/vertices&#95;500&#95;simplices&#95;983&#95;seed&#95;1493be4e9d7a5906&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>7467.95 ns [7450.06, 7486.12] (0.95 confidence)</td>
<td>7445.73 ns [7431.13, 7460.26] (0.95 confidence)</td>
<td>1.00298</td>
</tr>
<tr>
<td>locate/exact&#95;hint/3d/vertices&#95;150&#95;simplices&#95;829&#95;seed&#95;a44b52b395a4b44d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>14955.6 ns [14920.9, 14985.7] (0.95 confidence)</td>
<td>14879.8 ns [14811.7, 14956.6] (0.95 confidence)</td>
<td>1.00509</td>
</tr>
<tr>
<td>locate/exact&#95;hint/3d/vertices&#95;500&#95;simplices&#95;3052&#95;seed&#95;1493be4f9d7a5907&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>14934.6 ns [14889.6, 14975.8] (0.95 confidence)</td>
<td>14706.1 ns [14661.7, 14748.4] (0.95 confidence)</td>
<td>1.01554</td>
</tr>
<tr>
<td>locate/exact&#95;hint/3d/vertices&#95;50&#95;simplices&#95;222&#95;seed&#95;f61fc639dc8c3c19&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>14906.2 ns [14874.7, 14938.9] (0.95 confidence)</td>
<td>14824.3 ns [14760.1, 14897] (0.95 confidence)</td>
<td>1.00553</td>
</tr>
<tr>
<td>locate/exact&#95;hint/4d/vertices&#95;100&#95;simplices&#95;1677&#95;seed&#95;dd618c71b9187830&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>2.00265e+06 ns [1.99722e+06, 2.00806e+06] (0.95 confidence)</td>
<td>2.01105e+06 ns [2.008e+06, 2.01443e+06] (0.95 confidence)</td>
<td>0.995822</td>
</tr>
<tr>
<td>locate/exact&#95;hint/4d/vertices&#95;50&#95;simplices&#95;626&#95;seed&#95;f61fc63edc8c3c1e&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>1.96394e+06 ns [1.95916e+06, 1.96892e+06] (0.95 confidence)</td>
<td>1.99022e+06 ns [1.98488e+06, 1.99576e+06] (0.95 confidence)</td>
<td>0.986797</td>
</tr>
<tr>
<td>locate/exact&#95;hint/5d/vertices&#95;25&#95;simplices&#95;410&#95;seed&#95;63a0e3186e461e08&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>4.74646e+06 ns [4.70902e+06, 4.78336e+06] (0.95 confidence)</td>
<td>4.82014e+06 ns [4.80615e+06, 4.83584e+06] (0.95 confidence)</td>
<td>0.984713</td>
</tr>
<tr>
<td>locate/exact&#95;hint/5d/vertices&#95;40&#95;simplices&#95;1208&#95;seed&#95;a86104fee3a3634d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>5.09459e+06 ns [5.05858e+06, 5.13575e+06] (0.95 confidence)</td>
<td>4.87681e+06 ns [4.85763e+06, 4.89403e+06] (0.95 confidence)</td>
<td>1.04466</td>
</tr>
<tr>
<td>locate/no&#95;hint/2d/vertices&#95;100&#95;simplices&#95;187&#95;seed&#95;dd618c77b9187836&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>73942.4 ns [73772.7, 74101.4] (0.95 confidence)</td>
<td>73803.7 ns [73568.6, 74054.8] (0.95 confidence)</td>
<td>1.00188</td>
</tr>
<tr>
<td>locate/no&#95;hint/2d/vertices&#95;2000&#95;simplices&#95;3978&#95;seed&#95;01acf93075e96412&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>266691 ns [265339, 268026] (0.95 confidence)</td>
<td>268703 ns [268173, 269199] (0.95 confidence)</td>
<td>0.992511</td>
</tr>
<tr>
<td>locate/no&#95;hint/2d/vertices&#95;500&#95;simplices&#95;983&#95;seed&#95;1493be4e9d7a5906&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>123111 ns [122854, 123407] (0.95 confidence)</td>
<td>122945 ns [122694, 123174] (0.95 confidence)</td>
<td>1.00135</td>
</tr>
<tr>
<td>locate/no&#95;hint/3d/vertices&#95;150&#95;simplices&#95;829&#95;seed&#95;a44b52b395a4b44d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>125032 ns [124725, 125363] (0.95 confidence)</td>
<td>124157 ns [123881, 124455] (0.95 confidence)</td>
<td>1.00705</td>
</tr>
<tr>
<td>locate/no&#95;hint/3d/vertices&#95;500&#95;simplices&#95;3052&#95;seed&#95;1493be4f9d7a5907&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>194656 ns [192926, 196363] (0.95 confidence)</td>
<td>193301 ns [191237, 195611] (0.95 confidence)</td>
<td>1.00701</td>
</tr>
<tr>
<td>locate/no&#95;hint/3d/vertices&#95;50&#95;simplices&#95;222&#95;seed&#95;f61fc639dc8c3c19&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>66280 ns [65923.2, 66675.1] (0.95 confidence)</td>
<td>66004.4 ns [65609.4, 66430.3] (0.95 confidence)</td>
<td>1.00417</td>
</tr>
<tr>
<td>locate/no&#95;hint/4d/vertices&#95;100&#95;simplices&#95;1677&#95;seed&#95;dd618c71b9187830&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>1.13152e+07 ns [1.12748e+07, 1.13646e+07] (0.95 confidence)</td>
<td>1.14584e+07 ns [1.14419e+07, 1.14742e+07] (0.95 confidence)</td>
<td>0.987508</td>
</tr>
<tr>
<td>locate/no&#95;hint/4d/vertices&#95;50&#95;simplices&#95;626&#95;seed&#95;f61fc63edc8c3c1e&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>1.16907e+07 ns [1.14344e+07, 1.20162e+07] (0.95 confidence)</td>
<td>1.13602e+07 ns [1.12981e+07, 1.14269e+07] (0.95 confidence)</td>
<td>1.0291</td>
</tr>
<tr>
<td>locate/no&#95;hint/5d/vertices&#95;25&#95;simplices&#95;410&#95;seed&#95;63a0e3186e461e08&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>2.22476e+07 ns [2.21273e+07, 2.23729e+07] (0.95 confidence)</td>
<td>2.31222e+07 ns [2.28316e+07, 2.34338e+07] (0.95 confidence)</td>
<td>0.962176</td>
</tr>
<tr>
<td>locate/no&#95;hint/5d/vertices&#95;40&#95;simplices&#95;1208&#95;seed&#95;a86104fee3a3634d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>3.9935e+07 ns [3.96591e+07, 4.0301e+07] (0.95 confidence)</td>
<td>4.07753e+07 ns [4.05563e+07, 4.09535e+07] (0.95 confidence)</td>
<td>0.979391</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;2d/simplices&#95;226</td>
<td>120332 ns [119030, 121833] (0.95 confidence)</td>
<td>123476 ns [121599, 125443] (0.95 confidence)</td>
<td>0.974535</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;3d/simplices&#95;115</td>
<td>148251 ns [147010, 149719] (0.95 confidence)</td>
<td>149869 ns [148718, 151553] (0.95 confidence)</td>
<td>0.989203</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;4d/simplices&#95;68</td>
<td>1.24598e+06 ns [1.22595e+06, 1.2721e+06] (0.95 confidence)</td>
<td>1.26246e+06 ns [1.25242e+06, 1.27327e+06] (0.95 confidence)</td>
<td>0.986948</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;5d/simplices&#95;20</td>
<td>464620 ns [462469, 466954] (0.95 confidence)</td>
<td>479750 ns [475333, 483723] (0.95 confidence)</td>
<td>0.968463</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;2d/simplices&#95;226</td>
<td>1.36242e+06 ns [1.35611e+06, 1.36974e+06] (0.95 confidence)</td>
<td>1.35406e+06 ns [1.35012e+06, 1.35813e+06] (0.95 confidence)</td>
<td>1.00617</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;3d/simplices&#95;115</td>
<td>1.57569e+08 ns [1.5743e+08, 1.57702e+08] (0.95 confidence)</td>
<td>1.64295e+08 ns [1.64141e+08, 1.64443e+08] (0.95 confidence)</td>
<td>0.959064</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;4d/simplices&#95;68</td>
<td>4.66297e+08 ns [4.63922e+08, 4.697e+08] (0.95 confidence)</td>
<td>4.89417e+08 ns [4.89051e+08, 4.89866e+08] (0.95 confidence)</td>
<td>0.95276</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;5d/simplices&#95;20</td>
<td>1.3484e+08 ns [1.3473e+08, 1.34981e+08] (0.95 confidence)</td>
<td>1.4196e+08 ns [1.41571e+08, 1.424e+08] (0.95 confidence)</td>
<td>0.949848</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;2d/simplices&#95;226</td>
<td>1.31622e+06 ns [1.30523e+06, 1.32959e+06] (0.95 confidence)</td>
<td>1.32068e+06 ns [1.28822e+06, 1.35777e+06] (0.95 confidence)</td>
<td>0.996629</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;3d/simplices&#95;115</td>
<td>1.58459e+08 ns [1.57582e+08, 1.59505e+08] (0.95 confidence)</td>
<td>1.64378e+08 ns [1.64102e+08, 1.64778e+08] (0.95 confidence)</td>
<td>0.96399</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;4d/simplices&#95;68</td>
<td>4.63854e+08 ns [4.63283e+08, 4.64443e+08] (0.95 confidence)</td>
<td>4.88643e+08 ns [4.88156e+08, 4.89173e+08] (0.95 confidence)</td>
<td>0.94927</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;5d/simplices&#95;20</td>
<td>1.3501e+08 ns [1.34452e+08, 1.35665e+08] (0.95 confidence)</td>
<td>1.41494e+08 ns [1.41257e+08, 1.41781e+08] (0.95 confidence)</td>
<td>0.95418</td>
</tr>
<tr>
<td>random/insphere&#95;1000&#95;queries</td>
<td>438376 ns [436815, 439916] (0.95 confidence)</td>
<td>434503 ns [432656, 436322] (0.95 confidence)</td>
<td>1.00891</td>
</tr>
<tr>
<td>random/insphere&#95;distance&#95;1000&#95;queries</td>
<td>66301.1 ns [66150.8, 66449.3] (0.95 confidence)</td>
<td>67362.2 ns [67216.8, 67507.4] (0.95 confidence)</td>
<td>0.984248</td>
</tr>
<tr>
<td>random/insphere&#95;lifted&#95;1000&#95;queries</td>
<td>439174 ns [437598, 440743] (0.95 confidence)</td>
<td>430987 ns [429232, 432785] (0.95 confidence)</td>
<td>1.01899</td>
</tr>
<tr>
<td>tds&#95;new&#95;2d/tds&#95;new/500</td>
<td>4.71743e+07 ns [4.70457e+07, 4.73463e+07] (0.95 confidence)</td>
<td>4.60882e+07 ns [4.58514e+07, 4.63443e+07] (0.95 confidence)</td>
<td>1.02357</td>
</tr>
<tr>
<td>tds&#95;new&#95;2d/tds&#95;new&#95;adversarial/500</td>
<td>5.52376e+07 ns [5.51349e+07, 5.53567e+07] (0.95 confidence)</td>
<td>5.52924e+07 ns [5.51721e+07, 5.54287e+07] (0.95 confidence)</td>
<td>0.999009</td>
</tr>
<tr>
<td>tds&#95;new&#95;3d/tds&#95;new/100</td>
<td>3.84615e+07 ns [3.82018e+07, 3.8726e+07] (0.95 confidence)</td>
<td>3.82655e+07 ns [3.81813e+07, 3.83772e+07] (0.95 confidence)</td>
<td>1.00512</td>
</tr>
<tr>
<td>tds&#95;new&#95;3d/tds&#95;new&#95;adversarial/100</td>
<td>8.76261e+07 ns [8.67932e+07, 8.87077e+07] (0.95 confidence)</td>
<td>8.66574e+07 ns [8.64032e+07, 8.69242e+07] (0.95 confidence)</td>
<td>1.01118</td>
</tr>
<tr>
<td>tds&#95;new&#95;4d/tds&#95;new/30</td>
<td>1.0287e+08 ns [1.01643e+08, 1.04689e+08] (0.95 confidence)</td>
<td>9.89932e+07 ns [9.83599e+07, 9.9571e+07] (0.95 confidence)</td>
<td>1.03917</td>
</tr>
<tr>
<td>tds&#95;new&#95;4d/tds&#95;new&#95;adversarial/30</td>
<td>9.44786e+07 ns [9.35694e+07, 9.55177e+07] (0.95 confidence)</td>
<td>9.43556e+07 ns [9.33446e+07, 9.57945e+07] (0.95 confidence)</td>
<td>1.0013</td>
</tr>
<tr>
<td>tds&#95;new&#95;5d/tds&#95;new/12</td>
<td>2.19815e+07 ns [2.18986e+07, 2.20749e+07] (0.95 confidence)</td>
<td>2.19248e+07 ns [2.18349e+07, 2.20246e+07] (0.95 confidence)</td>
<td>1.00259</td>
</tr>
<tr>
<td>tds&#95;new&#95;5d/tds&#95;new&#95;adversarial/12</td>
<td>2.03956e+07 ns [2.03436e+07, 2.04603e+07] (0.95 confidence)</td>
<td>2.01764e+07 ns [2.00793e+07, 2.02828e+07] (0.95 confidence)</td>
<td>1.01086</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;2d/500</td>
<td>796112 ns [792048, 800059] (0.95 confidence)</td>
<td>823865 ns [814741, 834106] (0.95 confidence)</td>
<td>0.966314</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;2d&#95;adversarial/500</td>
<td>1.10387e+06 ns [1.09953e+06, 1.10845e+06] (0.95 confidence)</td>
<td>1.10439e+06 ns [1.09642e+06, 1.11279e+06] (0.95 confidence)</td>
<td>0.999526</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;3d/100</td>
<td>1.2608e+06 ns [1.25373e+06, 1.26899e+06] (0.95 confidence)</td>
<td>1.2777e+06 ns [1.26733e+06, 1.28842e+06] (0.95 confidence)</td>
<td>0.986773</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;3d&#95;adversarial/100</td>
<td>2.26253e+06 ns [2.24704e+06, 2.27874e+06] (0.95 confidence)</td>
<td>2.24668e+06 ns [2.23497e+06, 2.25957e+06] (0.95 confidence)</td>
<td>1.00706</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;4d/30</td>
<td>4.75511e+06 ns [4.71376e+06, 4.79825e+06] (0.95 confidence)</td>
<td>4.85448e+06 ns [4.83007e+06, 4.8783e+06] (0.95 confidence)</td>
<td>0.979531</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;4d&#95;adversarial/30</td>
<td>4.6774e+06 ns [4.63321e+06, 4.72064e+06] (0.95 confidence)</td>
<td>4.66636e+06 ns [4.62533e+06, 4.70744e+06] (0.95 confidence)</td>
<td>1.00237</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;5d/12</td>
<td>2.14094e+06 ns [2.1361e+06, 2.1475e+06] (0.95 confidence)</td>
<td>2.18163e+06 ns [2.17067e+06, 2.19345e+06] (0.95 confidence)</td>
<td>0.98135</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;5d&#95;adversarial/12</td>
<td>1.91552e+06 ns [1.91165e+06, 1.9202e+06] (0.95 confidence)</td>
<td>1.96688e+06 ns [1.95392e+06, 1.98177e+06] (0.95 confidence)</td>
<td>0.97389</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;2d/500</td>
<td>800331 ns [795357, 805793] (0.95 confidence)</td>
<td>815218 ns [806486, 824405] (0.95 confidence)</td>
<td>0.981739</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;2d&#95;adversarial/500</td>
<td>1.10155e+06 ns [1.09749e+06, 1.10586e+06] (0.95 confidence)</td>
<td>1.10268e+06 ns [1.09553e+06, 1.11086e+06] (0.95 confidence)</td>
<td>0.998982</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;3d/100</td>
<td>1.25953e+06 ns [1.2535e+06, 1.26661e+06] (0.95 confidence)</td>
<td>1.26906e+06 ns [1.25836e+06, 1.2804e+06] (0.95 confidence)</td>
<td>0.992484</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;3d&#95;adversarial/100</td>
<td>2.27408e+06 ns [2.26468e+06, 2.28579e+06] (0.95 confidence)</td>
<td>2.25084e+06 ns [2.23518e+06, 2.26708e+06] (0.95 confidence)</td>
<td>1.01032</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;4d/30</td>
<td>4.71441e+06 ns [4.67601e+06, 4.75858e+06] (0.95 confidence)</td>
<td>4.79289e+06 ns [4.75915e+06, 4.82555e+06] (0.95 confidence)</td>
<td>0.983626</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;4d&#95;adversarial/30</td>
<td>4.61384e+06 ns [4.58378e+06, 4.64843e+06] (0.95 confidence)</td>
<td>4.66849e+06 ns [4.63343e+06, 4.70517e+06] (0.95 confidence)</td>
<td>0.988293</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;5d/12</td>
<td>2.13785e+06 ns [2.13043e+06, 2.14635e+06] (0.95 confidence)</td>
<td>2.1853e+06 ns [2.17408e+06, 2.19696e+06] (0.95 confidence)</td>
<td>0.978286</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;5d&#95;adversarial/12</td>
<td>1.92845e+06 ns [1.9215e+06, 1.93596e+06] (0.95 confidence)</td>
<td>1.96791e+06 ns [1.95676e+06, 1.97942e+06] (0.95 confidence)</td>
<td>0.979946</td>
</tr>
<tr>
<td>validation/validate&#95;2d/500</td>
<td>8.42149e+06 ns [8.35032e+06, 8.54001e+06] (0.95 confidence)</td>
<td>8.22932e+06 ns [8.20092e+06, 8.26624e+06] (0.95 confidence)</td>
<td>1.02335</td>
</tr>
<tr>
<td>validation/validate&#95;2d&#95;adversarial/500</td>
<td>4.97753e+07 ns [4.96188e+07, 4.99798e+07] (0.95 confidence)</td>
<td>4.96874e+07 ns [4.93913e+07, 5.00446e+07] (0.95 confidence)</td>
<td>1.00177</td>
</tr>
<tr>
<td>validation/validate&#95;3d/100</td>
<td>8.54654e+08 ns [8.53564e+08, 8.55723e+08] (0.95 confidence)</td>
<td>8.8752e+08 ns [8.84724e+08, 8.90519e+08] (0.95 confidence)</td>
<td>0.962969</td>
</tr>
<tr>
<td>validation/validate&#95;3d&#95;adversarial/100</td>
<td>1.04574e+09 ns [1.0449e+09, 1.04665e+09] (0.95 confidence)</td>
<td>1.09782e+09 ns [1.08803e+09, 1.10857e+09] (0.95 confidence)</td>
<td>0.95256</td>
</tr>
<tr>
<td>validation/validate&#95;4d/30</td>
<td>3.88478e+09 ns [3.87372e+09, 3.89668e+09] (0.95 confidence)</td>
<td>4.11934e+09 ns [4.08604e+09, 4.1576e+09] (0.95 confidence)</td>
<td>0.943059</td>
</tr>
<tr>
<td>validation/validate&#95;4d&#95;adversarial/30</td>
<td>3.94786e+09 ns [3.94067e+09, 3.95643e+09] (0.95 confidence)</td>
<td>4.17445e+09 ns [4.16082e+09, 4.19017e+09] (0.95 confidence)</td>
<td>0.94572</td>
</tr>
<tr>
<td>validation/validate&#95;5d/12</td>
<td>8.75957e+08 ns [8.75198e+08, 8.7681e+08] (0.95 confidence)</td>
<td>9.29108e+08 ns [9.25809e+08, 9.33453e+08] (0.95 confidence)</td>
<td>0.942794</td>
</tr>
<tr>
<td>validation/validate&#95;5d&#95;adversarial/12</td>
<td>8.06135e+08 ns [8.04927e+08, 8.07453e+08] (0.95 confidence)</td>
<td>8.49113e+08 ns [8.47238e+08, 8.51175e+08] (0.95 confidence)</td>
<td>0.949385</td>
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
<td>51.8362 ns [51.2366, 52.1585] (0.95 confidence)</td>
<td>51.8949 ns [51.529, 52.4116] (0.95 confidence)</td>
<td>0.99887</td>
</tr>
<tr>
<td>2d/insphere&#95;distance</td>
<td>13.8599 ns [13.8307, 13.8979] (0.95 confidence)</td>
<td>13.8175 ns [13.7365, 13.8541] (0.95 confidence)</td>
<td>1.00307</td>
</tr>
<tr>
<td>2d/insphere&#95;lifted</td>
<td>51.233 ns [50.9942, 51.5585] (0.95 confidence)</td>
<td>52.2876 ns [52.0918, 52.5755] (0.95 confidence)</td>
<td>0.979832</td>
</tr>
<tr>
<td>3d/insphere</td>
<td>109.735 ns [109.299, 110.63] (0.95 confidence)</td>
<td>110.378 ns [109.993, 110.801] (0.95 confidence)</td>
<td>0.994177</td>
</tr>
<tr>
<td>3d/insphere&#95;distance</td>
<td>33.4779 ns [33.4077, 33.6805] (0.95 confidence)</td>
<td>33.0664 ns [32.9972, 33.1251] (0.95 confidence)</td>
<td>1.01245</td>
</tr>
<tr>
<td>3d/insphere&#95;lifted</td>
<td>107.69 ns [107.203, 108.045] (0.95 confidence)</td>
<td>110.237 ns [109.789, 110.921] (0.95 confidence)</td>
<td>0.976896</td>
</tr>
<tr>
<td>4d/insphere</td>
<td>747.254 ns [743.935, 751.983] (0.95 confidence)</td>
<td>749.212 ns [744.653, 750.903] (0.95 confidence)</td>
<td>0.997386</td>
</tr>
<tr>
<td>4d/insphere&#95;distance</td>
<td>54.2687 ns [54.2542, 54.2938] (0.95 confidence)</td>
<td>54.2894 ns [54.2581, 54.4273] (0.95 confidence)</td>
<td>0.999618</td>
</tr>
<tr>
<td>4d/insphere&#95;lifted</td>
<td>740.927 ns [738.385, 744.717] (0.95 confidence)</td>
<td>751.506 ns [749.536, 753.424] (0.95 confidence)</td>
<td>0.985923</td>
</tr>
<tr>
<td>5d/insphere</td>
<td>1321.61 ns [1317.08, 1325.46] (0.95 confidence)</td>
<td>1293.93 ns [1289.05, 1304.66] (0.95 confidence)</td>
<td>1.02139</td>
</tr>
<tr>
<td>5d/insphere&#95;distance</td>
<td>88.3579 ns [88.1652, 88.4853] (0.95 confidence)</td>
<td>88.392 ns [88.1315, 88.5883] (0.95 confidence)</td>
<td>0.999614</td>
</tr>
<tr>
<td>5d/insphere&#95;lifted</td>
<td>1305.27 ns [1301.4, 1312.06] (0.95 confidence)</td>
<td>1317.13 ns [1307.06, 1321.49] (0.95 confidence)</td>
<td>0.990997</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k1&#95;roundtrip</td>
<td>53665.2 ns [53644, 53704.8] (0.95 confidence)</td>
<td>55089.4 ns [55044.1, 55280] (0.95 confidence)</td>
<td>0.974148</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k1&#95;roundtrip&#95;adversarial</td>
<td>164875 ns [164554, 165356] (0.95 confidence)</td>
<td>167692 ns [166777, 171837] (0.95 confidence)</td>
<td>0.983203</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k2&#95;edge&#95;flip</td>
<td>26766.8 ns [26650, 27513] (0.95 confidence)</td>
<td>27413.1 ns [27043, 28343.3] (0.95 confidence)</td>
<td>0.976424</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k2&#95;edge&#95;flip&#95;adversarial</td>
<td>101681 ns [101502, 102219] (0.95 confidence)</td>
<td>101965 ns [101872, 102078] (0.95 confidence)</td>
<td>0.997211</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k1&#95;roundtrip</td>
<td>2.36638e+06 ns [2.36306e+06, 2.36821e+06] (0.95 confidence)</td>
<td>2.36202e+06 ns [2.35303e+06, 2.37303e+06] (0.95 confidence)</td>
<td>1.00184</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k1&#95;roundtrip&#95;adversarial</td>
<td>1.97131e+06 ns [1.96387e+06, 1.97599e+06] (0.95 confidence)</td>
<td>1.93961e+06 ns [1.93292e+06, 1.94944e+06] (0.95 confidence)</td>
<td>1.01635</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k2&#95;roundtrip</td>
<td>6.49442e+06 ns [6.4798e+06, 6.51515e+06] (0.95 confidence)</td>
<td>6.6696e+06 ns [6.66147e+06, 6.69137e+06] (0.95 confidence)</td>
<td>0.973733</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k2&#95;roundtrip&#95;adversarial</td>
<td>5.93294e+06 ns [5.92268e+06, 5.93911e+06] (0.95 confidence)</td>
<td>5.99031e+06 ns [5.97918e+06, 6.03122e+06] (0.95 confidence)</td>
<td>0.990423</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k3&#95;forward</td>
<td>1.39016e+06 ns [1.38279e+06, 1.40097e+06] (0.95 confidence)</td>
<td>1.38727e+06 ns [1.38512e+06, 1.39033e+06] (0.95 confidence)</td>
<td>1.00209</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k3&#95;forward&#95;adversarial</td>
<td>1.83462e+06 ns [1.83124e+06, 1.84293e+06] (0.95 confidence)</td>
<td>1.87065e+06 ns [1.86115e+06, 1.8809e+06] (0.95 confidence)</td>
<td>0.980739</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k1&#95;roundtrip</td>
<td>3.88635e+07 ns [3.88231e+07, 3.89165e+07] (0.95 confidence)</td>
<td>3.90445e+07 ns [3.89734e+07, 3.9246e+07] (0.95 confidence)</td>
<td>0.995365</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k1&#95;roundtrip&#95;adversarial</td>
<td>3.28979e+07 ns [3.28187e+07, 3.29917e+07] (0.95 confidence)</td>
<td>3.30554e+07 ns [3.29848e+07, 3.32068e+07] (0.95 confidence)</td>
<td>0.995233</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k2&#95;roundtrip</td>
<td>5.53838e+07 ns [5.53046e+07, 5.55841e+07] (0.95 confidence)</td>
<td>5.65643e+07 ns [5.64912e+07, 5.66487e+07] (0.95 confidence)</td>
<td>0.97913</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k2&#95;roundtrip&#95;adversarial</td>
<td>1.94585e+08 ns [1.94078e+08, 1.96252e+08] (0.95 confidence)</td>
<td>1.96697e+08 ns [1.96533e+08, 1.97063e+08] (0.95 confidence)</td>
<td>0.989262</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k3&#95;roundtrip</td>
<td>6.36385e+07 ns [6.33403e+07, 6.37594e+07] (0.95 confidence)</td>
<td>6.5908e+07 ns [6.58294e+07, 6.60502e+07] (0.95 confidence)</td>
<td>0.965565</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k3&#95;roundtrip&#95;adversarial</td>
<td>3.47461e+07 ns [3.46812e+07, 3.49702e+07] (0.95 confidence)</td>
<td>3.51809e+07 ns [3.51481e+07, 3.53371e+07] (0.95 confidence)</td>
<td>0.987641</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k1&#95;roundtrip</td>
<td>1.57249e+08 ns [1.56849e+08, 1.57814e+08] (0.95 confidence)</td>
<td>1.57775e+08 ns [1.57509e+08, 1.58215e+08] (0.95 confidence)</td>
<td>0.996666</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k1&#95;roundtrip&#95;adversarial</td>
<td>1.61807e+08 ns [1.60347e+08, 1.63258e+08] (0.95 confidence)</td>
<td>1.61481e+08 ns [1.61198e+08, 1.61998e+08] (0.95 confidence)</td>
<td>1.00202</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k2&#95;roundtrip</td>
<td>327560 ns [326072, 330038] (0.95 confidence)</td>
<td>331775 ns [328680, 336177] (0.95 confidence)</td>
<td>0.987295</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k2&#95;roundtrip&#95;adversarial</td>
<td>351579 ns [350446, 353137] (0.95 confidence)</td>
<td>377827 ns [369171, 383335] (0.95 confidence)</td>
<td>0.930529</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k3&#95;roundtrip</td>
<td>3.2704e+08 ns [3.25964e+08, 3.29062e+08] (0.95 confidence)</td>
<td>3.44528e+08 ns [3.43152e+08, 3.45703e+08] (0.95 confidence)</td>
<td>0.949242</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k3&#95;roundtrip&#95;adversarial</td>
<td>9.02243e+08 ns [8.97902e+08, 9.09675e+08] (0.95 confidence)</td>
<td>8.95369e+08 ns [8.92936e+08, 8.97203e+08] (0.95 confidence)</td>
<td>1.00768</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;2d/500</td>
<td>208375 ns [208286, 208760] (0.95 confidence)</td>
<td>209238 ns [208703, 209779] (0.95 confidence)</td>
<td>0.995875</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;2d&#95;adversarial/500</td>
<td>209078 ns [208700, 209568] (0.95 confidence)</td>
<td>209043 ns [208776, 209491] (0.95 confidence)</td>
<td>1.00016</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;3d/100</td>
<td>166993 ns [166395, 167538] (0.95 confidence)</td>
<td>168168 ns [167820, 168294] (0.95 confidence)</td>
<td>0.993013</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;3d&#95;adversarial/100</td>
<td>168796 ns [167924, 169619] (0.95 confidence)</td>
<td>170664 ns [170115, 171132] (0.95 confidence)</td>
<td>0.989055</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;4d/30</td>
<td>133378 ns [133255, 133801] (0.95 confidence)</td>
<td>133519 ns [133278, 133806] (0.95 confidence)</td>
<td>0.998946</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;4d&#95;adversarial/30</td>
<td>132137 ns [131752, 132262] (0.95 confidence)</td>
<td>132588 ns [132390, 133042] (0.95 confidence)</td>
<td>0.996599</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;5d/12</td>
<td>37245.2 ns [37207.6, 37381] (0.95 confidence)</td>
<td>37249.3 ns [37096, 37346.1] (0.95 confidence)</td>
<td>0.99989</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;5d&#95;adversarial/12</td>
<td>36674.6 ns [36587.8, 36738] (0.95 confidence)</td>
<td>36554.4 ns [36490.9, 36609.4] (0.95 confidence)</td>
<td>1.00329</td>
</tr>
<tr>
<td>circumcenter/solve&#95;path/near&#95;singular&#95;exact&#95;fallback&#95;3d</td>
<td>17648.7 ns [17630.9, 17673.6] (0.95 confidence)</td>
<td>17246 ns [17181.6, 17282.5] (0.95 confidence)</td>
<td>1.02335</td>
</tr>
<tr>
<td>circumcenter/solve&#95;path/regular&#95;lu&#95;3d</td>
<td>23.5747 ns [23.5596, 23.6028] (0.95 confidence)</td>
<td>23.3849 ns [23.3064, 23.4099] (0.95 confidence)</td>
<td>1.00811</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;2d/500</td>
<td>392740 ns [392094, 393491] (0.95 confidence)</td>
<td>394583 ns [394051, 395239] (0.95 confidence)</td>
<td>0.99533</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;2d&#95;adversarial/500</td>
<td>388022 ns [387771, 388453] (0.95 confidence)</td>
<td>387942 ns [387362, 389115] (0.95 confidence)</td>
<td>1.00021</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;3d/100</td>
<td>277313 ns [276855, 277745] (0.95 confidence)</td>
<td>278225 ns [278131, 278369] (0.95 confidence)</td>
<td>0.996724</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;3d&#95;adversarial/100</td>
<td>4.01122e+06 ns [4.00094e+06, 4.03434e+06] (0.95 confidence)</td>
<td>4.05937e+06 ns [4.04381e+06, 4.0738e+06] (0.95 confidence)</td>
<td>0.988138</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;4d/30</td>
<td>9.12379e+06 ns [9.10284e+06, 9.14415e+06] (0.95 confidence)</td>
<td>8.94607e+06 ns [8.92769e+06, 8.95601e+06] (0.95 confidence)</td>
<td>1.01987</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;4d&#95;adversarial/30</td>
<td>7.12308e+06 ns [7.10532e+06, 7.1307e+06] (0.95 confidence)</td>
<td>7.07226e+06 ns [7.06077e+06, 7.09273e+06] (0.95 confidence)</td>
<td>1.00719</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;5d/12</td>
<td>3.11797e+06 ns [3.11549e+06, 3.12584e+06] (0.95 confidence)</td>
<td>3.09623e+06 ns [3.07825e+06, 3.13184e+06] (0.95 confidence)</td>
<td>1.00702</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;5d&#95;adversarial/12</td>
<td>2.71512e+06 ns [2.70956e+06, 2.71784e+06] (0.95 confidence)</td>
<td>2.69742e+06 ns [2.6928e+06, 2.70794e+06] (0.95 confidence)</td>
<td>1.00656</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;nearest&#95;visible&#95;facet&#95;3d/100</td>
<td>1171.2 ns [1169.42, 1189.07] (0.95 confidence)</td>
<td>1165.87 ns [1154.46, 1170.25] (0.95 confidence)</td>
<td>1.00457</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;nearest&#95;visible&#95;facet&#95;3d&#95;adversarial/100</td>
<td>41388.8 ns [41261.4, 41615.1] (0.95 confidence)</td>
<td>42478.3 ns [42164, 42759.3] (0.95 confidence)</td>
<td>0.974352</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;visible&#95;facets&#95;3d/100</td>
<td>1258.47 ns [1234.68, 1299.59] (0.95 confidence)</td>
<td>1229.1 ns [1226.76, 1231.53] (0.95 confidence)</td>
<td>1.0239</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;visible&#95;facets&#95;3d&#95;adversarial/100</td>
<td>41388.3 ns [41312.1, 41559.3] (0.95 confidence)</td>
<td>41481.4 ns [41214.5, 41947.3] (0.95 confidence)</td>
<td>0.997755</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/is&#95;point&#95;outside&#95;3d/100</td>
<td>96.4058 ns [96.215, 96.6017] (0.95 confidence)</td>
<td>97.4004 ns [97.2356, 97.4811] (0.95 confidence)</td>
<td>0.989788</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/is&#95;point&#95;outside&#95;3d&#95;adversarial/100</td>
<td>777.584 ns [770.718, 784.95] (0.95 confidence)</td>
<td>772.905 ns [764.699, 781.356] (0.95 confidence)</td>
<td>1.00605</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/boundary&#95;point&#95;distance</td>
<td>13.7574 ns [13.6236, 13.7705] (0.95 confidence)</td>
<td>13.8047 ns [13.7848, 13.8159] (0.95 confidence)</td>
<td>0.996579</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/boundary&#95;point&#95;insphere</td>
<td>0.903529 ns [0.900646, 0.906961] (0.95 confidence)</td>
<td>0.831578 ns [0.828869, 0.835866] (0.95 confidence)</td>
<td>1.08652</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/boundary&#95;point&#95;lifted</td>
<td>976.899 ns [974.848, 978.101] (0.95 confidence)</td>
<td>1001.84 ns [1000.03, 1003.38] (0.95 confidence)</td>
<td>0.975106</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/far&#95;point&#95;distance</td>
<td>13.8418 ns [13.7792, 13.8656] (0.95 confidence)</td>
<td>13.7827 ns [13.7668, 13.8051] (0.95 confidence)</td>
<td>1.00429</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/far&#95;point&#95;insphere</td>
<td>50.9411 ns [50.6328, 51.2917] (0.95 confidence)</td>
<td>51.4389 ns [51.0609, 51.8342] (0.95 confidence)</td>
<td>0.990322</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/far&#95;point&#95;lifted</td>
<td>49.812 ns [49.5176, 50.0288] (0.95 confidence)</td>
<td>50.2676 ns [50.0215, 50.7169] (0.95 confidence)</td>
<td>0.990936</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/near&#95;boundary&#95;distance</td>
<td>13.9701 ns [13.9128, 14.0195] (0.95 confidence)</td>
<td>14.0571 ns [14.0344, 14.0803] (0.95 confidence)</td>
<td>0.993814</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/near&#95;boundary&#95;insphere</td>
<td>48.3694 ns [47.9694, 48.7102] (0.95 confidence)</td>
<td>49.2676 ns [48.8169, 49.4739] (0.95 confidence)</td>
<td>0.981769</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/near&#95;boundary&#95;lifted</td>
<td>47.4481 ns [47.1498, 47.8577] (0.95 confidence)</td>
<td>47.797 ns [47.5053, 47.9346] (0.95 confidence)</td>
<td>0.9927</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/boundary&#95;point&#95;distance</td>
<td>33.402 ns [33.2658, 33.4563] (0.95 confidence)</td>
<td>32.9789 ns [32.9358, 33.0385] (0.95 confidence)</td>
<td>1.01283</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/boundary&#95;point&#95;insphere</td>
<td>1.19964 ns [1.19881, 1.20043] (0.95 confidence)</td>
<td>1.18811 ns [1.18116, 1.19191] (0.95 confidence)</td>
<td>1.0097</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/boundary&#95;point&#95;lifted</td>
<td>1786.73 ns [1784.73, 1792.58] (0.95 confidence)</td>
<td>1851.14 ns [1848.7, 1853.97] (0.95 confidence)</td>
<td>0.965203</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/far&#95;point&#95;distance</td>
<td>33.3809 ns [33.2715, 33.459] (0.95 confidence)</td>
<td>33.2935 ns [33.2409, 33.3767] (0.95 confidence)</td>
<td>1.00262</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/far&#95;point&#95;insphere</td>
<td>107.919 ns [107.611, 108.111] (0.95 confidence)</td>
<td>112.497 ns [112.027, 113.013] (0.95 confidence)</td>
<td>0.959306</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/far&#95;point&#95;lifted</td>
<td>108.401 ns [107.802, 109.385] (0.95 confidence)</td>
<td>111.406 ns [110.758, 112.591] (0.95 confidence)</td>
<td>0.97303</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/near&#95;boundary&#95;distance</td>
<td>33.0618 ns [33.0021, 33.1471] (0.95 confidence)</td>
<td>33.5305 ns [33.3869, 33.691] (0.95 confidence)</td>
<td>0.986023</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/near&#95;boundary&#95;insphere</td>
<td>97.252 ns [96.6742, 97.7014] (0.95 confidence)</td>
<td>99.8856 ns [99.2323, 100.566] (0.95 confidence)</td>
<td>0.973634</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/near&#95;boundary&#95;lifted</td>
<td>97.2287 ns [96.5233, 98.2006] (0.95 confidence)</td>
<td>100.681 ns [99.5869, 101.847] (0.95 confidence)</td>
<td>0.965713</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/boundary&#95;point&#95;distance</td>
<td>54.8222 ns [54.5864, 55.0554] (0.95 confidence)</td>
<td>55.873 ns [55.4486, 56.0655] (0.95 confidence)</td>
<td>0.981193</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/boundary&#95;point&#95;insphere</td>
<td>1.63383 ns [1.62154, 1.6479] (0.95 confidence)</td>
<td>1.62978 ns [1.56947, 1.66554] (0.95 confidence)</td>
<td>1.00249</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/boundary&#95;point&#95;lifted</td>
<td>3818.29 ns [3812.47, 3825.59] (0.95 confidence)</td>
<td>3826.85 ns [3820.84, 3832.37] (0.95 confidence)</td>
<td>0.997763</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/far&#95;point&#95;distance</td>
<td>54.3179 ns [54.2455, 54.9015] (0.95 confidence)</td>
<td>54.5453 ns [54.3474, 54.7307] (0.95 confidence)</td>
<td>0.995831</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/far&#95;point&#95;insphere</td>
<td>771.22 ns [767.047, 776.416] (0.95 confidence)</td>
<td>742.234 ns [739.758, 745.672] (0.95 confidence)</td>
<td>1.03905</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/far&#95;point&#95;lifted</td>
<td>732.226 ns [728.377, 737.402] (0.95 confidence)</td>
<td>749.714 ns [747.576, 752.337] (0.95 confidence)</td>
<td>0.976673</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/near&#95;boundary&#95;distance</td>
<td>55.6551 ns [55.533, 55.6776] (0.95 confidence)</td>
<td>54.7599 ns [54.7132, 54.85] (0.95 confidence)</td>
<td>1.01635</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/near&#95;boundary&#95;insphere</td>
<td>727.915 ns [723.762, 731.321] (0.95 confidence)</td>
<td>734.888 ns [731.23, 739.601] (0.95 confidence)</td>
<td>0.990512</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/near&#95;boundary&#95;lifted</td>
<td>713.821 ns [712.47, 715.403] (0.95 confidence)</td>
<td>726.513 ns [723.878, 731.1] (0.95 confidence)</td>
<td>0.98253</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/boundary&#95;point&#95;distance</td>
<td>88.3446 ns [88.1994, 88.4314] (0.95 confidence)</td>
<td>88.0229 ns [87.6217, 88.4613] (0.95 confidence)</td>
<td>1.00365</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/boundary&#95;point&#95;insphere</td>
<td>1.71326 ns [1.70851, 1.72036] (0.95 confidence)</td>
<td>1.70331 ns [1.70119, 1.70499] (0.95 confidence)</td>
<td>1.00584</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/boundary&#95;point&#95;lifted</td>
<td>5795.28 ns [5792.11, 5803.53] (0.95 confidence)</td>
<td>5911.41 ns [5900.58, 5925.15] (0.95 confidence)</td>
<td>0.980356</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/far&#95;point&#95;distance</td>
<td>88.1192 ns [87.9762, 88.3127] (0.95 confidence)</td>
<td>88.1399 ns [87.551, 88.3877] (0.95 confidence)</td>
<td>0.999765</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/far&#95;point&#95;insphere</td>
<td>1282.54 ns [1278.17, 1293.32] (0.95 confidence)</td>
<td>1310.14 ns [1300.78, 1315.94] (0.95 confidence)</td>
<td>0.978938</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/far&#95;point&#95;lifted</td>
<td>1291.5 ns [1287.78, 1295.72] (0.95 confidence)</td>
<td>1286.58 ns [1282.84, 1293.59] (0.95 confidence)</td>
<td>1.00382</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/near&#95;boundary&#95;distance</td>
<td>88.2111 ns [87.8487, 88.8037] (0.95 confidence)</td>
<td>87.8915 ns [87.5211, 88.4635] (0.95 confidence)</td>
<td>1.00364</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/near&#95;boundary&#95;insphere</td>
<td>1256.93 ns [1250.47, 1264.72] (0.95 confidence)</td>
<td>1279.48 ns [1275.02, 1283.27] (0.95 confidence)</td>
<td>0.982376</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/near&#95;boundary&#95;lifted</td>
<td>1282.73 ns [1278.28, 1288.24] (0.95 confidence)</td>
<td>1303.7 ns [1296.24, 1314.95] (0.95 confidence)</td>
<td>0.983919</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;2d/vertices&#95;120&#95;simplices&#95;226</td>
<td>1.61426e+06 ns [1.61079e+06, 1.61806e+06] (0.95 confidence)</td>
<td>1.61927e+06 ns [1.61621e+06, 1.62742e+06] (0.95 confidence)</td>
<td>0.996903</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;3d/vertices&#95;30&#95;simplices&#95;115</td>
<td>1.58114e+08 ns [1.5804e+08, 1.58823e+08] (0.95 confidence)</td>
<td>1.6513e+08 ns [1.648e+08, 1.65724e+08] (0.95 confidence)</td>
<td>0.957513</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;4d/vertices&#95;16&#95;simplices&#95;68</td>
<td>4.65812e+08 ns [4.64691e+08, 4.67222e+08] (0.95 confidence)</td>
<td>4.92091e+08 ns [4.91011e+08, 4.93771e+08] (0.95 confidence)</td>
<td>0.946596</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;5d/vertices&#95;10&#95;simplices&#95;20</td>
<td>1.35032e+08 ns [1.3493e+08, 1.35308e+08] (0.95 confidence)</td>
<td>1.44527e+08 ns [1.43402e+08, 1.45e+08] (0.95 confidence)</td>
<td>0.934303</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;2d/10</td>
<td>453777 ns [452393, 454328] (0.95 confidence)</td>
<td>452588 ns [451371, 453762] (0.95 confidence)</td>
<td>1.00263</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;2d&#95;adversarial/10</td>
<td>2.89747e+06 ns [2.89364e+06, 2.89856e+06] (0.95 confidence)</td>
<td>2.91364e+06 ns [2.9112e+06, 2.91847e+06] (0.95 confidence)</td>
<td>0.994451</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;3d/10</td>
<td>2.80501e+06 ns [2.80115e+06, 2.81437e+06] (0.95 confidence)</td>
<td>2.79725e+06 ns [2.78209e+06, 2.8037e+06] (0.95 confidence)</td>
<td>1.00277</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;3d&#95;adversarial/10</td>
<td>1.36454e+07 ns [1.36234e+07, 1.37234e+07] (0.95 confidence)</td>
<td>1.35863e+07 ns [1.35669e+07, 1.3667e+07] (0.95 confidence)</td>
<td>1.00435</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;4d/6</td>
<td>2.68582e+07 ns [2.6765e+07, 2.73152e+07] (0.95 confidence)</td>
<td>2.68534e+07 ns [2.67314e+07, 2.69914e+07] (0.95 confidence)</td>
<td>1.00018</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;4d&#95;adversarial/6</td>
<td>6.7343e+07 ns [6.69243e+07, 6.74565e+07] (0.95 confidence)</td>
<td>6.76782e+07 ns [6.72186e+07, 6.8822e+07] (0.95 confidence)</td>
<td>0.995047</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;5d/4</td>
<td>5.15018e+07 ns [5.14409e+07, 5.16007e+07] (0.95 confidence)</td>
<td>5.16354e+07 ns [5.12428e+07, 5.18852e+07] (0.95 confidence)</td>
<td>0.997412</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;5d&#95;adversarial/4</td>
<td>5.56174e+07 ns [5.54218e+07, 5.56986e+07] (0.95 confidence)</td>
<td>5.58755e+07 ns [5.5585e+07, 5.60925e+07] (0.95 confidence)</td>
<td>0.99538</td>
</tr>
<tr>
<td>locate/exact&#95;hint/2d/vertices&#95;100&#95;simplices&#95;187&#95;seed&#95;dd618c77b9187836&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>7096.09 ns [7083.12, 7100.6] (0.95 confidence)</td>
<td>7134.59 ns [7096.25, 7144.38] (0.95 confidence)</td>
<td>0.994604</td>
</tr>
<tr>
<td>locate/exact&#95;hint/2d/vertices&#95;2000&#95;simplices&#95;3978&#95;seed&#95;01acf93075e96412&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>7301.61 ns [7296.61, 7312.02] (0.95 confidence)</td>
<td>7346.16 ns [7331.12, 7359.67] (0.95 confidence)</td>
<td>0.993936</td>
</tr>
<tr>
<td>locate/exact&#95;hint/2d/vertices&#95;500&#95;simplices&#95;983&#95;seed&#95;1493be4e9d7a5906&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>7465.36 ns [7439.38, 7501.26] (0.95 confidence)</td>
<td>7447.03 ns [7421.92, 7469.78] (0.95 confidence)</td>
<td>1.00246</td>
</tr>
<tr>
<td>locate/exact&#95;hint/3d/vertices&#95;150&#95;simplices&#95;829&#95;seed&#95;a44b52b395a4b44d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>14961.9 ns [14922.6, 15000.1] (0.95 confidence)</td>
<td>14845.3 ns [14787.2, 14966] (0.95 confidence)</td>
<td>1.00786</td>
</tr>
<tr>
<td>locate/exact&#95;hint/3d/vertices&#95;500&#95;simplices&#95;3052&#95;seed&#95;1493be4f9d7a5907&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>14968.5 ns [14871.6, 14987.1] (0.95 confidence)</td>
<td>14711.4 ns [14636.8, 14766.8] (0.95 confidence)</td>
<td>1.01748</td>
</tr>
<tr>
<td>locate/exact&#95;hint/3d/vertices&#95;50&#95;simplices&#95;222&#95;seed&#95;f61fc639dc8c3c19&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>14907.5 ns [14856.8, 14950.9] (0.95 confidence)</td>
<td>14767.9 ns [14750, 14938.3] (0.95 confidence)</td>
<td>1.00945</td>
</tr>
<tr>
<td>locate/exact&#95;hint/4d/vertices&#95;100&#95;simplices&#95;1677&#95;seed&#95;dd618c71b9187830&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>2.00233e+06 ns [1.99627e+06, 2.00887e+06] (0.95 confidence)</td>
<td>2.01094e+06 ns [2.00674e+06, 2.01434e+06] (0.95 confidence)</td>
<td>0.995719</td>
</tr>
<tr>
<td>locate/exact&#95;hint/4d/vertices&#95;50&#95;simplices&#95;626&#95;seed&#95;f61fc63edc8c3c1e&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>1.96324e+06 ns [1.95631e+06, 1.97227e+06] (0.95 confidence)</td>
<td>1.98844e+06 ns [1.98143e+06, 1.99943e+06] (0.95 confidence)</td>
<td>0.987327</td>
</tr>
<tr>
<td>locate/exact&#95;hint/5d/vertices&#95;25&#95;simplices&#95;410&#95;seed&#95;63a0e3186e461e08&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>4.74551e+06 ns [4.68576e+06, 4.80328e+06] (0.95 confidence)</td>
<td>4.82142e+06 ns [4.79739e+06, 4.83305e+06] (0.95 confidence)</td>
<td>0.984256</td>
</tr>
<tr>
<td>locate/exact&#95;hint/5d/vertices&#95;40&#95;simplices&#95;1208&#95;seed&#95;a86104fee3a3634d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>5.08725e+06 ns [5.03207e+06, 5.13213e+06] (0.95 confidence)</td>
<td>4.88357e+06 ns [4.85228e+06, 4.90178e+06] (0.95 confidence)</td>
<td>1.04171</td>
</tr>
<tr>
<td>locate/no&#95;hint/2d/vertices&#95;100&#95;simplices&#95;187&#95;seed&#95;dd618c77b9187836&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>74045.4 ns [73673.9, 74169.9] (0.95 confidence)</td>
<td>73682.8 ns [73483, 74176.6] (0.95 confidence)</td>
<td>1.00492</td>
</tr>
<tr>
<td>locate/no&#95;hint/2d/vertices&#95;2000&#95;simplices&#95;3978&#95;seed&#95;01acf93075e96412&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>266782 ns [264381, 268992] (0.95 confidence)</td>
<td>268736 ns [268173, 269288] (0.95 confidence)</td>
<td>0.992728</td>
</tr>
<tr>
<td>locate/no&#95;hint/2d/vertices&#95;500&#95;simplices&#95;983&#95;seed&#95;1493be4e9d7a5906&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>123012 ns [122715, 123419] (0.95 confidence)</td>
<td>122971 ns [122630, 123321] (0.95 confidence)</td>
<td>1.00033</td>
</tr>
<tr>
<td>locate/no&#95;hint/3d/vertices&#95;150&#95;simplices&#95;829&#95;seed&#95;a44b52b395a4b44d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>124983 ns [124645, 125379] (0.95 confidence)</td>
<td>124094 ns [123737, 124516] (0.95 confidence)</td>
<td>1.00717</td>
</tr>
<tr>
<td>locate/no&#95;hint/3d/vertices&#95;500&#95;simplices&#95;3052&#95;seed&#95;1493be4f9d7a5907&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>194735 ns [191666, 197266] (0.95 confidence)</td>
<td>191612 ns [190293, 196939] (0.95 confidence)</td>
<td>1.0163</td>
</tr>
<tr>
<td>locate/no&#95;hint/3d/vertices&#95;50&#95;simplices&#95;222&#95;seed&#95;f61fc639dc8c3c19&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>65921.7 ns [65808.2, 67047.3] (0.95 confidence)</td>
<td>65802.7 ns [65323.2, 66650.7] (0.95 confidence)</td>
<td>1.00181</td>
</tr>
<tr>
<td>locate/no&#95;hint/4d/vertices&#95;100&#95;simplices&#95;1677&#95;seed&#95;dd618c71b9187830&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>1.12975e+07 ns [1.12645e+07, 1.13483e+07] (0.95 confidence)</td>
<td>1.14644e+07 ns [1.14321e+07, 1.14827e+07] (0.95 confidence)</td>
<td>0.98544</td>
</tr>
<tr>
<td>locate/no&#95;hint/4d/vertices&#95;50&#95;simplices&#95;626&#95;seed&#95;f61fc63edc8c3c1e&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>1.15728e+07 ns [1.12974e+07, 1.19304e+07] (0.95 confidence)</td>
<td>1.12928e+07 ns [1.12727e+07, 1.14766e+07] (0.95 confidence)</td>
<td>1.02479</td>
</tr>
<tr>
<td>locate/no&#95;hint/5d/vertices&#95;25&#95;simplices&#95;410&#95;seed&#95;63a0e3186e461e08&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>2.22483e+07 ns [2.20968e+07, 2.24368e+07] (0.95 confidence)</td>
<td>2.29997e+07 ns [2.26654e+07, 2.36611e+07] (0.95 confidence)</td>
<td>0.967329</td>
</tr>
<tr>
<td>locate/no&#95;hint/5d/vertices&#95;40&#95;simplices&#95;1208&#95;seed&#95;a86104fee3a3634d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>3.97017e+07 ns [3.96066e+07, 4.01428e+07] (0.95 confidence)</td>
<td>4.08647e+07 ns [4.05985e+07, 4.0994e+07] (0.95 confidence)</td>
<td>0.971541</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;2d/simplices&#95;226</td>
<td>119843 ns [118135, 122249] (0.95 confidence)</td>
<td>122863 ns [120269, 126602] (0.95 confidence)</td>
<td>0.975414</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;3d/simplices&#95;115</td>
<td>147697 ns [146372, 149740] (0.95 confidence)</td>
<td>149119 ns [148337, 150292] (0.95 confidence)</td>
<td>0.99046</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;4d/simplices&#95;68</td>
<td>1.22934e+06 ns [1.22e+06, 1.27017e+06] (0.95 confidence)</td>
<td>1.25972e+06 ns [1.24525e+06, 1.28232e+06] (0.95 confidence)</td>
<td>0.975886</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;5d/simplices&#95;20</td>
<td>464349 ns [461254, 467487] (0.95 confidence)</td>
<td>482061 ns [473983, 486067] (0.95 confidence)</td>
<td>0.963257</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;2d/simplices&#95;226</td>
<td>1.35834e+06 ns [1.35397e+06, 1.37075e+06] (0.95 confidence)</td>
<td>1.35405e+06 ns [1.34696e+06, 1.35837e+06] (0.95 confidence)</td>
<td>1.00317</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;3d/simplices&#95;115</td>
<td>1.57629e+08 ns [1.57396e+08, 1.57704e+08] (0.95 confidence)</td>
<td>1.64328e+08 ns [1.64046e+08, 1.64553e+08] (0.95 confidence)</td>
<td>0.959239</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;4d/simplices&#95;68</td>
<td>4.64323e+08 ns [4.63572e+08, 4.67797e+08] (0.95 confidence)</td>
<td>4.89324e+08 ns [4.88972e+08, 4.89683e+08] (0.95 confidence)</td>
<td>0.948908</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;5d/simplices&#95;20</td>
<td>1.34802e+08 ns [1.34689e+08, 1.34911e+08] (0.95 confidence)</td>
<td>1.4188e+08 ns [1.41461e+08, 1.42464e+08] (0.95 confidence)</td>
<td>0.950113</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;2d/simplices&#95;226</td>
<td>1.31202e+06 ns [1.30609e+06, 1.32492e+06] (0.95 confidence)</td>
<td>1.2864e+06 ns [1.27865e+06, 1.37452e+06] (0.95 confidence)</td>
<td>1.01992</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;3d/simplices&#95;115</td>
<td>1.57821e+08 ns [1.57287e+08, 1.60001e+08] (0.95 confidence)</td>
<td>1.64213e+08 ns [1.64017e+08, 1.64466e+08] (0.95 confidence)</td>
<td>0.961076</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;4d/simplices&#95;68</td>
<td>4.63746e+08 ns [4.62955e+08, 4.64703e+08] (0.95 confidence)</td>
<td>4.88572e+08 ns [4.87933e+08, 4.89346e+08] (0.95 confidence)</td>
<td>0.949187</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;5d/simplices&#95;20</td>
<td>1.34871e+08 ns [1.34036e+08, 1.35499e+08] (0.95 confidence)</td>
<td>1.41344e+08 ns [1.41195e+08, 1.41812e+08] (0.95 confidence)</td>
<td>0.954208</td>
</tr>
<tr>
<td>random/insphere&#95;1000&#95;queries</td>
<td>439098 ns [437383, 440595] (0.95 confidence)</td>
<td>434439 ns [432056, 436353] (0.95 confidence)</td>
<td>1.01072</td>
</tr>
<tr>
<td>random/insphere&#95;distance&#95;1000&#95;queries</td>
<td>66446.1 ns [66216.9, 66579.9] (0.95 confidence)</td>
<td>67483.9 ns [67048.3, 67790.4] (0.95 confidence)</td>
<td>0.98462</td>
</tr>
<tr>
<td>random/insphere&#95;lifted&#95;1000&#95;queries</td>
<td>439122 ns [436337, 442144] (0.95 confidence)</td>
<td>431475 ns [427940, 433122] (0.95 confidence)</td>
<td>1.01772</td>
</tr>
<tr>
<td>tds&#95;new&#95;2d/tds&#95;new/500</td>
<td>4.70807e+07 ns [4.70218e+07, 4.73022e+07] (0.95 confidence)</td>
<td>4.58772e+07 ns [4.57603e+07, 4.65144e+07] (0.95 confidence)</td>
<td>1.02623</td>
</tr>
<tr>
<td>tds&#95;new&#95;2d/tds&#95;new&#95;adversarial/500</td>
<td>5.52101e+07 ns [5.50795e+07, 5.53742e+07] (0.95 confidence)</td>
<td>5.52893e+07 ns [5.51476e+07, 5.5385e+07] (0.95 confidence)</td>
<td>0.998568</td>
</tr>
<tr>
<td>tds&#95;new&#95;3d/tds&#95;new/100</td>
<td>3.84446e+07 ns [3.79665e+07, 3.88517e+07] (0.95 confidence)</td>
<td>3.82393e+07 ns [3.81591e+07, 3.83073e+07] (0.95 confidence)</td>
<td>1.00537</td>
</tr>
<tr>
<td>tds&#95;new&#95;3d/tds&#95;new&#95;adversarial/100</td>
<td>8.70235e+07 ns [8.66073e+07, 8.84273e+07] (0.95 confidence)</td>
<td>8.66396e+07 ns [8.64099e+07, 8.70051e+07] (0.95 confidence)</td>
<td>1.00443</td>
</tr>
<tr>
<td>tds&#95;new&#95;4d/tds&#95;new/30</td>
<td>1.01916e+08 ns [1.01262e+08, 1.03409e+08] (0.95 confidence)</td>
<td>9.91059e+07 ns [9.82692e+07, 1.00005e+08] (0.95 confidence)</td>
<td>1.02835</td>
</tr>
<tr>
<td>tds&#95;new&#95;4d/tds&#95;new&#95;adversarial/30</td>
<td>9.43546e+07 ns [9.30957e+07, 9.55096e+07] (0.95 confidence)</td>
<td>9.38753e+07 ns [9.31216e+07, 9.46734e+07] (0.95 confidence)</td>
<td>1.00511</td>
</tr>
<tr>
<td>tds&#95;new&#95;5d/tds&#95;new/12</td>
<td>2.19239e+07 ns [2.18716e+07, 2.21519e+07] (0.95 confidence)</td>
<td>2.18773e+07 ns [2.17847e+07, 2.20563e+07] (0.95 confidence)</td>
<td>1.00213</td>
</tr>
<tr>
<td>tds&#95;new&#95;5d/tds&#95;new&#95;adversarial/12</td>
<td>2.03666e+07 ns [2.03301e+07, 2.04422e+07] (0.95 confidence)</td>
<td>2.01132e+07 ns [2.00461e+07, 2.03166e+07] (0.95 confidence)</td>
<td>1.0126</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;2d/500</td>
<td>796331 ns [792007, 804812] (0.95 confidence)</td>
<td>819009 ns [810620, 836398] (0.95 confidence)</td>
<td>0.97231</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;2d&#95;adversarial/500</td>
<td>1.0992e+06 ns [1.09741e+06, 1.11211e+06] (0.95 confidence)</td>
<td>1.10238e+06 ns [1.08831e+06, 1.11474e+06] (0.95 confidence)</td>
<td>0.997116</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;3d/100</td>
<td>1.25541e+06 ns [1.25056e+06, 1.26039e+06] (0.95 confidence)</td>
<td>1.27675e+06 ns [1.26467e+06, 1.29446e+06] (0.95 confidence)</td>
<td>0.983287</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;3d&#95;adversarial/100</td>
<td>2.25883e+06 ns [2.25101e+06, 2.2864e+06] (0.95 confidence)</td>
<td>2.24011e+06 ns [2.2251e+06, 2.266e+06] (0.95 confidence)</td>
<td>1.00836</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;4d/30</td>
<td>4.7649e+06 ns [4.66975e+06, 4.8019e+06] (0.95 confidence)</td>
<td>4.85108e+06 ns [4.82204e+06, 4.90244e+06] (0.95 confidence)</td>
<td>0.982235</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;4d&#95;adversarial/30</td>
<td>4.69599e+06 ns [4.59165e+06, 4.75492e+06] (0.95 confidence)</td>
<td>4.65774e+06 ns [4.59226e+06, 4.75728e+06] (0.95 confidence)</td>
<td>1.00821</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;5d/12</td>
<td>2.13796e+06 ns [2.13331e+06, 2.14474e+06] (0.95 confidence)</td>
<td>2.17768e+06 ns [2.16013e+06, 2.19111e+06] (0.95 confidence)</td>
<td>0.981761</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;5d&#95;adversarial/12</td>
<td>1.91391e+06 ns [1.90989e+06, 1.92073e+06] (0.95 confidence)</td>
<td>1.96031e+06 ns [1.94437e+06, 1.98187e+06] (0.95 confidence)</td>
<td>0.976329</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;2d/500</td>
<td>800057 ns [791882, 805872] (0.95 confidence)</td>
<td>813088 ns [799063, 827478] (0.95 confidence)</td>
<td>0.983973</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;2d&#95;adversarial/500</td>
<td>1.09973e+06 ns [1.09454e+06, 1.11004e+06] (0.95 confidence)</td>
<td>1.09659e+06 ns [1.08904e+06, 1.11443e+06] (0.95 confidence)</td>
<td>1.00286</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;3d/100</td>
<td>1.25479e+06 ns [1.25108e+06, 1.26573e+06] (0.95 confidence)</td>
<td>1.26712e+06 ns [1.25207e+06, 1.28091e+06] (0.95 confidence)</td>
<td>0.990274</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;3d&#95;adversarial/100</td>
<td>2.26676e+06 ns [2.26351e+06, 2.27067e+06] (0.95 confidence)</td>
<td>2.25e+06 ns [2.22212e+06, 2.27206e+06] (0.95 confidence)</td>
<td>1.00745</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;4d/30</td>
<td>4.67581e+06 ns [4.65415e+06, 4.70216e+06] (0.95 confidence)</td>
<td>4.79195e+06 ns [4.72742e+06, 4.84561e+06] (0.95 confidence)</td>
<td>0.975762</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;4d&#95;adversarial/30</td>
<td>4.5925e+06 ns [4.56835e+06, 4.64542e+06] (0.95 confidence)</td>
<td>4.64816e+06 ns [4.60721e+06, 4.7236e+06] (0.95 confidence)</td>
<td>0.988024</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;5d/12</td>
<td>2.13428e+06 ns [2.12683e+06, 2.15213e+06] (0.95 confidence)</td>
<td>2.18567e+06 ns [2.16114e+06, 2.19992e+06] (0.95 confidence)</td>
<td>0.97649</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;5d&#95;adversarial/12</td>
<td>1.92809e+06 ns [1.9151e+06, 1.93614e+06] (0.95 confidence)</td>
<td>1.96376e+06 ns [1.94502e+06, 1.98822e+06] (0.95 confidence)</td>
<td>0.981836</td>
</tr>
<tr>
<td>validation/validate&#95;2d/500</td>
<td>8.34733e+06 ns [8.32607e+06, 8.40825e+06] (0.95 confidence)</td>
<td>8.20916e+06 ns [8.17937e+06, 8.26202e+06] (0.95 confidence)</td>
<td>1.01683</td>
</tr>
<tr>
<td>validation/validate&#95;2d&#95;adversarial/500</td>
<td>4.96223e+07 ns [4.95831e+07, 4.97922e+07] (0.95 confidence)</td>
<td>4.94151e+07 ns [4.92745e+07, 5.01795e+07] (0.95 confidence)</td>
<td>1.00419</td>
</tr>
<tr>
<td>validation/validate&#95;3d/100</td>
<td>8.54715e+08 ns [8.5217e+08, 8.55996e+08] (0.95 confidence)</td>
<td>8.86407e+08 ns [8.83659e+08, 8.93589e+08] (0.95 confidence)</td>
<td>0.964246</td>
</tr>
<tr>
<td>validation/validate&#95;3d&#95;adversarial/100</td>
<td>1.04536e+09 ns [1.04446e+09, 1.04749e+09] (0.95 confidence)</td>
<td>1.08787e+09 ns [1.08386e+09, 1.11479e+09] (0.95 confidence)</td>
<td>0.960925</td>
</tr>
<tr>
<td>validation/validate&#95;4d/30</td>
<td>3.88182e+09 ns [3.86246e+09, 3.9029e+09] (0.95 confidence)</td>
<td>4.08714e+09 ns [4.08021e+09, 4.19479e+09] (0.95 confidence)</td>
<td>0.949764</td>
</tr>
<tr>
<td>validation/validate&#95;4d&#95;adversarial/30</td>
<td>3.94112e+09 ns [3.9373e+09, 3.9484e+09] (0.95 confidence)</td>
<td>4.16476e+09 ns [4.16136e+09, 4.18661e+09] (0.95 confidence)</td>
<td>0.946302</td>
</tr>
<tr>
<td>validation/validate&#95;5d/12</td>
<td>8.7586e+08 ns [8.74588e+08, 8.7639e+08] (0.95 confidence)</td>
<td>9.26429e+08 ns [9.23955e+08, 9.31433e+08] (0.95 confidence)</td>
<td>0.945415</td>
</tr>
<tr>
<td>validation/validate&#95;5d&#95;adversarial/12</td>
<td>8.0507e+08 ns [8.03818e+08, 8.08482e+08] (0.95 confidence)</td>
<td>8.47672e+08 ns [8.46138e+08, 8.5433e+08] (0.95 confidence)</td>
<td>0.949742</td>
</tr>
</tbody>
</table>

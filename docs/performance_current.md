<!-- research-repo-tools/run-report/v3 -->
# Delaunay public API performance

Run: `09e66eef3fc9e342931ff0f995ed3f2db4d8aed748c757b2f9c2533aa0dec4ad`

Evidence: [run payload](performance-runs/runs/09e66eef3fc9e342931ff0f995ed3f2db4d8aed748c757b2f9c2533aa0dec4ad/run.json) ·
[evidence envelope](performance-runs/runs/09e66eef3fc9e342931ff0f995ed3f2db4d8aed748c757b2f9c2533aa0dec4ad/evidence.json).

Intervals are marginal timing intervals. No significance or ratio interval is inferred.

- baseline: measured in **baseline** phase (v0.8.3), revision `3fa8a9907101c523574a0d17991a1f652b5419dc`.
- current: measured in **current** phase (v0.8.4), revision `317a114297007b1aa47082a73ed1c638a1e52fbf`.

## Mean (ns)

<table>
<thead>
<tr>
<th>Case</th>
<th>baseline</th>
<th>current</th>
</tr>
</thead>
<tbody>
<tr>
<td>2d/insphere</td>
<td>50.3629 ns [50.1007, 50.632] (0.95 confidence)</td>
<td>49.9714 ns [49.7588, 50.1938] (0.95 confidence)</td>
</tr>
<tr>
<td>2d/insphere&#95;distance</td>
<td>13.4344 ns [13.4123, 13.4565] (0.95 confidence)</td>
<td>13.7432 ns [13.7119, 13.7772] (0.95 confidence)</td>
</tr>
<tr>
<td>2d/insphere&#95;lifted</td>
<td>49.3667 ns [49.1805, 49.5559] (0.95 confidence)</td>
<td>52.2499 ns [51.8981, 52.6023] (0.95 confidence)</td>
</tr>
<tr>
<td>3d/insphere</td>
<td>107.807 ns [107.127, 108.55] (0.95 confidence)</td>
<td>111.598 ns [110.877, 112.327] (0.95 confidence)</td>
</tr>
<tr>
<td>3d/insphere&#95;distance</td>
<td>32.5374 ns [32.4341, 32.6407] (0.95 confidence)</td>
<td>33.2631 ns [33.1675, 33.3635] (0.95 confidence)</td>
</tr>
<tr>
<td>3d/insphere&#95;lifted</td>
<td>105.5 ns [105.122, 105.901] (0.95 confidence)</td>
<td>107.329 ns [106.914, 107.781] (0.95 confidence)</td>
</tr>
<tr>
<td>4d/insphere</td>
<td>749.057 ns [746.628, 751.571] (0.95 confidence)</td>
<td>759.7 ns [758.152, 761.301] (0.95 confidence)</td>
</tr>
<tr>
<td>4d/insphere&#95;distance</td>
<td>50.8807 ns [50.7535, 51.0078] (0.95 confidence)</td>
<td>54.4821 ns [54.4075, 54.5543] (0.95 confidence)</td>
</tr>
<tr>
<td>4d/insphere&#95;lifted</td>
<td>752.908 ns [749.769, 756.022] (0.95 confidence)</td>
<td>764.499 ns [760.881, 768.229] (0.95 confidence)</td>
</tr>
<tr>
<td>5d/insphere</td>
<td>1387.9 ns [1383.98, 1391.9] (0.95 confidence)</td>
<td>1287.3 ns [1284.02, 1290.71] (0.95 confidence)</td>
</tr>
<tr>
<td>5d/insphere&#95;distance</td>
<td>87.4293 ns [87.1704, 87.6922] (0.95 confidence)</td>
<td>87.0696 ns [86.8265, 87.3225] (0.95 confidence)</td>
</tr>
<tr>
<td>5d/insphere&#95;lifted</td>
<td>1374.05 ns [1370.15, 1377.96] (0.95 confidence)</td>
<td>1300.07 ns [1295.7, 1304.4] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k1&#95;roundtrip</td>
<td>53991 ns [53907.8, 54086.5] (0.95 confidence)</td>
<td>56295.4 ns [56144.4, 56459.1] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k1&#95;roundtrip&#95;adversarial</td>
<td>162550 ns [162378, 162722] (0.95 confidence)</td>
<td>166391 ns [166034, 166747] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k2&#95;edge&#95;flip</td>
<td>26258.3 ns [26231.1, 26287.7] (0.95 confidence)</td>
<td>27593.7 ns [27533.9, 27667.4] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k2&#95;edge&#95;flip&#95;adversarial</td>
<td>101699 ns [99923.2, 103885] (0.95 confidence)</td>
<td>102942 ns [102646, 103239] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k1&#95;roundtrip</td>
<td>2.3395e+06 ns [2.33445e+06, 2.34426e+06] (0.95 confidence)</td>
<td>2.41399e+06 ns [2.4096e+06, 2.41826e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k1&#95;roundtrip&#95;adversarial</td>
<td>1.95481e+06 ns [1.94607e+06, 1.9653e+06] (0.95 confidence)</td>
<td>2.00716e+06 ns [2.00541e+06, 2.00886e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k2&#95;roundtrip</td>
<td>6.58713e+06 ns [6.47698e+06, 6.71913e+06] (0.95 confidence)</td>
<td>6.63185e+06 ns [6.62046e+06, 6.64449e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k2&#95;roundtrip&#95;adversarial</td>
<td>5.82996e+06 ns [5.81511e+06, 5.84711e+06] (0.95 confidence)</td>
<td>6.02659e+06 ns [6.01725e+06, 6.03505e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k3&#95;forward</td>
<td>1.35985e+06 ns [1.35702e+06, 1.36271e+06] (0.95 confidence)</td>
<td>1.39557e+06 ns [1.39295e+06, 1.39875e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k3&#95;forward&#95;adversarial</td>
<td>1.88579e+06 ns [1.87877e+06, 1.89349e+06] (0.95 confidence)</td>
<td>1.87074e+06 ns [1.8671e+06, 1.8745e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k1&#95;roundtrip</td>
<td>3.91942e+07 ns [3.89438e+07, 3.94883e+07] (0.95 confidence)</td>
<td>3.9626e+07 ns [3.94912e+07, 3.98113e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k1&#95;roundtrip&#95;adversarial</td>
<td>3.31749e+07 ns [3.30477e+07, 3.33109e+07] (0.95 confidence)</td>
<td>3.33839e+07 ns [3.33368e+07, 3.34367e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k2&#95;roundtrip</td>
<td>5.61884e+07 ns [5.59148e+07, 5.64898e+07] (0.95 confidence)</td>
<td>5.63516e+07 ns [5.62749e+07, 5.64325e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k2&#95;roundtrip&#95;adversarial</td>
<td>2.00254e+08 ns [1.98791e+08, 2.01988e+08] (0.95 confidence)</td>
<td>1.98677e+08 ns [1.98407e+08, 1.98921e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k3&#95;roundtrip</td>
<td>6.55919e+07 ns [6.51718e+07, 6.6026e+07] (0.95 confidence)</td>
<td>6.50869e+07 ns [6.49496e+07, 6.52353e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k3&#95;roundtrip&#95;adversarial</td>
<td>3.52388e+07 ns [3.5102e+07, 3.53859e+07] (0.95 confidence)</td>
<td>3.51438e+07 ns [3.50984e+07, 3.51917e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k1&#95;roundtrip</td>
<td>1.55988e+08 ns [1.55554e+08, 1.56426e+08] (0.95 confidence)</td>
<td>1.59178e+08 ns [1.58857e+08, 1.5952e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k1&#95;roundtrip&#95;adversarial</td>
<td>1.60754e+08 ns [1.59915e+08, 1.61653e+08] (0.95 confidence)</td>
<td>1.62011e+08 ns [1.61557e+08, 1.62463e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k2&#95;roundtrip</td>
<td>329220 ns [328636, 329841] (0.95 confidence)</td>
<td>329108 ns [328883, 329324] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k2&#95;roundtrip&#95;adversarial</td>
<td>352172 ns [351379, 352996] (0.95 confidence)</td>
<td>356146 ns [355584, 356751] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k3&#95;roundtrip</td>
<td>3.26703e+08 ns [3.25532e+08, 3.2809e+08] (0.95 confidence)</td>
<td>3.30575e+08 ns [3.30234e+08, 3.30918e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k3&#95;roundtrip&#95;adversarial</td>
<td>9.10658e+08 ns [9.07812e+08, 9.13841e+08] (0.95 confidence)</td>
<td>8.82636e+08 ns [8.79657e+08, 8.86312e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;2d/500</td>
<td>209632 ns [209098, 210187] (0.95 confidence)</td>
<td>211250 ns [210527, 212025] (0.95 confidence)</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;2d&#95;adversarial/500</td>
<td>209507 ns [209292, 209744] (0.95 confidence)</td>
<td>210174 ns [209394, 210922] (0.95 confidence)</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;3d/100</td>
<td>166339 ns [166016, 166741] (0.95 confidence)</td>
<td>167056 ns [166588, 167540] (0.95 confidence)</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;3d&#95;adversarial/100</td>
<td>168132 ns [167863, 168417] (0.95 confidence)</td>
<td>168413 ns [168092, 168732] (0.95 confidence)</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;4d/30</td>
<td>131622 ns [131385, 131915] (0.95 confidence)</td>
<td>137706 ns [137303, 138107] (0.95 confidence)</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;4d&#95;adversarial/30</td>
<td>131050 ns [130824, 131290] (0.95 confidence)</td>
<td>134747 ns [134256, 135263] (0.95 confidence)</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;5d/12</td>
<td>36669.9 ns [36519.8, 36840.6] (0.95 confidence)</td>
<td>37385.2 ns [37266, 37504.3] (0.95 confidence)</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;5d&#95;adversarial/12</td>
<td>35788.4 ns [35725.1, 35858.7] (0.95 confidence)</td>
<td>36980.9 ns [36801.1, 37160.8] (0.95 confidence)</td>
</tr>
<tr>
<td>circumcenter/solve&#95;path/near&#95;singular&#95;exact&#95;fallback&#95;3d</td>
<td>17669.4 ns [17626.9, 17711.7] (0.95 confidence)</td>
<td>17606.8 ns [17584.7, 17628.7] (0.95 confidence)</td>
</tr>
<tr>
<td>circumcenter/solve&#95;path/regular&#95;lu&#95;3d</td>
<td>22.9912 ns [22.9638, 23.0195] (0.95 confidence)</td>
<td>23.7471 ns [23.7008, 23.7993] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;2d/500</td>
<td>343199 ns [341356, 345118] (0.95 confidence)</td>
<td>391508 ns [390646, 392422] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;2d&#95;adversarial/500</td>
<td>332771 ns [331997, 334037] (0.95 confidence)</td>
<td>389851 ns [388690, 391118] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;3d/100</td>
<td>275770 ns [275370, 276220] (0.95 confidence)</td>
<td>277801 ns [276837, 278751] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;3d&#95;adversarial/100</td>
<td>4.00082e+06 ns [3.9926e+06, 4.00978e+06] (0.95 confidence)</td>
<td>4.03635e+06 ns [4.02751e+06, 4.04514e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;4d/30</td>
<td>8.78704e+06 ns [8.73664e+06, 8.84747e+06] (0.95 confidence)</td>
<td>8.9468e+06 ns [8.92504e+06, 8.96889e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;4d&#95;adversarial/30</td>
<td>6.99418e+06 ns [6.95724e+06, 7.03573e+06] (0.95 confidence)</td>
<td>7.07419e+06 ns [7.04404e+06, 7.11162e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;5d/12</td>
<td>3.07103e+06 ns [3.06251e+06, 3.0796e+06] (0.95 confidence)</td>
<td>3.07207e+06 ns [3.05725e+06, 3.09035e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;5d&#95;adversarial/12</td>
<td>2.63914e+06 ns [2.63246e+06, 2.64712e+06] (0.95 confidence)</td>
<td>2.73111e+06 ns [2.71267e+06, 2.74983e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;nearest&#95;visible&#95;facet&#95;3d/100</td>
<td>1134.72 ns [1130.38, 1138.79] (0.95 confidence)</td>
<td>1149.57 ns [1144.79, 1154.44] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;nearest&#95;visible&#95;facet&#95;3d&#95;adversarial/100</td>
<td>40912.5 ns [40804.1, 41034.4] (0.95 confidence)</td>
<td>41004.1 ns [40895.6, 41133.8] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;visible&#95;facets&#95;3d/100</td>
<td>1212.91 ns [1201.63, 1227.13] (0.95 confidence)</td>
<td>1222.44 ns [1216.82, 1228.52] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;visible&#95;facets&#95;3d&#95;adversarial/100</td>
<td>41046.2 ns [40913.5, 41199.7] (0.95 confidence)</td>
<td>41131.9 ns [41039.7, 41228] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/is&#95;point&#95;outside&#95;3d/100</td>
<td>94.6067 ns [94.4914, 94.7352] (0.95 confidence)</td>
<td>95.7992 ns [95.6301, 95.9675] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/is&#95;point&#95;outside&#95;3d&#95;adversarial/100</td>
<td>770.856 ns [768.53, 773.297] (0.95 confidence)</td>
<td>780.13 ns [776.819, 783.986] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/boundary&#95;point&#95;distance</td>
<td>13.5388 ns [13.514, 13.5649] (0.95 confidence)</td>
<td>13.4266 ns [13.3943, 13.4638] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/boundary&#95;point&#95;insphere</td>
<td>0.83637 ns [0.833595, 0.8392] (0.95 confidence)</td>
<td>0.827141 ns [0.824016, 0.830381] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/boundary&#95;point&#95;lifted</td>
<td>971.001 ns [969.089, 972.955] (0.95 confidence)</td>
<td>978.53 ns [976.62, 980.488] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/far&#95;point&#95;distance</td>
<td>13.5132 ns [13.4817, 13.5458] (0.95 confidence)</td>
<td>13.6775 ns [13.6539, 13.7016] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/far&#95;point&#95;insphere</td>
<td>48.9921 ns [48.8046, 49.1883] (0.95 confidence)</td>
<td>50.7656 ns [50.4847, 51.0571] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/far&#95;point&#95;lifted</td>
<td>50.5973 ns [50.2123, 51.0038] (0.95 confidence)</td>
<td>50.5494 ns [50.2123, 50.8927] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/near&#95;boundary&#95;distance</td>
<td>13.8014 ns [13.757, 13.8495] (0.95 confidence)</td>
<td>13.8464 ns [13.8242, 13.868] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/near&#95;boundary&#95;insphere</td>
<td>47.0807 ns [46.8132, 47.361] (0.95 confidence)</td>
<td>48.0824 ns [47.7365, 48.4402] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/near&#95;boundary&#95;lifted</td>
<td>47.3911 ns [47.0183, 47.7893] (0.95 confidence)</td>
<td>47.5431 ns [47.261, 47.852] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/boundary&#95;point&#95;distance</td>
<td>32.7541 ns [32.6327, 32.8795] (0.95 confidence)</td>
<td>32.8153 ns [32.7445, 32.8948] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/boundary&#95;point&#95;insphere</td>
<td>1.21494 ns [1.20195, 1.23087] (0.95 confidence)</td>
<td>1.18188 ns [1.17849, 1.18493] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/boundary&#95;point&#95;lifted</td>
<td>1744.54 ns [1742.43, 1746.66] (0.95 confidence)</td>
<td>1776.98 ns [1775.41, 1778.57] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/far&#95;point&#95;distance</td>
<td>32.9828 ns [32.84, 33.1373] (0.95 confidence)</td>
<td>33.1066 ns [33.0361, 33.1795] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/far&#95;point&#95;insphere</td>
<td>108.102 ns [107.54, 108.695] (0.95 confidence)</td>
<td>108.558 ns [107.966, 109.172] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/far&#95;point&#95;lifted</td>
<td>111.197 ns [110.58, 111.827] (0.95 confidence)</td>
<td>112.344 ns [111.798, 112.911] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/near&#95;boundary&#95;distance</td>
<td>32.6832 ns [32.5787, 32.7926] (0.95 confidence)</td>
<td>32.7475 ns [32.6955, 32.803] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/near&#95;boundary&#95;insphere</td>
<td>94.5311 ns [94.2925, 94.7792] (0.95 confidence)</td>
<td>92.2219 ns [91.8899, 92.5626] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/near&#95;boundary&#95;lifted</td>
<td>95.1302 ns [94.7213, 95.565] (0.95 confidence)</td>
<td>91.2777 ns [91.0512, 91.5091] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/boundary&#95;point&#95;distance</td>
<td>50.9981 ns [50.888, 51.1097] (0.95 confidence)</td>
<td>54.6747 ns [54.5304, 54.8235] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/boundary&#95;point&#95;insphere</td>
<td>1.57133 ns [1.54916, 1.59235] (0.95 confidence)</td>
<td>1.56298 ns [1.53286, 1.59204] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/boundary&#95;point&#95;lifted</td>
<td>3668.74 ns [3660.8, 3677.01] (0.95 confidence)</td>
<td>3767.89 ns [3749.64, 3792.2] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/far&#95;point&#95;distance</td>
<td>50.9772 ns [50.8599, 51.0953] (0.95 confidence)</td>
<td>53.7173 ns [53.5935, 53.8393] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/far&#95;point&#95;insphere</td>
<td>749.122 ns [743.645, 755.32] (0.95 confidence)</td>
<td>736.251 ns [734.968, 737.559] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/far&#95;point&#95;lifted</td>
<td>753 ns [750.441, 755.613] (0.95 confidence)</td>
<td>738.213 ns [735.498, 741.145] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/near&#95;boundary&#95;distance</td>
<td>53.1202 ns [52.8804, 53.372] (0.95 confidence)</td>
<td>53.8261 ns [53.7072, 53.945] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/near&#95;boundary&#95;insphere</td>
<td>766.167 ns [762.646, 769.643] (0.95 confidence)</td>
<td>738.957 ns [736.84, 741.114] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/near&#95;boundary&#95;lifted</td>
<td>740.171 ns [737.355, 743.093] (0.95 confidence)</td>
<td>733.476 ns [731.047, 735.986] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/boundary&#95;point&#95;distance</td>
<td>87.4714 ns [87.2609, 87.6868] (0.95 confidence)</td>
<td>87.3212 ns [87.0458, 87.5977] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/boundary&#95;point&#95;insphere</td>
<td>1.69288 ns [1.68869, 1.69786] (0.95 confidence)</td>
<td>1.70035 ns [1.69221, 1.71066] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/boundary&#95;point&#95;lifted</td>
<td>6047.39 ns [6008.53, 6095.32] (0.95 confidence)</td>
<td>5867.63 ns [5857.43, 5878.07] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/far&#95;point&#95;distance</td>
<td>88.2959 ns [87.8611, 88.8142] (0.95 confidence)</td>
<td>87.2551 ns [86.9283, 87.586] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/far&#95;point&#95;insphere</td>
<td>1364.07 ns [1359.99, 1368.36] (0.95 confidence)</td>
<td>1299.78 ns [1296.04, 1303.57] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/far&#95;point&#95;lifted</td>
<td>1343.25 ns [1340.42, 1346.12] (0.95 confidence)</td>
<td>1286.39 ns [1283.31, 1289.59] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/near&#95;boundary&#95;distance</td>
<td>88.5645 ns [88.23, 88.918] (0.95 confidence)</td>
<td>87.9891 ns [87.6638, 88.3188] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/near&#95;boundary&#95;insphere</td>
<td>1342.36 ns [1337.99, 1347.03] (0.95 confidence)</td>
<td>1264.07 ns [1259.86, 1268.42] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/near&#95;boundary&#95;lifted</td>
<td>1343.51 ns [1339.28, 1347.75] (0.95 confidence)</td>
<td>1276.64 ns [1271.19, 1282.32] (0.95 confidence)</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;2d/vertices&#95;120&#95;simplices&#95;226</td>
<td>1.61314e+06 ns [1.60877e+06, 1.61763e+06] (0.95 confidence)</td>
<td>1.58605e+06 ns [1.58205e+06, 1.59041e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;3d/vertices&#95;30&#95;simplices&#95;115</td>
<td>1.59381e+08 ns [1.59147e+08, 1.59635e+08] (0.95 confidence)</td>
<td>1.58796e+08 ns [1.58487e+08, 1.59104e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;4d/vertices&#95;16&#95;simplices&#95;68</td>
<td>4.79355e+08 ns [4.79046e+08, 4.79656e+08] (0.95 confidence)</td>
<td>4.71827e+08 ns [4.71496e+08, 4.7218e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;5d/vertices&#95;10&#95;simplices&#95;20</td>
<td>1.35104e+08 ns [1.34874e+08, 1.35354e+08] (0.95 confidence)</td>
<td>1.3601e+08 ns [1.35542e+08, 1.36586e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;2d/10</td>
<td>438463 ns [437691, 439250] (0.95 confidence)</td>
<td>446524 ns [445029, 448210] (0.95 confidence)</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;2d&#95;adversarial/10</td>
<td>2.86142e+06 ns [2.85555e+06, 2.86739e+06] (0.95 confidence)</td>
<td>2.88115e+06 ns [2.86639e+06, 2.90276e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;3d/10</td>
<td>2.81502e+06 ns [2.79423e+06, 2.83573e+06] (0.95 confidence)</td>
<td>2.78132e+06 ns [2.76522e+06, 2.79975e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;3d&#95;adversarial/10</td>
<td>1.39265e+07 ns [1.38698e+07, 1.39934e+07] (0.95 confidence)</td>
<td>1.35294e+07 ns [1.34597e+07, 1.36191e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;4d/6</td>
<td>2.65165e+07 ns [2.64711e+07, 2.65605e+07] (0.95 confidence)</td>
<td>2.66044e+07 ns [2.64984e+07, 2.67056e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;4d&#95;adversarial/6</td>
<td>6.57378e+07 ns [6.55044e+07, 6.59952e+07] (0.95 confidence)</td>
<td>6.71733e+07 ns [6.68671e+07, 6.75122e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;5d/4</td>
<td>4.79581e+07 ns [4.79015e+07, 4.80238e+07] (0.95 confidence)</td>
<td>5.156e+07 ns [5.13765e+07, 5.17438e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;5d&#95;adversarial/4</td>
<td>5.18702e+07 ns [5.16746e+07, 5.20777e+07] (0.95 confidence)</td>
<td>5.59724e+07 ns [5.58076e+07, 5.61503e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/2d/vertices&#95;100&#95;simplices&#95;187&#95;seed&#95;dd618c77b9187836&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>7081.94 ns [7068.22, 7095.66] (0.95 confidence)</td>
<td>7239.28 ns [7216.59, 7272.07] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/2d/vertices&#95;2000&#95;simplices&#95;3978&#95;seed&#95;01acf93075e96412&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>7274.62 ns [7263.47, 7290.86] (0.95 confidence)</td>
<td>7512.78 ns [7490.83, 7537.37] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/2d/vertices&#95;500&#95;simplices&#95;983&#95;seed&#95;1493be4e9d7a5906&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>7398.82 ns [7385.78, 7411.32] (0.95 confidence)</td>
<td>7710.9 ns [7650.55, 7799.29] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/3d/vertices&#95;150&#95;simplices&#95;829&#95;seed&#95;a44b52b395a4b44d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>14620.8 ns [14552.5, 14700.4] (0.95 confidence)</td>
<td>15143.6 ns [15082.8, 15206.8] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/3d/vertices&#95;500&#95;simplices&#95;3052&#95;seed&#95;1493be4f9d7a5907&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>14483 ns [14428.5, 14533] (0.95 confidence)</td>
<td>15071.4 ns [15023.9, 15117.8] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/3d/vertices&#95;50&#95;simplices&#95;222&#95;seed&#95;f61fc639dc8c3c19&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>14566.2 ns [14526.3, 14605.3] (0.95 confidence)</td>
<td>15218.3 ns [15163, 15277.9] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/4d/vertices&#95;100&#95;simplices&#95;1677&#95;seed&#95;dd618c71b9187830&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>1.98439e+06 ns [1.98071e+06, 1.98819e+06] (0.95 confidence)</td>
<td>2.03576e+06 ns [2.0319e+06, 2.03999e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/4d/vertices&#95;50&#95;simplices&#95;626&#95;seed&#95;f61fc63edc8c3c1e&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>1.96e+06 ns [1.94697e+06, 1.97463e+06] (0.95 confidence)</td>
<td>2.02676e+06 ns [2.00081e+06, 2.05853e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/5d/vertices&#95;25&#95;simplices&#95;410&#95;seed&#95;63a0e3186e461e08&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>4.77285e+06 ns [4.74467e+06, 4.79882e+06] (0.95 confidence)</td>
<td>4.82906e+06 ns [4.80623e+06, 4.85173e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/5d/vertices&#95;40&#95;simplices&#95;1208&#95;seed&#95;a86104fee3a3634d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>4.85416e+06 ns [4.83084e+06, 4.87952e+06] (0.95 confidence)</td>
<td>4.87375e+06 ns [4.85958e+06, 4.88793e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/2d/vertices&#95;100&#95;simplices&#95;187&#95;seed&#95;dd618c77b9187836&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>73882.7 ns [73699.6, 74068.9] (0.95 confidence)</td>
<td>76547.4 ns [75702, 77660.7] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/2d/vertices&#95;2000&#95;simplices&#95;3978&#95;seed&#95;01acf93075e96412&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>268491 ns [266780, 270672] (0.95 confidence)</td>
<td>276458 ns [274145, 279685] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/2d/vertices&#95;500&#95;simplices&#95;983&#95;seed&#95;1493be4e9d7a5906&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>123961 ns [123167, 124787] (0.95 confidence)</td>
<td>125920 ns [125494, 126310] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/3d/vertices&#95;150&#95;simplices&#95;829&#95;seed&#95;a44b52b395a4b44d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>124045 ns [123827, 124332] (0.95 confidence)</td>
<td>129863 ns [128882, 130850] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/3d/vertices&#95;500&#95;simplices&#95;3052&#95;seed&#95;1493be4f9d7a5907&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>191201 ns [189411, 193110] (0.95 confidence)</td>
<td>201218 ns [197993, 205126] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/3d/vertices&#95;50&#95;simplices&#95;222&#95;seed&#95;f61fc639dc8c3c19&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>64983.6 ns [64694.3, 65309.3] (0.95 confidence)</td>
<td>69547.6 ns [68299.2, 71462.6] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/4d/vertices&#95;100&#95;simplices&#95;1677&#95;seed&#95;dd618c71b9187830&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>1.12189e+07 ns [1.12025e+07, 1.12351e+07] (0.95 confidence)</td>
<td>1.16001e+07 ns [1.1538e+07, 1.16563e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/4d/vertices&#95;50&#95;simplices&#95;626&#95;seed&#95;f61fc63edc8c3c1e&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>1.12709e+07 ns [1.11852e+07, 1.13586e+07] (0.95 confidence)</td>
<td>1.14055e+07 ns [1.1377e+07, 1.14328e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/5d/vertices&#95;25&#95;simplices&#95;410&#95;seed&#95;63a0e3186e461e08&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>2.19654e+07 ns [2.19286e+07, 2.20029e+07] (0.95 confidence)</td>
<td>2.26343e+07 ns [2.25583e+07, 2.27176e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/5d/vertices&#95;40&#95;simplices&#95;1208&#95;seed&#95;a86104fee3a3634d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>4.00702e+07 ns [3.99473e+07, 4.02048e+07] (0.95 confidence)</td>
<td>4.13841e+07 ns [4.10856e+07, 4.17361e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;2d/simplices&#95;226</td>
<td>115003 ns [113590, 116852] (0.95 confidence)</td>
<td>118699 ns [118022, 119742] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;3d/simplices&#95;115</td>
<td>141173 ns [140126, 142660] (0.95 confidence)</td>
<td>155207 ns [153563, 156859] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;4d/simplices&#95;68</td>
<td>1.20253e+06 ns [1.19344e+06, 1.21485e+06] (0.95 confidence)</td>
<td>1.27174e+06 ns [1.26183e+06, 1.28122e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;5d/simplices&#95;20</td>
<td>455130 ns [452645, 458226] (0.95 confidence)</td>
<td>483222 ns [480869, 486056] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;2d/simplices&#95;226</td>
<td>1.32186e+06 ns [1.31451e+06, 1.3356e+06] (0.95 confidence)</td>
<td>1.32725e+06 ns [1.32312e+06, 1.33212e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;3d/simplices&#95;115</td>
<td>1.549e+08 ns [1.54525e+08, 1.55339e+08] (0.95 confidence)</td>
<td>1.59589e+08 ns [1.592e+08, 1.59906e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;4d/simplices&#95;68</td>
<td>4.6906e+08 ns [4.6741e+08, 4.71092e+08] (0.95 confidence)</td>
<td>4.71969e+08 ns [4.71084e+08, 4.72921e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;5d/simplices&#95;20</td>
<td>1.33311e+08 ns [1.33108e+08, 1.33521e+08] (0.95 confidence)</td>
<td>1.38963e+08 ns [1.36488e+08, 1.4229e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;2d/simplices&#95;226</td>
<td>1.23762e+06 ns [1.23506e+06, 1.24136e+06] (0.95 confidence)</td>
<td>1.25076e+06 ns [1.24641e+06, 1.25557e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;3d/simplices&#95;115</td>
<td>1.54604e+08 ns [1.54401e+08, 1.54873e+08] (0.95 confidence)</td>
<td>1.59552e+08 ns [1.5926e+08, 1.59807e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;4d/simplices&#95;68</td>
<td>4.6949e+08 ns [4.68395e+08, 4.71262e+08] (0.95 confidence)</td>
<td>4.72816e+08 ns [4.71484e+08, 4.74132e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;5d/simplices&#95;20</td>
<td>1.32966e+08 ns [1.32768e+08, 1.33227e+08] (0.95 confidence)</td>
<td>1.36289e+08 ns [1.3595e+08, 1.36622e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>random/insphere&#95;1000&#95;queries</td>
<td>414983 ns [413766, 416195] (0.95 confidence)</td>
<td>425759 ns [424725, 426748] (0.95 confidence)</td>
</tr>
<tr>
<td>random/insphere&#95;distance&#95;1000&#95;queries</td>
<td>63190.2 ns [62875, 63503.8] (0.95 confidence)</td>
<td>65623.2 ns [65453.4, 65795] (0.95 confidence)</td>
</tr>
<tr>
<td>random/insphere&#95;lifted&#95;1000&#95;queries</td>
<td>415100 ns [413851, 416329] (0.95 confidence)</td>
<td>421148 ns [419926, 422384] (0.95 confidence)</td>
</tr>
<tr>
<td>tds&#95;new&#95;2d/tds&#95;new/500</td>
<td>4.58298e+07 ns [4.56359e+07, 4.60524e+07] (0.95 confidence)</td>
<td>4.58099e+07 ns [4.57014e+07, 4.59412e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>tds&#95;new&#95;2d/tds&#95;new&#95;adversarial/500</td>
<td>5.36853e+07 ns [5.36105e+07, 5.37802e+07] (0.95 confidence)</td>
<td>5.52124e+07 ns [5.45396e+07, 5.60237e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>tds&#95;new&#95;3d/tds&#95;new/100</td>
<td>3.69066e+07 ns [3.68702e+07, 3.69438e+07] (0.95 confidence)</td>
<td>3.75031e+07 ns [3.72991e+07, 3.78407e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>tds&#95;new&#95;3d/tds&#95;new&#95;adversarial/100</td>
<td>8.36544e+07 ns [8.35455e+07, 8.37654e+07] (0.95 confidence)</td>
<td>8.5111e+07 ns [8.49203e+07, 8.53172e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>tds&#95;new&#95;4d/tds&#95;new/30</td>
<td>9.27933e+07 ns [9.25165e+07, 9.30309e+07] (0.95 confidence)</td>
<td>9.69391e+07 ns [9.64397e+07, 9.7463e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>tds&#95;new&#95;4d/tds&#95;new&#95;adversarial/30</td>
<td>8.76889e+07 ns [8.74524e+07, 8.79472e+07] (0.95 confidence)</td>
<td>9.32569e+07 ns [9.26047e+07, 9.39296e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>tds&#95;new&#95;5d/tds&#95;new/12</td>
<td>2.03289e+07 ns [2.02847e+07, 2.03842e+07] (0.95 confidence)</td>
<td>2.20169e+07 ns [2.18847e+07, 2.21549e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>tds&#95;new&#95;5d/tds&#95;new&#95;adversarial/12</td>
<td>1.8768e+07 ns [1.8727e+07, 1.88139e+07] (0.95 confidence)</td>
<td>2.0144e+07 ns [2.00767e+07, 2.02078e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;2d/500</td>
<td>801130 ns [798392, 804216] (0.95 confidence)</td>
<td>803256 ns [796832, 810647] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;2d&#95;adversarial/500</td>
<td>1.11441e+06 ns [1.10652e+06, 1.12334e+06] (0.95 confidence)</td>
<td>1.10062e+06 ns [1.09588e+06, 1.10648e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;3d/100</td>
<td>1.09475e+06 ns [1.08545e+06, 1.10593e+06] (0.95 confidence)</td>
<td>1.26063e+06 ns [1.24818e+06, 1.27497e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;3d&#95;adversarial/100</td>
<td>2.0728e+06 ns [2.05892e+06, 2.09007e+06] (0.95 confidence)</td>
<td>2.29704e+06 ns [2.27329e+06, 2.32178e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;4d/30</td>
<td>3.91667e+06 ns [3.89426e+06, 3.94176e+06] (0.95 confidence)</td>
<td>4.65163e+06 ns [4.62715e+06, 4.67919e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;4d&#95;adversarial/30</td>
<td>3.9108e+06 ns [3.89131e+06, 3.93431e+06] (0.95 confidence)</td>
<td>4.53008e+06 ns [4.49007e+06, 4.57261e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;5d/12</td>
<td>1.64355e+06 ns [1.63576e+06, 1.65299e+06] (0.95 confidence)</td>
<td>2.22168e+06 ns [2.21012e+06, 2.23297e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;5d&#95;adversarial/12</td>
<td>1.45774e+06 ns [1.45228e+06, 1.46423e+06] (0.95 confidence)</td>
<td>2.03222e+06 ns [2.00881e+06, 2.05906e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;2d/500</td>
<td>811161 ns [804633, 818051] (0.95 confidence)</td>
<td>807948 ns [805044, 810962] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;2d&#95;adversarial/500</td>
<td>1.10928e+06 ns [1.10314e+06, 1.11699e+06] (0.95 confidence)</td>
<td>1.10756e+06 ns [1.10003e+06, 1.11638e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;3d/100</td>
<td>1.08753e+06 ns [1.08005e+06, 1.09706e+06] (0.95 confidence)</td>
<td>1.26255e+06 ns [1.25076e+06, 1.27508e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;3d&#95;adversarial/100</td>
<td>2.09203e+06 ns [2.08352e+06, 2.10219e+06] (0.95 confidence)</td>
<td>2.21589e+06 ns [2.20568e+06, 2.23e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;4d/30</td>
<td>3.90339e+06 ns [3.89295e+06, 3.91543e+06] (0.95 confidence)</td>
<td>4.60834e+06 ns [4.5705e+06, 4.65022e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;4d&#95;adversarial/30</td>
<td>3.88432e+06 ns [3.86614e+06, 3.90274e+06] (0.95 confidence)</td>
<td>4.5034e+06 ns [4.47919e+06, 4.53115e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;5d/12</td>
<td>1.63436e+06 ns [1.62792e+06, 1.64243e+06] (0.95 confidence)</td>
<td>2.20013e+06 ns [2.18496e+06, 2.21644e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;5d&#95;adversarial/12</td>
<td>1.46203e+06 ns [1.45489e+06, 1.47004e+06] (0.95 confidence)</td>
<td>2.05526e+06 ns [2.04179e+06, 2.06787e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/validate&#95;2d/500</td>
<td>8.08984e+06 ns [8.0598e+06, 8.12096e+06] (0.95 confidence)</td>
<td>8.26879e+06 ns [8.24496e+06, 8.3039e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/validate&#95;2d&#95;adversarial/500</td>
<td>5.33314e+07 ns [5.31204e+07, 5.36288e+07] (0.95 confidence)</td>
<td>4.9046e+07 ns [4.8909e+07, 4.92022e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/validate&#95;3d/100</td>
<td>8.32432e+08 ns [8.31613e+08, 8.33192e+08] (0.95 confidence)</td>
<td>8.53956e+08 ns [8.45629e+08, 8.64738e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/validate&#95;3d&#95;adversarial/100</td>
<td>1.03672e+09 ns [1.03521e+09, 1.03861e+09] (0.95 confidence)</td>
<td>1.02922e+09 ns [1.02506e+09, 1.03481e+09] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/validate&#95;4d/30</td>
<td>3.89267e+09 ns [3.88764e+09, 3.89776e+09] (0.95 confidence)</td>
<td>3.88936e+09 ns [3.88379e+09, 3.89558e+09] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/validate&#95;4d&#95;adversarial/30</td>
<td>4.01795e+09 ns [4.01429e+09, 4.02161e+09] (0.95 confidence)</td>
<td>3.95997e+09 ns [3.95652e+09, 3.96281e+09] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/validate&#95;5d/12</td>
<td>8.68036e+08 ns [8.67052e+08, 8.6905e+08] (0.95 confidence)</td>
<td>8.90802e+08 ns [8.88827e+08, 8.93075e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/validate&#95;5d&#95;adversarial/12</td>
<td>8.12403e+08 ns [8.10949e+08, 8.14223e+08] (0.95 confidence)</td>
<td>8.27638e+08 ns [8.26199e+08, 8.2914e+08] (0.95 confidence)</td>
</tr>
</tbody>
</table>

## Median (ns)

<table>
<thead>
<tr>
<th>Case</th>
<th>baseline</th>
<th>current</th>
</tr>
</thead>
<tbody>
<tr>
<td>2d/insphere</td>
<td>50.1092 ns [49.848, 50.5596] (0.95 confidence)</td>
<td>49.7992 ns [49.5648, 49.944] (0.95 confidence)</td>
</tr>
<tr>
<td>2d/insphere&#95;distance</td>
<td>13.4401 ns [13.4206, 13.4698] (0.95 confidence)</td>
<td>13.726 ns [13.6908, 13.7554] (0.95 confidence)</td>
</tr>
<tr>
<td>2d/insphere&#95;lifted</td>
<td>49.2497 ns [49.0745, 49.592] (0.95 confidence)</td>
<td>52.2912 ns [51.4278, 52.7419] (0.95 confidence)</td>
</tr>
<tr>
<td>3d/insphere</td>
<td>106.658 ns [106.221, 107.189] (0.95 confidence)</td>
<td>111.255 ns [110.496, 112.198] (0.95 confidence)</td>
</tr>
<tr>
<td>3d/insphere&#95;distance</td>
<td>32.4923 ns [32.3933, 32.6123] (0.95 confidence)</td>
<td>33.1128 ns [33.0555, 33.2232] (0.95 confidence)</td>
</tr>
<tr>
<td>3d/insphere&#95;lifted</td>
<td>104.815 ns [104.6, 105.321] (0.95 confidence)</td>
<td>106.862 ns [106.441, 107.378] (0.95 confidence)</td>
</tr>
<tr>
<td>4d/insphere</td>
<td>747.004 ns [743.918, 749.977] (0.95 confidence)</td>
<td>759.223 ns [757.018, 760.743] (0.95 confidence)</td>
</tr>
<tr>
<td>4d/insphere&#95;distance</td>
<td>50.8489 ns [50.7376, 50.9947] (0.95 confidence)</td>
<td>54.5802 ns [54.4765, 54.6172] (0.95 confidence)</td>
</tr>
<tr>
<td>4d/insphere&#95;lifted</td>
<td>753.788 ns [750.524, 758.167] (0.95 confidence)</td>
<td>759.285 ns [756.423, 763.036] (0.95 confidence)</td>
</tr>
<tr>
<td>5d/insphere</td>
<td>1385.45 ns [1382.1, 1394.56] (0.95 confidence)</td>
<td>1286.71 ns [1279.96, 1290.72] (0.95 confidence)</td>
</tr>
<tr>
<td>5d/insphere&#95;distance</td>
<td>87.3461 ns [87.0225, 87.7041] (0.95 confidence)</td>
<td>86.7869 ns [86.5354, 87.0731] (0.95 confidence)</td>
</tr>
<tr>
<td>5d/insphere&#95;lifted</td>
<td>1376.03 ns [1369.93, 1379.99] (0.95 confidence)</td>
<td>1302.44 ns [1299.59, 1309.34] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k1&#95;roundtrip</td>
<td>54000.8 ns [53909.2, 54035.1] (0.95 confidence)</td>
<td>56204.2 ns [56072.9, 56509.9] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k1&#95;roundtrip&#95;adversarial</td>
<td>162505 ns [162330, 162869] (0.95 confidence)</td>
<td>166362 ns [166098, 166712] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k2&#95;edge&#95;flip</td>
<td>26249.9 ns [26232.6, 26287.1] (0.95 confidence)</td>
<td>27565.7 ns [27524.3, 27644.7] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k2&#95;edge&#95;flip&#95;adversarial</td>
<td>99726.9 ns [99343, 104002] (0.95 confidence)</td>
<td>102897 ns [102472, 103421] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k1&#95;roundtrip</td>
<td>2.34166e+06 ns [2.3307e+06, 2.34545e+06] (0.95 confidence)</td>
<td>2.41318e+06 ns [2.41013e+06, 2.41962e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k1&#95;roundtrip&#95;adversarial</td>
<td>1.95074e+06 ns [1.94268e+06, 1.96614e+06] (0.95 confidence)</td>
<td>2.00787e+06 ns [2.00469e+06, 2.00924e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k2&#95;roundtrip</td>
<td>6.48654e+06 ns [6.4556e+06, 6.71931e+06] (0.95 confidence)</td>
<td>6.63016e+06 ns [6.61571e+06, 6.64311e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k2&#95;roundtrip&#95;adversarial</td>
<td>5.81532e+06 ns [5.80934e+06, 5.8581e+06] (0.95 confidence)</td>
<td>6.0278e+06 ns [6.01418e+06, 6.03928e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k3&#95;forward</td>
<td>1.35868e+06 ns [1.35581e+06, 1.36452e+06] (0.95 confidence)</td>
<td>1.39463e+06 ns [1.39224e+06, 1.39778e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k3&#95;forward&#95;adversarial</td>
<td>1.88336e+06 ns [1.87501e+06, 1.89621e+06] (0.95 confidence)</td>
<td>1.87019e+06 ns [1.86514e+06, 1.87771e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k1&#95;roundtrip</td>
<td>3.90274e+07 ns [3.88461e+07, 3.94982e+07] (0.95 confidence)</td>
<td>3.95682e+07 ns [3.94503e+07, 3.97138e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k1&#95;roundtrip&#95;adversarial</td>
<td>3.31514e+07 ns [3.29728e+07, 3.33958e+07] (0.95 confidence)</td>
<td>3.33688e+07 ns [3.33203e+07, 3.34525e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k2&#95;roundtrip</td>
<td>5.61352e+07 ns [5.58892e+07, 5.64552e+07] (0.95 confidence)</td>
<td>5.63513e+07 ns [5.62663e+07, 5.64285e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k2&#95;roundtrip&#95;adversarial</td>
<td>1.99368e+08 ns [1.98132e+08, 2.02114e+08] (0.95 confidence)</td>
<td>1.98789e+08 ns [1.98305e+08, 1.991e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k3&#95;roundtrip</td>
<td>6.55242e+07 ns [6.48063e+07, 6.62473e+07] (0.95 confidence)</td>
<td>6.50276e+07 ns [6.48731e+07, 6.53269e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k3&#95;roundtrip&#95;adversarial</td>
<td>3.52332e+07 ns [3.50915e+07, 3.53577e+07] (0.95 confidence)</td>
<td>3.51389e+07 ns [3.50783e+07, 3.52025e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k1&#95;roundtrip</td>
<td>1.56021e+08 ns [1.5524e+08, 1.56609e+08] (0.95 confidence)</td>
<td>1.59211e+08 ns [1.58697e+08, 1.5958e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k1&#95;roundtrip&#95;adversarial</td>
<td>1.60523e+08 ns [1.59329e+08, 1.62374e+08] (0.95 confidence)</td>
<td>1.62075e+08 ns [1.61284e+08, 1.62754e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k2&#95;roundtrip</td>
<td>329262 ns [328432, 329725] (0.95 confidence)</td>
<td>329144 ns [328738, 329472] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k2&#95;roundtrip&#95;adversarial</td>
<td>351655 ns [351125, 353596] (0.95 confidence)</td>
<td>355912 ns [355447, 356975] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k3&#95;roundtrip</td>
<td>3.25946e+08 ns [3.25327e+08, 3.2806e+08] (0.95 confidence)</td>
<td>3.3058e+08 ns [3.29996e+08, 3.31248e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k3&#95;roundtrip&#95;adversarial</td>
<td>9.10817e+08 ns [9.05801e+08, 9.13939e+08] (0.95 confidence)</td>
<td>8.80643e+08 ns [8.79082e+08, 8.87402e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;2d/500</td>
<td>209562 ns [209034, 210230] (0.95 confidence)</td>
<td>211221 ns [210268, 212175] (0.95 confidence)</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;2d&#95;adversarial/500</td>
<td>209513 ns [209181, 209774] (0.95 confidence)</td>
<td>210209 ns [209479, 211489] (0.95 confidence)</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;3d/100</td>
<td>166076 ns [165838, 166377] (0.95 confidence)</td>
<td>167047 ns [166601, 167730] (0.95 confidence)</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;3d&#95;adversarial/100</td>
<td>168031 ns [167794, 168426] (0.95 confidence)</td>
<td>168502 ns [167967, 169046] (0.95 confidence)</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;4d/30</td>
<td>131569 ns [131126, 131936] (0.95 confidence)</td>
<td>137524 ns [137215, 138533] (0.95 confidence)</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;4d&#95;adversarial/30</td>
<td>131032 ns [130833, 131317] (0.95 confidence)</td>
<td>134701 ns [133737, 135556] (0.95 confidence)</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;5d/12</td>
<td>36654.2 ns [36470.3, 36765.5] (0.95 confidence)</td>
<td>37389.5 ns [37223.5, 37551.3] (0.95 confidence)</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;5d&#95;adversarial/12</td>
<td>35795.4 ns [35701.2, 35822.3] (0.95 confidence)</td>
<td>37132.8 ns [36604.9, 37270.3] (0.95 confidence)</td>
</tr>
<tr>
<td>circumcenter/solve&#95;path/near&#95;singular&#95;exact&#95;fallback&#95;3d</td>
<td>17640.3 ns [17580.6, 17731.2] (0.95 confidence)</td>
<td>17613.1 ns [17600.3, 17640.5] (0.95 confidence)</td>
</tr>
<tr>
<td>circumcenter/solve&#95;path/regular&#95;lu&#95;3d</td>
<td>22.9798 ns [22.9558, 23.0117] (0.95 confidence)</td>
<td>23.7081 ns [23.6859, 23.7351] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;2d/500</td>
<td>342006 ns [340284, 346989] (0.95 confidence)</td>
<td>391106 ns [390514, 391997] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;2d&#95;adversarial/500</td>
<td>332161 ns [331937, 332384] (0.95 confidence)</td>
<td>389051 ns [388532, 390267] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;3d/100</td>
<td>275635 ns [275398, 275863] (0.95 confidence)</td>
<td>278225 ns [276392, 279325] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;3d&#95;adversarial/100</td>
<td>3.99637e+06 ns [3.9847e+06, 4.00945e+06] (0.95 confidence)</td>
<td>4.03778e+06 ns [4.02506e+06, 4.04675e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;4d/30</td>
<td>8.74927e+06 ns [8.70822e+06, 8.78023e+06] (0.95 confidence)</td>
<td>8.94034e+06 ns [8.91084e+06, 8.97114e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;4d&#95;adversarial/30</td>
<td>6.9623e+06 ns [6.94382e+06, 7.04273e+06] (0.95 confidence)</td>
<td>7.0671e+06 ns [7.0428e+06, 7.08885e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;5d/12</td>
<td>3.06658e+06 ns [3.05793e+06, 3.08456e+06] (0.95 confidence)</td>
<td>3.0662e+06 ns [3.05676e+06, 3.07035e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;5d&#95;adversarial/12</td>
<td>2.63396e+06 ns [2.62985e+06, 2.63979e+06] (0.95 confidence)</td>
<td>2.74079e+06 ns [2.69681e+06, 2.75869e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;nearest&#95;visible&#95;facet&#95;3d/100</td>
<td>1137.03 ns [1128.88, 1142.16] (0.95 confidence)</td>
<td>1147.85 ns [1141.34, 1157.37] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;nearest&#95;visible&#95;facet&#95;3d&#95;adversarial/100</td>
<td>40780.6 ns [40718, 41006.2] (0.95 confidence)</td>
<td>40876 ns [40835.4, 41076.6] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;visible&#95;facets&#95;3d/100</td>
<td>1201.26 ns [1199.6, 1204.63] (0.95 confidence)</td>
<td>1218.89 ns [1212.32, 1229.78] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;visible&#95;facets&#95;3d&#95;adversarial/100</td>
<td>40899.2 ns [40825.6, 41050.1] (0.95 confidence)</td>
<td>41117.1 ns [40947.5, 41210.7] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/is&#95;point&#95;outside&#95;3d/100</td>
<td>94.5441 ns [94.463, 94.6819] (0.95 confidence)</td>
<td>95.7865 ns [95.6158, 96.0063] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/is&#95;point&#95;outside&#95;3d&#95;adversarial/100</td>
<td>769.964 ns [766.355, 773.105] (0.95 confidence)</td>
<td>779.861 ns [777.258, 781.948] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/boundary&#95;point&#95;distance</td>
<td>13.5288 ns [13.51, 13.5461] (0.95 confidence)</td>
<td>13.4062 ns [13.3819, 13.4332] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/boundary&#95;point&#95;insphere</td>
<td>0.834018 ns [0.829442, 0.840486] (0.95 confidence)</td>
<td>0.824192 ns [0.820206, 0.828035] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/boundary&#95;point&#95;lifted</td>
<td>971.276 ns [968.414, 973.27] (0.95 confidence)</td>
<td>978.066 ns [975.719, 979.711] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/far&#95;point&#95;distance</td>
<td>13.5336 ns [13.472, 13.562] (0.95 confidence)</td>
<td>13.6807 ns [13.6655, 13.6952] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/far&#95;point&#95;insphere</td>
<td>48.7027 ns [48.5061, 49.2727] (0.95 confidence)</td>
<td>50.4632 ns [50.1711, 50.9138] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/far&#95;point&#95;lifted</td>
<td>50.3157 ns [50.0255, 50.8672] (0.95 confidence)</td>
<td>50.2393 ns [49.9158, 50.6124] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/near&#95;boundary&#95;distance</td>
<td>13.7547 ns [13.7128, 13.8018] (0.95 confidence)</td>
<td>13.863 ns [13.8355, 13.8754] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/near&#95;boundary&#95;insphere</td>
<td>46.7575 ns [46.4319, 47.0373] (0.95 confidence)</td>
<td>47.9387 ns [47.3356, 48.7013] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/near&#95;boundary&#95;lifted</td>
<td>46.9163 ns [46.674, 47.2773] (0.95 confidence)</td>
<td>47.1651 ns [46.9816, 47.4493] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/boundary&#95;point&#95;distance</td>
<td>32.7802 ns [32.6747, 32.8726] (0.95 confidence)</td>
<td>32.7567 ns [32.7287, 32.8127] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/boundary&#95;point&#95;insphere</td>
<td>1.20305 ns [1.19205, 1.2131] (0.95 confidence)</td>
<td>1.18514 ns [1.18415, 1.18685] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/boundary&#95;point&#95;lifted</td>
<td>1743.94 ns [1741.74, 1747.78] (0.95 confidence)</td>
<td>1776.77 ns [1775.6, 1778.42] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/far&#95;point&#95;distance</td>
<td>32.9444 ns [32.7991, 33.0584] (0.95 confidence)</td>
<td>33.055 ns [32.9826, 33.1455] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/far&#95;point&#95;insphere</td>
<td>106.896 ns [106.396, 107.985] (0.95 confidence)</td>
<td>108.06 ns [107.29, 108.514] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/far&#95;point&#95;lifted</td>
<td>110.883 ns [109.859, 111.96] (0.95 confidence)</td>
<td>111.93 ns [111.072, 112.671] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/near&#95;boundary&#95;distance</td>
<td>32.5457 ns [32.4897, 32.6581] (0.95 confidence)</td>
<td>32.6778 ns [32.5913, 32.7874] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/near&#95;boundary&#95;insphere</td>
<td>94.6386 ns [94.2886, 94.8593] (0.95 confidence)</td>
<td>91.7483 ns [91.4309, 92.275] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/near&#95;boundary&#95;lifted</td>
<td>94.4608 ns [94.2695, 94.7991] (0.95 confidence)</td>
<td>91.1181 ns [90.8352, 91.4838] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/boundary&#95;point&#95;distance</td>
<td>50.9147 ns [50.8632, 51.0103] (0.95 confidence)</td>
<td>54.5232 ns [54.426, 54.7848] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/boundary&#95;point&#95;insphere</td>
<td>1.59133 ns [1.56825, 1.63252] (0.95 confidence)</td>
<td>1.62258 ns [1.59762, 1.63584] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/boundary&#95;point&#95;lifted</td>
<td>3661.35 ns [3644.14, 3669.88] (0.95 confidence)</td>
<td>3743.3 ns [3737.16, 3751.01] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/far&#95;point&#95;distance</td>
<td>50.9457 ns [50.7928, 51.1858] (0.95 confidence)</td>
<td>53.785 ns [53.6993, 53.8672] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/far&#95;point&#95;insphere</td>
<td>739.458 ns [736.823, 743.367] (0.95 confidence)</td>
<td>734.974 ns [734.041, 736.579] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/far&#95;point&#95;lifted</td>
<td>750.996 ns [748.18, 754.36] (0.95 confidence)</td>
<td>735.177 ns [733.162, 738.653] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/near&#95;boundary&#95;distance</td>
<td>52.8975 ns [52.7226, 53.0971] (0.95 confidence)</td>
<td>53.76 ns [53.6802, 53.8752] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/near&#95;boundary&#95;insphere</td>
<td>767.016 ns [762.379, 770.426] (0.95 confidence)</td>
<td>738.435 ns [735.268, 740.814] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/near&#95;boundary&#95;lifted</td>
<td>740.084 ns [734.833, 742.987] (0.95 confidence)</td>
<td>731.9 ns [730.117, 733.995] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/boundary&#95;point&#95;distance</td>
<td>87.3195 ns [87.0258, 87.6664] (0.95 confidence)</td>
<td>87.2981 ns [86.9738, 87.5919] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/boundary&#95;point&#95;insphere</td>
<td>1.6864 ns [1.68458, 1.68851] (0.95 confidence)</td>
<td>1.68596 ns [1.68471, 1.68775] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/boundary&#95;point&#95;lifted</td>
<td>5994 ns [5978.03, 6007.1] (0.95 confidence)</td>
<td>5866.53 ns [5853.3, 5878.73] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/far&#95;point&#95;distance</td>
<td>87.9312 ns [87.578, 88.4201] (0.95 confidence)</td>
<td>86.8675 ns [86.5583, 87.3029] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/far&#95;point&#95;insphere</td>
<td>1362.44 ns [1360.16, 1364.8] (0.95 confidence)</td>
<td>1299.78 ns [1294.16, 1305.41] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/far&#95;point&#95;lifted</td>
<td>1342.84 ns [1339.3, 1346.42] (0.95 confidence)</td>
<td>1283.44 ns [1278, 1291.41] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/near&#95;boundary&#95;distance</td>
<td>88.1716 ns [87.862, 88.6902] (0.95 confidence)</td>
<td>88.0049 ns [87.48, 88.4176] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/near&#95;boundary&#95;insphere</td>
<td>1341.82 ns [1337.86, 1347.84] (0.95 confidence)</td>
<td>1264.31 ns [1253.57, 1271.42] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/near&#95;boundary&#95;lifted</td>
<td>1344.52 ns [1339.95, 1348.64] (0.95 confidence)</td>
<td>1269.15 ns [1263.02, 1277.78] (0.95 confidence)</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;2d/vertices&#95;120&#95;simplices&#95;226</td>
<td>1.6106e+06 ns [1.60906e+06, 1.61988e+06] (0.95 confidence)</td>
<td>1.58499e+06 ns [1.58111e+06, 1.59224e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;3d/vertices&#95;30&#95;simplices&#95;115</td>
<td>1.59241e+08 ns [1.59056e+08, 1.59898e+08] (0.95 confidence)</td>
<td>1.58735e+08 ns [1.58588e+08, 1.59076e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;4d/vertices&#95;16&#95;simplices&#95;68</td>
<td>4.79301e+08 ns [4.79099e+08, 4.79747e+08] (0.95 confidence)</td>
<td>4.71789e+08 ns [4.71364e+08, 4.72273e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;5d/vertices&#95;10&#95;simplices&#95;20</td>
<td>1.35148e+08 ns [1.34706e+08, 1.35436e+08] (0.95 confidence)</td>
<td>1.35781e+08 ns [1.35361e+08, 1.36668e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;2d/10</td>
<td>438082 ns [437281, 439893] (0.95 confidence)</td>
<td>445654 ns [444366, 448249] (0.95 confidence)</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;2d&#95;adversarial/10</td>
<td>2.86017e+06 ns [2.85193e+06, 2.87156e+06] (0.95 confidence)</td>
<td>2.87317e+06 ns [2.86372e+06, 2.8881e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;3d/10</td>
<td>2.80449e+06 ns [2.78661e+06, 2.85818e+06] (0.95 confidence)</td>
<td>2.76847e+06 ns [2.75686e+06, 2.80461e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;3d&#95;adversarial/10</td>
<td>1.38972e+07 ns [1.38257e+07, 1.39696e+07] (0.95 confidence)</td>
<td>1.34891e+07 ns [1.34256e+07, 1.35395e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;4d/6</td>
<td>2.6514e+07 ns [2.64656e+07, 2.65863e+07] (0.95 confidence)</td>
<td>2.66587e+07 ns [2.64179e+07, 2.67239e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;4d&#95;adversarial/6</td>
<td>6.54877e+07 ns [6.54055e+07, 6.63194e+07] (0.95 confidence)</td>
<td>6.69383e+07 ns [6.65567e+07, 6.78261e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;5d/4</td>
<td>4.79069e+07 ns [4.78736e+07, 4.80433e+07] (0.95 confidence)</td>
<td>5.16133e+07 ns [5.12724e+07, 5.19268e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;5d&#95;adversarial/4</td>
<td>5.17334e+07 ns [5.15766e+07, 5.21e+07] (0.95 confidence)</td>
<td>5.59504e+07 ns [5.57077e+07, 5.62505e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/2d/vertices&#95;100&#95;simplices&#95;187&#95;seed&#95;dd618c77b9187836&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>7082.06 ns [7059.69, 7105.54] (0.95 confidence)</td>
<td>7224.27 ns [7207.74, 7250.53] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/2d/vertices&#95;2000&#95;simplices&#95;3978&#95;seed&#95;01acf93075e96412&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>7263.14 ns [7262.07, 7279.77] (0.95 confidence)</td>
<td>7501.88 ns [7480.79, 7547.05] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/2d/vertices&#95;500&#95;simplices&#95;983&#95;seed&#95;1493be4e9d7a5906&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>7399.87 ns [7378.24, 7420.05] (0.95 confidence)</td>
<td>7664.94 ns [7637.86, 7734.44] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/3d/vertices&#95;150&#95;simplices&#95;829&#95;seed&#95;a44b52b395a4b44d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>14609.1 ns [14578.8, 14648.2] (0.95 confidence)</td>
<td>15127.8 ns [15056.4, 15227.6] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/3d/vertices&#95;500&#95;simplices&#95;3052&#95;seed&#95;1493be4f9d7a5907&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>14495.2 ns [14424.1, 14547.9] (0.95 confidence)</td>
<td>15087.7 ns [14996.4, 15135.6] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/3d/vertices&#95;50&#95;simplices&#95;222&#95;seed&#95;f61fc639dc8c3c19&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>14576.4 ns [14505.6, 14611.1] (0.95 confidence)</td>
<td>15191.1 ns [15152.9, 15289.5] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/4d/vertices&#95;100&#95;simplices&#95;1677&#95;seed&#95;dd618c71b9187830&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>1.9836e+06 ns [1.97822e+06, 1.99155e+06] (0.95 confidence)</td>
<td>2.03493e+06 ns [2.02926e+06, 2.0414e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/4d/vertices&#95;50&#95;simplices&#95;626&#95;seed&#95;f61fc63edc8c3c1e&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>1.94638e+06 ns [1.94188e+06, 1.97995e+06] (0.95 confidence)</td>
<td>2.00914e+06 ns [1.99173e+06, 2.05971e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/5d/vertices&#95;25&#95;simplices&#95;410&#95;seed&#95;63a0e3186e461e08&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>4.79056e+06 ns [4.73236e+06, 4.80565e+06] (0.95 confidence)</td>
<td>4.8289e+06 ns [4.79594e+06, 4.85938e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/5d/vertices&#95;40&#95;simplices&#95;1208&#95;seed&#95;a86104fee3a3634d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>4.84244e+06 ns [4.814e+06, 4.88676e+06] (0.95 confidence)</td>
<td>4.87845e+06 ns [4.85105e+06, 4.88873e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/2d/vertices&#95;100&#95;simplices&#95;187&#95;seed&#95;dd618c77b9187836&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>73859.5 ns [73585, 74142.7] (0.95 confidence)</td>
<td>76133.8 ns [75632.7, 77069.9] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/2d/vertices&#95;2000&#95;simplices&#95;3978&#95;seed&#95;01acf93075e96412&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>266907 ns [266410, 270219] (0.95 confidence)</td>
<td>275166 ns [273843, 277460] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/2d/vertices&#95;500&#95;simplices&#95;983&#95;seed&#95;1493be4e9d7a5906&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>123902 ns [122725, 125259] (0.95 confidence)</td>
<td>126235 ns [125310, 126485] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/3d/vertices&#95;150&#95;simplices&#95;829&#95;seed&#95;a44b52b395a4b44d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>123998 ns [123737, 124169] (0.95 confidence)</td>
<td>129685 ns [128498, 130943] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/3d/vertices&#95;500&#95;simplices&#95;3052&#95;seed&#95;1493be4f9d7a5907&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>190140 ns [188533, 194990] (0.95 confidence)</td>
<td>200305 ns [196416, 204526] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/3d/vertices&#95;50&#95;simplices&#95;222&#95;seed&#95;f61fc639dc8c3c19&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>64754.9 ns [64610.4, 65431.5] (0.95 confidence)</td>
<td>68612.7 ns [68075, 69922.5] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/4d/vertices&#95;100&#95;simplices&#95;1677&#95;seed&#95;dd618c71b9187830&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>1.12311e+07 ns [1.11889e+07, 1.12375e+07] (0.95 confidence)</td>
<td>1.16295e+07 ns [1.15206e+07, 1.16777e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/4d/vertices&#95;50&#95;simplices&#95;626&#95;seed&#95;f61fc63edc8c3c1e&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>1.12474e+07 ns [1.11289e+07, 1.14043e+07] (0.95 confidence)</td>
<td>1.13948e+07 ns [1.13787e+07, 1.14494e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/5d/vertices&#95;25&#95;simplices&#95;410&#95;seed&#95;63a0e3186e461e08&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>2.19571e+07 ns [2.19042e+07, 2.20358e+07] (0.95 confidence)</td>
<td>2.25962e+07 ns [2.25014e+07, 2.27729e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/5d/vertices&#95;40&#95;simplices&#95;1208&#95;seed&#95;a86104fee3a3634d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>4.00131e+07 ns [3.98984e+07, 4.02994e+07] (0.95 confidence)</td>
<td>4.13241e+07 ns [4.09816e+07, 4.16525e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;2d/simplices&#95;226</td>
<td>113706 ns [113380, 116433] (0.95 confidence)</td>
<td>118087 ns [117946, 118774] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;3d/simplices&#95;115</td>
<td>140309 ns [139802, 141797] (0.95 confidence)</td>
<td>155413 ns [152286, 158090] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;4d/simplices&#95;68</td>
<td>1.19527e+06 ns [1.19108e+06, 1.20959e+06] (0.95 confidence)</td>
<td>1.26828e+06 ns [1.26473e+06, 1.28291e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;5d/simplices&#95;20</td>
<td>453331 ns [451896, 458299] (0.95 confidence)</td>
<td>482687 ns [479600, 484885] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;2d/simplices&#95;226</td>
<td>1.31516e+06 ns [1.31421e+06, 1.31691e+06] (0.95 confidence)</td>
<td>1.32428e+06 ns [1.3215e+06, 1.33438e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;3d/simplices&#95;115</td>
<td>1.5458e+08 ns [1.54382e+08, 1.55717e+08] (0.95 confidence)</td>
<td>1.59732e+08 ns [1.59239e+08, 1.60071e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;4d/simplices&#95;68</td>
<td>4.67943e+08 ns [4.6699e+08, 4.71205e+08] (0.95 confidence)</td>
<td>4.71706e+08 ns [4.7072e+08, 4.73105e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;5d/simplices&#95;20</td>
<td>1.33248e+08 ns [1.32979e+08, 1.33702e+08] (0.95 confidence)</td>
<td>1.36566e+08 ns [1.36235e+08, 1.42933e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;2d/simplices&#95;226</td>
<td>1.23551e+06 ns [1.23473e+06, 1.23851e+06] (0.95 confidence)</td>
<td>1.25009e+06 ns [1.24388e+06, 1.25688e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;3d/simplices&#95;115</td>
<td>1.54503e+08 ns [1.54353e+08, 1.54775e+08] (0.95 confidence)</td>
<td>1.59599e+08 ns [1.59298e+08, 1.59917e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;4d/simplices&#95;68</td>
<td>4.68775e+08 ns [4.68203e+08, 4.69334e+08] (0.95 confidence)</td>
<td>4.72702e+08 ns [4.71021e+08, 4.74873e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;5d/simplices&#95;20</td>
<td>1.32884e+08 ns [1.32736e+08, 1.33101e+08] (0.95 confidence)</td>
<td>1.3625e+08 ns [1.35804e+08, 1.36801e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>random/insphere&#95;1000&#95;queries</td>
<td>414611 ns [413134, 415783] (0.95 confidence)</td>
<td>425710 ns [424627, 426552] (0.95 confidence)</td>
</tr>
<tr>
<td>random/insphere&#95;distance&#95;1000&#95;queries</td>
<td>63732.1 ns [63100.7, 64093.6] (0.95 confidence)</td>
<td>65710 ns [65364.6, 66013.6] (0.95 confidence)</td>
</tr>
<tr>
<td>random/insphere&#95;lifted&#95;1000&#95;queries</td>
<td>415647 ns [413885, 416441] (0.95 confidence)</td>
<td>420119 ns [419303, 421430] (0.95 confidence)</td>
</tr>
<tr>
<td>tds&#95;new&#95;2d/tds&#95;new/500</td>
<td>4.57108e+07 ns [4.56421e+07, 4.60575e+07] (0.95 confidence)</td>
<td>4.57306e+07 ns [4.56665e+07, 4.59635e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>tds&#95;new&#95;2d/tds&#95;new&#95;adversarial/500</td>
<td>5.36238e+07 ns [5.35895e+07, 5.37677e+07] (0.95 confidence)</td>
<td>5.48306e+07 ns [5.41766e+07, 5.59895e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>tds&#95;new&#95;3d/tds&#95;new/100</td>
<td>3.6891e+07 ns [3.68746e+07, 3.69504e+07] (0.95 confidence)</td>
<td>3.73342e+07 ns [3.72714e+07, 3.74754e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>tds&#95;new&#95;3d/tds&#95;new&#95;adversarial/100</td>
<td>8.36588e+07 ns [8.34731e+07, 8.38007e+07] (0.95 confidence)</td>
<td>8.50726e+07 ns [8.48532e+07, 8.53613e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>tds&#95;new&#95;4d/tds&#95;new/30</td>
<td>9.29788e+07 ns [9.24575e+07, 9.30817e+07] (0.95 confidence)</td>
<td>9.68993e+07 ns [9.63217e+07, 9.74279e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>tds&#95;new&#95;4d/tds&#95;new&#95;adversarial/30</td>
<td>8.7584e+07 ns [8.74508e+07, 8.79803e+07] (0.95 confidence)</td>
<td>9.31245e+07 ns [9.22912e+07, 9.43113e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>tds&#95;new&#95;5d/tds&#95;new/12</td>
<td>2.03154e+07 ns [2.02609e+07, 2.03676e+07] (0.95 confidence)</td>
<td>2.20554e+07 ns [2.17983e+07, 2.21252e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>tds&#95;new&#95;5d/tds&#95;new&#95;adversarial/12</td>
<td>1.87543e+07 ns [1.87061e+07, 1.8823e+07] (0.95 confidence)</td>
<td>2.01585e+07 ns [2.00459e+07, 2.02357e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;2d/500</td>
<td>799819 ns [795378, 805903] (0.95 confidence)</td>
<td>798370 ns [793957, 809065] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;2d&#95;adversarial/500</td>
<td>1.1116e+06 ns [1.10194e+06, 1.12261e+06] (0.95 confidence)</td>
<td>1.09745e+06 ns [1.09278e+06, 1.10645e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;3d/100</td>
<td>1.08627e+06 ns [1.08194e+06, 1.10765e+06] (0.95 confidence)</td>
<td>1.24811e+06 ns [1.24347e+06, 1.27866e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;3d&#95;adversarial/100</td>
<td>2.06373e+06 ns [2.0512e+06, 2.07848e+06] (0.95 confidence)</td>
<td>2.30001e+06 ns [2.28126e+06, 2.31235e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;4d/30</td>
<td>3.90205e+06 ns [3.87952e+06, 3.93979e+06] (0.95 confidence)</td>
<td>4.63703e+06 ns [4.61541e+06, 4.69135e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;4d&#95;adversarial/30</td>
<td>3.89595e+06 ns [3.87945e+06, 3.93443e+06] (0.95 confidence)</td>
<td>4.48868e+06 ns [4.47928e+06, 4.62317e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;5d/12</td>
<td>1.63808e+06 ns [1.63165e+06, 1.64264e+06] (0.95 confidence)</td>
<td>2.22542e+06 ns [2.20907e+06, 2.2319e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;5d&#95;adversarial/12</td>
<td>1.45267e+06 ns [1.44949e+06, 1.45832e+06] (0.95 confidence)</td>
<td>2.01777e+06 ns [1.98922e+06, 2.05224e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;2d/500</td>
<td>806365 ns [804279, 819642] (0.95 confidence)</td>
<td>805717 ns [803169, 812790] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;2d&#95;adversarial/500</td>
<td>1.1068e+06 ns [1.10044e+06, 1.11426e+06] (0.95 confidence)</td>
<td>1.10322e+06 ns [1.09618e+06, 1.12075e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;3d/100</td>
<td>1.08289e+06 ns [1.07588e+06, 1.08437e+06] (0.95 confidence)</td>
<td>1.24881e+06 ns [1.2427e+06, 1.28673e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;3d&#95;adversarial/100</td>
<td>2.08592e+06 ns [2.08096e+06, 2.09407e+06] (0.95 confidence)</td>
<td>2.2076e+06 ns [2.20215e+06, 2.21738e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;4d/30</td>
<td>3.89641e+06 ns [3.89275e+06, 3.91188e+06] (0.95 confidence)</td>
<td>4.56328e+06 ns [4.54607e+06, 4.63134e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;4d&#95;adversarial/30</td>
<td>3.89025e+06 ns [3.84674e+06, 3.9132e+06] (0.95 confidence)</td>
<td>4.48366e+06 ns [4.46312e+06, 4.5549e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;5d/12</td>
<td>1.62775e+06 ns [1.62652e+06, 1.63767e+06] (0.95 confidence)</td>
<td>2.18356e+06 ns [2.17152e+06, 2.22679e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;5d&#95;adversarial/12</td>
<td>1.45753e+06 ns [1.44897e+06, 1.47137e+06] (0.95 confidence)</td>
<td>2.05986e+06 ns [2.03154e+06, 2.07317e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/validate&#95;2d/500</td>
<td>8.08093e+06 ns [8.05422e+06, 8.13458e+06] (0.95 confidence)</td>
<td>8.25328e+06 ns [8.23195e+06, 8.27608e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/validate&#95;2d&#95;adversarial/500</td>
<td>5.32499e+07 ns [5.31235e+07, 5.33963e+07] (0.95 confidence)</td>
<td>4.89219e+07 ns [4.87912e+07, 4.92393e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/validate&#95;3d/100</td>
<td>8.32626e+08 ns [8.31399e+08, 8.33958e+08] (0.95 confidence)</td>
<td>8.47454e+08 ns [8.43921e+08, 8.48996e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/validate&#95;3d&#95;adversarial/100</td>
<td>1.03543e+09 ns [1.03458e+09, 1.03684e+09] (0.95 confidence)</td>
<td>1.02596e+09 ns [1.02339e+09, 1.03239e+09] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/validate&#95;4d/30</td>
<td>3.89185e+09 ns [3.88675e+09, 3.90015e+09] (0.95 confidence)</td>
<td>3.88661e+09 ns [3.87856e+09, 3.8948e+09] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/validate&#95;4d&#95;adversarial/30</td>
<td>4.01774e+09 ns [4.01138e+09, 4.02486e+09] (0.95 confidence)</td>
<td>3.96141e+09 ns [3.9566e+09, 3.96395e+09] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/validate&#95;5d/12</td>
<td>8.67833e+08 ns [8.66515e+08, 8.69927e+08] (0.95 confidence)</td>
<td>8.90769e+08 ns [8.87632e+08, 8.92131e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/validate&#95;5d&#95;adversarial/12</td>
<td>8.11157e+08 ns [8.10397e+08, 8.13022e+08] (0.95 confidence)</td>
<td>8.271e+08 ns [8.25242e+08, 8.29735e+08] (0.95 confidence)</td>
</tr>
</tbody>
</table>

Harness mode: common.

Ratio eligibility: compatible recorded identities

## Complete phase inventory

<table>
<thead>
<tr>
<th>Benchmark</th>
<th>Baseline</th>
<th>Current</th>
</tr>
</thead>
<tbody>
<tr>
<td>2d/insphere</td>
<td>50.3629 ns [50.1007, 50.632] (0.95 confidence)</td>
<td>49.9714 ns [49.7588, 50.1938] (0.95 confidence)</td>
</tr>
<tr>
<td>2d/insphere&#95;distance</td>
<td>13.4344 ns [13.4123, 13.4565] (0.95 confidence)</td>
<td>13.7432 ns [13.7119, 13.7772] (0.95 confidence)</td>
</tr>
<tr>
<td>2d/insphere&#95;lifted</td>
<td>49.3667 ns [49.1805, 49.5559] (0.95 confidence)</td>
<td>52.2499 ns [51.8981, 52.6023] (0.95 confidence)</td>
</tr>
<tr>
<td>3d/insphere</td>
<td>107.807 ns [107.127, 108.55] (0.95 confidence)</td>
<td>111.598 ns [110.877, 112.327] (0.95 confidence)</td>
</tr>
<tr>
<td>3d/insphere&#95;distance</td>
<td>32.5374 ns [32.4341, 32.6407] (0.95 confidence)</td>
<td>33.2631 ns [33.1675, 33.3635] (0.95 confidence)</td>
</tr>
<tr>
<td>3d/insphere&#95;lifted</td>
<td>105.5 ns [105.122, 105.901] (0.95 confidence)</td>
<td>107.329 ns [106.914, 107.781] (0.95 confidence)</td>
</tr>
<tr>
<td>4d/insphere</td>
<td>749.057 ns [746.628, 751.571] (0.95 confidence)</td>
<td>759.7 ns [758.152, 761.301] (0.95 confidence)</td>
</tr>
<tr>
<td>4d/insphere&#95;distance</td>
<td>50.8807 ns [50.7535, 51.0078] (0.95 confidence)</td>
<td>54.4821 ns [54.4075, 54.5543] (0.95 confidence)</td>
</tr>
<tr>
<td>4d/insphere&#95;lifted</td>
<td>752.908 ns [749.769, 756.022] (0.95 confidence)</td>
<td>764.499 ns [760.881, 768.229] (0.95 confidence)</td>
</tr>
<tr>
<td>5d/insphere</td>
<td>1387.9 ns [1383.98, 1391.9] (0.95 confidence)</td>
<td>1287.3 ns [1284.02, 1290.71] (0.95 confidence)</td>
</tr>
<tr>
<td>5d/insphere&#95;distance</td>
<td>87.4293 ns [87.1704, 87.6922] (0.95 confidence)</td>
<td>87.0696 ns [86.8265, 87.3225] (0.95 confidence)</td>
</tr>
<tr>
<td>5d/insphere&#95;lifted</td>
<td>1374.05 ns [1370.15, 1377.96] (0.95 confidence)</td>
<td>1300.07 ns [1295.7, 1304.4] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k1&#95;roundtrip</td>
<td>53991 ns [53907.8, 54086.5] (0.95 confidence)</td>
<td>56295.4 ns [56144.4, 56459.1] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k1&#95;roundtrip&#95;adversarial</td>
<td>162550 ns [162378, 162722] (0.95 confidence)</td>
<td>166391 ns [166034, 166747] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k2&#95;edge&#95;flip</td>
<td>26258.3 ns [26231.1, 26287.7] (0.95 confidence)</td>
<td>27593.7 ns [27533.9, 27667.4] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;2d/k2&#95;edge&#95;flip&#95;adversarial</td>
<td>101699 ns [99923.2, 103885] (0.95 confidence)</td>
<td>102942 ns [102646, 103239] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k1&#95;roundtrip</td>
<td>2.3395e+06 ns [2.33445e+06, 2.34426e+06] (0.95 confidence)</td>
<td>2.41399e+06 ns [2.4096e+06, 2.41826e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k1&#95;roundtrip&#95;adversarial</td>
<td>1.95481e+06 ns [1.94607e+06, 1.9653e+06] (0.95 confidence)</td>
<td>2.00716e+06 ns [2.00541e+06, 2.00886e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k2&#95;roundtrip</td>
<td>6.58713e+06 ns [6.47698e+06, 6.71913e+06] (0.95 confidence)</td>
<td>6.63185e+06 ns [6.62046e+06, 6.64449e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k2&#95;roundtrip&#95;adversarial</td>
<td>5.82996e+06 ns [5.81511e+06, 5.84711e+06] (0.95 confidence)</td>
<td>6.02659e+06 ns [6.01725e+06, 6.03505e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k3&#95;forward</td>
<td>1.35985e+06 ns [1.35702e+06, 1.36271e+06] (0.95 confidence)</td>
<td>1.39557e+06 ns [1.39295e+06, 1.39875e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;3d/k3&#95;forward&#95;adversarial</td>
<td>1.88579e+06 ns [1.87877e+06, 1.89349e+06] (0.95 confidence)</td>
<td>1.87074e+06 ns [1.8671e+06, 1.8745e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k1&#95;roundtrip</td>
<td>3.91942e+07 ns [3.89438e+07, 3.94883e+07] (0.95 confidence)</td>
<td>3.9626e+07 ns [3.94912e+07, 3.98113e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k1&#95;roundtrip&#95;adversarial</td>
<td>3.31749e+07 ns [3.30477e+07, 3.33109e+07] (0.95 confidence)</td>
<td>3.33839e+07 ns [3.33368e+07, 3.34367e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k2&#95;roundtrip</td>
<td>5.61884e+07 ns [5.59148e+07, 5.64898e+07] (0.95 confidence)</td>
<td>5.63516e+07 ns [5.62749e+07, 5.64325e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k2&#95;roundtrip&#95;adversarial</td>
<td>2.00254e+08 ns [1.98791e+08, 2.01988e+08] (0.95 confidence)</td>
<td>1.98677e+08 ns [1.98407e+08, 1.98921e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k3&#95;roundtrip</td>
<td>6.55919e+07 ns [6.51718e+07, 6.6026e+07] (0.95 confidence)</td>
<td>6.50869e+07 ns [6.49496e+07, 6.52353e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;4d/k3&#95;roundtrip&#95;adversarial</td>
<td>3.52388e+07 ns [3.5102e+07, 3.53859e+07] (0.95 confidence)</td>
<td>3.51438e+07 ns [3.50984e+07, 3.51917e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k1&#95;roundtrip</td>
<td>1.55988e+08 ns [1.55554e+08, 1.56426e+08] (0.95 confidence)</td>
<td>1.59178e+08 ns [1.58857e+08, 1.5952e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k1&#95;roundtrip&#95;adversarial</td>
<td>1.60754e+08 ns [1.59915e+08, 1.61653e+08] (0.95 confidence)</td>
<td>1.62011e+08 ns [1.61557e+08, 1.62463e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k2&#95;roundtrip</td>
<td>329220 ns [328636, 329841] (0.95 confidence)</td>
<td>329108 ns [328883, 329324] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k2&#95;roundtrip&#95;adversarial</td>
<td>352172 ns [351379, 352996] (0.95 confidence)</td>
<td>356146 ns [355584, 356751] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k3&#95;roundtrip</td>
<td>3.26703e+08 ns [3.25532e+08, 3.2809e+08] (0.95 confidence)</td>
<td>3.30575e+08 ns [3.30234e+08, 3.30918e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>bistellar&#95;flips&#95;5d/k3&#95;roundtrip&#95;adversarial</td>
<td>9.10658e+08 ns [9.07812e+08, 9.13841e+08] (0.95 confidence)</td>
<td>8.82636e+08 ns [8.79657e+08, 8.86312e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;2d/500</td>
<td>209632 ns [209098, 210187] (0.95 confidence)</td>
<td>211250 ns [210527, 212025] (0.95 confidence)</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;2d&#95;adversarial/500</td>
<td>209507 ns [209292, 209744] (0.95 confidence)</td>
<td>210174 ns [209394, 210922] (0.95 confidence)</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;3d/100</td>
<td>166339 ns [166016, 166741] (0.95 confidence)</td>
<td>167056 ns [166588, 167540] (0.95 confidence)</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;3d&#95;adversarial/100</td>
<td>168132 ns [167863, 168417] (0.95 confidence)</td>
<td>168413 ns [168092, 168732] (0.95 confidence)</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;4d/30</td>
<td>131622 ns [131385, 131915] (0.95 confidence)</td>
<td>137706 ns [137303, 138107] (0.95 confidence)</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;4d&#95;adversarial/30</td>
<td>131050 ns [130824, 131290] (0.95 confidence)</td>
<td>134747 ns [134256, 135263] (0.95 confidence)</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;5d/12</td>
<td>36669.9 ns [36519.8, 36840.6] (0.95 confidence)</td>
<td>37385.2 ns [37266, 37504.3] (0.95 confidence)</td>
</tr>
<tr>
<td>boundary&#95;facets/boundary&#95;facets&#95;5d&#95;adversarial/12</td>
<td>35788.4 ns [35725.1, 35858.7] (0.95 confidence)</td>
<td>36980.9 ns [36801.1, 37160.8] (0.95 confidence)</td>
</tr>
<tr>
<td>circumcenter/solve&#95;path/near&#95;singular&#95;exact&#95;fallback&#95;3d</td>
<td>17669.4 ns [17626.9, 17711.7] (0.95 confidence)</td>
<td>17606.8 ns [17584.7, 17628.7] (0.95 confidence)</td>
</tr>
<tr>
<td>circumcenter/solve&#95;path/regular&#95;lu&#95;3d</td>
<td>22.9912 ns [22.9638, 23.0195] (0.95 confidence)</td>
<td>23.7471 ns [23.7008, 23.7993] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;2d/500</td>
<td>343199 ns [341356, 345118] (0.95 confidence)</td>
<td>391508 ns [390646, 392422] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;2d&#95;adversarial/500</td>
<td>332771 ns [331997, 334037] (0.95 confidence)</td>
<td>389851 ns [388690, 391118] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;3d/100</td>
<td>275770 ns [275370, 276220] (0.95 confidence)</td>
<td>277801 ns [276837, 278751] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;3d&#95;adversarial/100</td>
<td>4.00082e+06 ns [3.9926e+06, 4.00978e+06] (0.95 confidence)</td>
<td>4.03635e+06 ns [4.02751e+06, 4.04514e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;4d/30</td>
<td>8.78704e+06 ns [8.73664e+06, 8.84747e+06] (0.95 confidence)</td>
<td>8.9468e+06 ns [8.92504e+06, 8.96889e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;4d&#95;adversarial/30</td>
<td>6.99418e+06 ns [6.95724e+06, 7.03573e+06] (0.95 confidence)</td>
<td>7.07419e+06 ns [7.04404e+06, 7.11162e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;5d/12</td>
<td>3.07103e+06 ns [3.06251e+06, 3.0796e+06] (0.95 confidence)</td>
<td>3.07207e+06 ns [3.05725e+06, 3.09035e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull/from&#95;triangulation&#95;5d&#95;adversarial/12</td>
<td>2.63914e+06 ns [2.63246e+06, 2.64712e+06] (0.95 confidence)</td>
<td>2.73111e+06 ns [2.71267e+06, 2.74983e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;nearest&#95;visible&#95;facet&#95;3d/100</td>
<td>1134.72 ns [1130.38, 1138.79] (0.95 confidence)</td>
<td>1149.57 ns [1144.79, 1154.44] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;nearest&#95;visible&#95;facet&#95;3d&#95;adversarial/100</td>
<td>40912.5 ns [40804.1, 41034.4] (0.95 confidence)</td>
<td>41004.1 ns [40895.6, 41133.8] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;visible&#95;facets&#95;3d/100</td>
<td>1212.91 ns [1201.63, 1227.13] (0.95 confidence)</td>
<td>1222.44 ns [1216.82, 1228.52] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/find&#95;visible&#95;facets&#95;3d&#95;adversarial/100</td>
<td>41046.2 ns [40913.5, 41199.7] (0.95 confidence)</td>
<td>41131.9 ns [41039.7, 41228] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/is&#95;point&#95;outside&#95;3d/100</td>
<td>94.6067 ns [94.4914, 94.7352] (0.95 confidence)</td>
<td>95.7992 ns [95.6301, 95.9675] (0.95 confidence)</td>
</tr>
<tr>
<td>convex&#95;hull&#95;queries/is&#95;point&#95;outside&#95;3d&#95;adversarial/100</td>
<td>770.856 ns [768.53, 773.297] (0.95 confidence)</td>
<td>780.13 ns [776.819, 783.986] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/boundary&#95;point&#95;distance</td>
<td>13.5388 ns [13.514, 13.5649] (0.95 confidence)</td>
<td>13.4266 ns [13.3943, 13.4638] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/boundary&#95;point&#95;insphere</td>
<td>0.83637 ns [0.833595, 0.8392] (0.95 confidence)</td>
<td>0.827141 ns [0.824016, 0.830381] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/boundary&#95;point&#95;lifted</td>
<td>971.001 ns [969.089, 972.955] (0.95 confidence)</td>
<td>978.53 ns [976.62, 980.488] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/far&#95;point&#95;distance</td>
<td>13.5132 ns [13.4817, 13.5458] (0.95 confidence)</td>
<td>13.6775 ns [13.6539, 13.7016] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/far&#95;point&#95;insphere</td>
<td>48.9921 ns [48.8046, 49.1883] (0.95 confidence)</td>
<td>50.7656 ns [50.4847, 51.0571] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/far&#95;point&#95;lifted</td>
<td>50.5973 ns [50.2123, 51.0038] (0.95 confidence)</td>
<td>50.5494 ns [50.2123, 50.8927] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/near&#95;boundary&#95;distance</td>
<td>13.8014 ns [13.757, 13.8495] (0.95 confidence)</td>
<td>13.8464 ns [13.8242, 13.868] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/near&#95;boundary&#95;insphere</td>
<td>47.0807 ns [46.8132, 47.361] (0.95 confidence)</td>
<td>48.0824 ns [47.7365, 48.4402] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;2d/near&#95;boundary&#95;lifted</td>
<td>47.3911 ns [47.0183, 47.7893] (0.95 confidence)</td>
<td>47.5431 ns [47.261, 47.852] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/boundary&#95;point&#95;distance</td>
<td>32.7541 ns [32.6327, 32.8795] (0.95 confidence)</td>
<td>32.8153 ns [32.7445, 32.8948] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/boundary&#95;point&#95;insphere</td>
<td>1.21494 ns [1.20195, 1.23087] (0.95 confidence)</td>
<td>1.18188 ns [1.17849, 1.18493] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/boundary&#95;point&#95;lifted</td>
<td>1744.54 ns [1742.43, 1746.66] (0.95 confidence)</td>
<td>1776.98 ns [1775.41, 1778.57] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/far&#95;point&#95;distance</td>
<td>32.9828 ns [32.84, 33.1373] (0.95 confidence)</td>
<td>33.1066 ns [33.0361, 33.1795] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/far&#95;point&#95;insphere</td>
<td>108.102 ns [107.54, 108.695] (0.95 confidence)</td>
<td>108.558 ns [107.966, 109.172] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/far&#95;point&#95;lifted</td>
<td>111.197 ns [110.58, 111.827] (0.95 confidence)</td>
<td>112.344 ns [111.798, 112.911] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/near&#95;boundary&#95;distance</td>
<td>32.6832 ns [32.5787, 32.7926] (0.95 confidence)</td>
<td>32.7475 ns [32.6955, 32.803] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/near&#95;boundary&#95;insphere</td>
<td>94.5311 ns [94.2925, 94.7792] (0.95 confidence)</td>
<td>92.2219 ns [91.8899, 92.5626] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;3d/near&#95;boundary&#95;lifted</td>
<td>95.1302 ns [94.7213, 95.565] (0.95 confidence)</td>
<td>91.2777 ns [91.0512, 91.5091] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/boundary&#95;point&#95;distance</td>
<td>50.9981 ns [50.888, 51.1097] (0.95 confidence)</td>
<td>54.6747 ns [54.5304, 54.8235] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/boundary&#95;point&#95;insphere</td>
<td>1.57133 ns [1.54916, 1.59235] (0.95 confidence)</td>
<td>1.56298 ns [1.53286, 1.59204] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/boundary&#95;point&#95;lifted</td>
<td>3668.74 ns [3660.8, 3677.01] (0.95 confidence)</td>
<td>3767.89 ns [3749.64, 3792.2] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/far&#95;point&#95;distance</td>
<td>50.9772 ns [50.8599, 51.0953] (0.95 confidence)</td>
<td>53.7173 ns [53.5935, 53.8393] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/far&#95;point&#95;insphere</td>
<td>749.122 ns [743.645, 755.32] (0.95 confidence)</td>
<td>736.251 ns [734.968, 737.559] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/far&#95;point&#95;lifted</td>
<td>753 ns [750.441, 755.613] (0.95 confidence)</td>
<td>738.213 ns [735.498, 741.145] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/near&#95;boundary&#95;distance</td>
<td>53.1202 ns [52.8804, 53.372] (0.95 confidence)</td>
<td>53.8261 ns [53.7072, 53.945] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/near&#95;boundary&#95;insphere</td>
<td>766.167 ns [762.646, 769.643] (0.95 confidence)</td>
<td>738.957 ns [736.84, 741.114] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;4d/near&#95;boundary&#95;lifted</td>
<td>740.171 ns [737.355, 743.093] (0.95 confidence)</td>
<td>733.476 ns [731.047, 735.986] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/boundary&#95;point&#95;distance</td>
<td>87.4714 ns [87.2609, 87.6868] (0.95 confidence)</td>
<td>87.3212 ns [87.0458, 87.5977] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/boundary&#95;point&#95;insphere</td>
<td>1.69288 ns [1.68869, 1.69786] (0.95 confidence)</td>
<td>1.70035 ns [1.69221, 1.71066] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/boundary&#95;point&#95;lifted</td>
<td>6047.39 ns [6008.53, 6095.32] (0.95 confidence)</td>
<td>5867.63 ns [5857.43, 5878.07] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/far&#95;point&#95;distance</td>
<td>88.2959 ns [87.8611, 88.8142] (0.95 confidence)</td>
<td>87.2551 ns [86.9283, 87.586] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/far&#95;point&#95;insphere</td>
<td>1364.07 ns [1359.99, 1368.36] (0.95 confidence)</td>
<td>1299.78 ns [1296.04, 1303.57] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/far&#95;point&#95;lifted</td>
<td>1343.25 ns [1340.42, 1346.12] (0.95 confidence)</td>
<td>1286.39 ns [1283.31, 1289.59] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/near&#95;boundary&#95;distance</td>
<td>88.5645 ns [88.23, 88.918] (0.95 confidence)</td>
<td>87.9891 ns [87.6638, 88.3188] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/near&#95;boundary&#95;insphere</td>
<td>1342.36 ns [1337.99, 1347.03] (0.95 confidence)</td>
<td>1264.07 ns [1259.86, 1268.42] (0.95 confidence)</td>
</tr>
<tr>
<td>edge&#95;cases&#95;5d/near&#95;boundary&#95;lifted</td>
<td>1343.51 ns [1339.28, 1347.75] (0.95 confidence)</td>
<td>1276.64 ns [1271.19, 1282.32] (0.95 confidence)</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;2d/vertices&#95;120&#95;simplices&#95;226</td>
<td>1.61314e+06 ns [1.60877e+06, 1.61763e+06] (0.95 confidence)</td>
<td>1.58605e+06 ns [1.58205e+06, 1.59041e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;3d/vertices&#95;30&#95;simplices&#95;115</td>
<td>1.59381e+08 ns [1.59147e+08, 1.59635e+08] (0.95 confidence)</td>
<td>1.58796e+08 ns [1.58487e+08, 1.59104e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;4d/vertices&#95;16&#95;simplices&#95;68</td>
<td>4.79355e+08 ns [4.79046e+08, 4.79656e+08] (0.95 confidence)</td>
<td>4.71827e+08 ns [4.71496e+08, 4.7218e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>explicit&#95;import/import&#95;pseudomanifold&#95;5d/vertices&#95;10&#95;simplices&#95;20</td>
<td>1.35104e+08 ns [1.34874e+08, 1.35354e+08] (0.95 confidence)</td>
<td>1.3601e+08 ns [1.35542e+08, 1.36586e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;2d/10</td>
<td>438463 ns [437691, 439250] (0.95 confidence)</td>
<td>446524 ns [445029, 448210] (0.95 confidence)</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;2d&#95;adversarial/10</td>
<td>2.86142e+06 ns [2.85555e+06, 2.86739e+06] (0.95 confidence)</td>
<td>2.88115e+06 ns [2.86639e+06, 2.90276e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;3d/10</td>
<td>2.81502e+06 ns [2.79423e+06, 2.83573e+06] (0.95 confidence)</td>
<td>2.78132e+06 ns [2.76522e+06, 2.79975e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;3d&#95;adversarial/10</td>
<td>1.39265e+07 ns [1.38698e+07, 1.39934e+07] (0.95 confidence)</td>
<td>1.35294e+07 ns [1.34597e+07, 1.36191e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;4d/6</td>
<td>2.65165e+07 ns [2.64711e+07, 2.65605e+07] (0.95 confidence)</td>
<td>2.66044e+07 ns [2.64984e+07, 2.67056e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;4d&#95;adversarial/6</td>
<td>6.57378e+07 ns [6.55044e+07, 6.59952e+07] (0.95 confidence)</td>
<td>6.71733e+07 ns [6.68671e+07, 6.75122e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;5d/4</td>
<td>4.79581e+07 ns [4.79015e+07, 4.80238e+07] (0.95 confidence)</td>
<td>5.156e+07 ns [5.13765e+07, 5.17438e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>incremental&#95;insert/insert&#95;5d&#95;adversarial/4</td>
<td>5.18702e+07 ns [5.16746e+07, 5.20777e+07] (0.95 confidence)</td>
<td>5.59724e+07 ns [5.58076e+07, 5.61503e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/2d/vertices&#95;100&#95;simplices&#95;187&#95;seed&#95;dd618c77b9187836&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>7081.94 ns [7068.22, 7095.66] (0.95 confidence)</td>
<td>7239.28 ns [7216.59, 7272.07] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/2d/vertices&#95;2000&#95;simplices&#95;3978&#95;seed&#95;01acf93075e96412&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>7274.62 ns [7263.47, 7290.86] (0.95 confidence)</td>
<td>7512.78 ns [7490.83, 7537.37] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/2d/vertices&#95;500&#95;simplices&#95;983&#95;seed&#95;1493be4e9d7a5906&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>7398.82 ns [7385.78, 7411.32] (0.95 confidence)</td>
<td>7710.9 ns [7650.55, 7799.29] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/3d/vertices&#95;150&#95;simplices&#95;829&#95;seed&#95;a44b52b395a4b44d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>14620.8 ns [14552.5, 14700.4] (0.95 confidence)</td>
<td>15143.6 ns [15082.8, 15206.8] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/3d/vertices&#95;500&#95;simplices&#95;3052&#95;seed&#95;1493be4f9d7a5907&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>14483 ns [14428.5, 14533] (0.95 confidence)</td>
<td>15071.4 ns [15023.9, 15117.8] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/3d/vertices&#95;50&#95;simplices&#95;222&#95;seed&#95;f61fc639dc8c3c19&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>14566.2 ns [14526.3, 14605.3] (0.95 confidence)</td>
<td>15218.3 ns [15163, 15277.9] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/4d/vertices&#95;100&#95;simplices&#95;1677&#95;seed&#95;dd618c71b9187830&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>1.98439e+06 ns [1.98071e+06, 1.98819e+06] (0.95 confidence)</td>
<td>2.03576e+06 ns [2.0319e+06, 2.03999e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/4d/vertices&#95;50&#95;simplices&#95;626&#95;seed&#95;f61fc63edc8c3c1e&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>1.96e+06 ns [1.94697e+06, 1.97463e+06] (0.95 confidence)</td>
<td>2.02676e+06 ns [2.00081e+06, 2.05853e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/5d/vertices&#95;25&#95;simplices&#95;410&#95;seed&#95;63a0e3186e461e08&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>4.77285e+06 ns [4.74467e+06, 4.79882e+06] (0.95 confidence)</td>
<td>4.82906e+06 ns [4.80623e+06, 4.85173e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/exact&#95;hint/5d/vertices&#95;40&#95;simplices&#95;1208&#95;seed&#95;a86104fee3a3634d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>4.85416e+06 ns [4.83084e+06, 4.87952e+06] (0.95 confidence)</td>
<td>4.87375e+06 ns [4.85958e+06, 4.88793e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/2d/vertices&#95;100&#95;simplices&#95;187&#95;seed&#95;dd618c77b9187836&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>73882.7 ns [73699.6, 74068.9] (0.95 confidence)</td>
<td>76547.4 ns [75702, 77660.7] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/2d/vertices&#95;2000&#95;simplices&#95;3978&#95;seed&#95;01acf93075e96412&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>268491 ns [266780, 270672] (0.95 confidence)</td>
<td>276458 ns [274145, 279685] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/2d/vertices&#95;500&#95;simplices&#95;983&#95;seed&#95;1493be4e9d7a5906&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>123961 ns [123167, 124787] (0.95 confidence)</td>
<td>125920 ns [125494, 126310] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/3d/vertices&#95;150&#95;simplices&#95;829&#95;seed&#95;a44b52b395a4b44d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>124045 ns [123827, 124332] (0.95 confidence)</td>
<td>129863 ns [128882, 130850] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/3d/vertices&#95;500&#95;simplices&#95;3052&#95;seed&#95;1493be4f9d7a5907&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>191201 ns [189411, 193110] (0.95 confidence)</td>
<td>201218 ns [197993, 205126] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/3d/vertices&#95;50&#95;simplices&#95;222&#95;seed&#95;f61fc639dc8c3c19&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>64983.6 ns [64694.3, 65309.3] (0.95 confidence)</td>
<td>69547.6 ns [68299.2, 71462.6] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/4d/vertices&#95;100&#95;simplices&#95;1677&#95;seed&#95;dd618c71b9187830&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>1.12189e+07 ns [1.12025e+07, 1.12351e+07] (0.95 confidence)</td>
<td>1.16001e+07 ns [1.1538e+07, 1.16563e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/4d/vertices&#95;50&#95;simplices&#95;626&#95;seed&#95;f61fc63edc8c3c1e&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>1.12709e+07 ns [1.11852e+07, 1.13586e+07] (0.95 confidence)</td>
<td>1.14055e+07 ns [1.1377e+07, 1.14328e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/5d/vertices&#95;25&#95;simplices&#95;410&#95;seed&#95;63a0e3186e461e08&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>2.19654e+07 ns [2.19286e+07, 2.20029e+07] (0.95 confidence)</td>
<td>2.26343e+07 ns [2.25583e+07, 2.27176e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>locate/no&#95;hint/5d/vertices&#95;40&#95;simplices&#95;1208&#95;seed&#95;a86104fee3a3634d&#95;input&#95;hull&#95;queries&#95;v1&#95;64</td>
<td>4.00702e+07 ns [3.99473e+07, 4.02048e+07] (0.95 confidence)</td>
<td>4.13841e+07 ns [4.10856e+07, 4.17361e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;2d/simplices&#95;226</td>
<td>115003 ns [113590, 116852] (0.95 confidence)</td>
<td>118699 ns [118022, 119742] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;3d/simplices&#95;115</td>
<td>141173 ns [140126, 142660] (0.95 confidence)</td>
<td>155207 ns [153563, 156859] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;4d/simplices&#95;68</td>
<td>1.20253e+06 ns [1.19344e+06, 1.21485e+06] (0.95 confidence)</td>
<td>1.27174e+06 ns [1.26183e+06, 1.28122e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;5d/simplices&#95;20</td>
<td>455130 ns [452645, 458226] (0.95 confidence)</td>
<td>483222 ns [480869, 486056] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;2d/simplices&#95;226</td>
<td>1.32186e+06 ns [1.31451e+06, 1.3356e+06] (0.95 confidence)</td>
<td>1.32725e+06 ns [1.32312e+06, 1.33212e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;3d/simplices&#95;115</td>
<td>1.549e+08 ns [1.54525e+08, 1.55339e+08] (0.95 confidence)</td>
<td>1.59589e+08 ns [1.592e+08, 1.59906e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;4d/simplices&#95;68</td>
<td>4.6906e+08 ns [4.6741e+08, 4.71092e+08] (0.95 confidence)</td>
<td>4.71969e+08 ns [4.71084e+08, 4.72921e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;5d/simplices&#95;20</td>
<td>1.33311e+08 ns [1.33108e+08, 1.33521e+08] (0.95 confidence)</td>
<td>1.38963e+08 ns [1.36488e+08, 1.4229e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;2d/simplices&#95;226</td>
<td>1.23762e+06 ns [1.23506e+06, 1.24136e+06] (0.95 confidence)</td>
<td>1.25076e+06 ns [1.24641e+06, 1.25557e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;3d/simplices&#95;115</td>
<td>1.54604e+08 ns [1.54401e+08, 1.54873e+08] (0.95 confidence)</td>
<td>1.59552e+08 ns [1.5926e+08, 1.59807e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;4d/simplices&#95;68</td>
<td>4.6949e+08 ns [4.68395e+08, 4.71262e+08] (0.95 confidence)</td>
<td>4.72816e+08 ns [4.71484e+08, 4.74132e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;5d/simplices&#95;20</td>
<td>1.32966e+08 ns [1.32768e+08, 1.33227e+08] (0.95 confidence)</td>
<td>1.36289e+08 ns [1.3595e+08, 1.36622e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>random/insphere&#95;1000&#95;queries</td>
<td>414983 ns [413766, 416195] (0.95 confidence)</td>
<td>425759 ns [424725, 426748] (0.95 confidence)</td>
</tr>
<tr>
<td>random/insphere&#95;distance&#95;1000&#95;queries</td>
<td>63190.2 ns [62875, 63503.8] (0.95 confidence)</td>
<td>65623.2 ns [65453.4, 65795] (0.95 confidence)</td>
</tr>
<tr>
<td>random/insphere&#95;lifted&#95;1000&#95;queries</td>
<td>415100 ns [413851, 416329] (0.95 confidence)</td>
<td>421148 ns [419926, 422384] (0.95 confidence)</td>
</tr>
<tr>
<td>tds&#95;new&#95;2d/tds&#95;new/500</td>
<td>4.58298e+07 ns [4.56359e+07, 4.60524e+07] (0.95 confidence)</td>
<td>4.58099e+07 ns [4.57014e+07, 4.59412e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>tds&#95;new&#95;2d/tds&#95;new&#95;adversarial/500</td>
<td>5.36853e+07 ns [5.36105e+07, 5.37802e+07] (0.95 confidence)</td>
<td>5.52124e+07 ns [5.45396e+07, 5.60237e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>tds&#95;new&#95;3d/tds&#95;new/100</td>
<td>3.69066e+07 ns [3.68702e+07, 3.69438e+07] (0.95 confidence)</td>
<td>3.75031e+07 ns [3.72991e+07, 3.78407e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>tds&#95;new&#95;3d/tds&#95;new&#95;adversarial/100</td>
<td>8.36544e+07 ns [8.35455e+07, 8.37654e+07] (0.95 confidence)</td>
<td>8.5111e+07 ns [8.49203e+07, 8.53172e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>tds&#95;new&#95;4d/tds&#95;new/30</td>
<td>9.27933e+07 ns [9.25165e+07, 9.30309e+07] (0.95 confidence)</td>
<td>9.69391e+07 ns [9.64397e+07, 9.7463e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>tds&#95;new&#95;4d/tds&#95;new&#95;adversarial/30</td>
<td>8.76889e+07 ns [8.74524e+07, 8.79472e+07] (0.95 confidence)</td>
<td>9.32569e+07 ns [9.26047e+07, 9.39296e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>tds&#95;new&#95;5d/tds&#95;new/12</td>
<td>2.03289e+07 ns [2.02847e+07, 2.03842e+07] (0.95 confidence)</td>
<td>2.20169e+07 ns [2.18847e+07, 2.21549e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>tds&#95;new&#95;5d/tds&#95;new&#95;adversarial/12</td>
<td>1.8768e+07 ns [1.8727e+07, 1.88139e+07] (0.95 confidence)</td>
<td>2.0144e+07 ns [2.00767e+07, 2.02078e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;2d/500</td>
<td>801130 ns [798392, 804216] (0.95 confidence)</td>
<td>803256 ns [796832, 810647] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;2d&#95;adversarial/500</td>
<td>1.11441e+06 ns [1.10652e+06, 1.12334e+06] (0.95 confidence)</td>
<td>1.10062e+06 ns [1.09588e+06, 1.10648e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;3d/100</td>
<td>1.09475e+06 ns [1.08545e+06, 1.10593e+06] (0.95 confidence)</td>
<td>1.26063e+06 ns [1.24818e+06, 1.27497e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;3d&#95;adversarial/100</td>
<td>2.0728e+06 ns [2.05892e+06, 2.09007e+06] (0.95 confidence)</td>
<td>2.29704e+06 ns [2.27329e+06, 2.32178e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;4d/30</td>
<td>3.91667e+06 ns [3.89426e+06, 3.94176e+06] (0.95 confidence)</td>
<td>4.65163e+06 ns [4.62715e+06, 4.67919e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;4d&#95;adversarial/30</td>
<td>3.9108e+06 ns [3.89131e+06, 3.93431e+06] (0.95 confidence)</td>
<td>4.53008e+06 ns [4.49007e+06, 4.57261e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;5d/12</td>
<td>1.64355e+06 ns [1.63576e+06, 1.65299e+06] (0.95 confidence)</td>
<td>2.22168e+06 ns [2.21012e+06, 2.23297e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/delaunay&#95;report&#95;5d&#95;adversarial/12</td>
<td>1.45774e+06 ns [1.45228e+06, 1.46423e+06] (0.95 confidence)</td>
<td>2.03222e+06 ns [2.00881e+06, 2.05906e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;2d/500</td>
<td>811161 ns [804633, 818051] (0.95 confidence)</td>
<td>807948 ns [805044, 810962] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;2d&#95;adversarial/500</td>
<td>1.10928e+06 ns [1.10314e+06, 1.11699e+06] (0.95 confidence)</td>
<td>1.10756e+06 ns [1.10003e+06, 1.11638e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;3d/100</td>
<td>1.08753e+06 ns [1.08005e+06, 1.09706e+06] (0.95 confidence)</td>
<td>1.26255e+06 ns [1.25076e+06, 1.27508e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;3d&#95;adversarial/100</td>
<td>2.09203e+06 ns [2.08352e+06, 2.10219e+06] (0.95 confidence)</td>
<td>2.21589e+06 ns [2.20568e+06, 2.23e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;4d/30</td>
<td>3.90339e+06 ns [3.89295e+06, 3.91543e+06] (0.95 confidence)</td>
<td>4.60834e+06 ns [4.5705e+06, 4.65022e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;4d&#95;adversarial/30</td>
<td>3.88432e+06 ns [3.86614e+06, 3.90274e+06] (0.95 confidence)</td>
<td>4.5034e+06 ns [4.47919e+06, 4.53115e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;5d/12</td>
<td>1.63436e+06 ns [1.62792e+06, 1.64243e+06] (0.95 confidence)</td>
<td>2.20013e+06 ns [2.18496e+06, 2.21644e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/is&#95;valid&#95;delaunay&#95;5d&#95;adversarial/12</td>
<td>1.46203e+06 ns [1.45489e+06, 1.47004e+06] (0.95 confidence)</td>
<td>2.05526e+06 ns [2.04179e+06, 2.06787e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/validate&#95;2d/500</td>
<td>8.08984e+06 ns [8.0598e+06, 8.12096e+06] (0.95 confidence)</td>
<td>8.26879e+06 ns [8.24496e+06, 8.3039e+06] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/validate&#95;2d&#95;adversarial/500</td>
<td>5.33314e+07 ns [5.31204e+07, 5.36288e+07] (0.95 confidence)</td>
<td>4.9046e+07 ns [4.8909e+07, 4.92022e+07] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/validate&#95;3d/100</td>
<td>8.32432e+08 ns [8.31613e+08, 8.33192e+08] (0.95 confidence)</td>
<td>8.53956e+08 ns [8.45629e+08, 8.64738e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/validate&#95;3d&#95;adversarial/100</td>
<td>1.03672e+09 ns [1.03521e+09, 1.03861e+09] (0.95 confidence)</td>
<td>1.02922e+09 ns [1.02506e+09, 1.03481e+09] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/validate&#95;4d/30</td>
<td>3.89267e+09 ns [3.88764e+09, 3.89776e+09] (0.95 confidence)</td>
<td>3.88936e+09 ns [3.88379e+09, 3.89558e+09] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/validate&#95;4d&#95;adversarial/30</td>
<td>4.01795e+09 ns [4.01429e+09, 4.02161e+09] (0.95 confidence)</td>
<td>3.95997e+09 ns [3.95652e+09, 3.96281e+09] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/validate&#95;5d/12</td>
<td>8.68036e+08 ns [8.67052e+08, 8.6905e+08] (0.95 confidence)</td>
<td>8.90802e+08 ns [8.88827e+08, 8.93075e+08] (0.95 confidence)</td>
</tr>
<tr>
<td>validation/validate&#95;5d&#95;adversarial/12</td>
<td>8.12403e+08 ns [8.10949e+08, 8.14223e+08] (0.95 confidence)</td>
<td>8.27638e+08 ns [8.26199e+08, 8.2914e+08] (0.95 confidence)</td>
</tr>
</tbody>
</table>

## Recorded measurement provenance

These observations belong to the measured phases. Rerendering does not recapture the publisher's environment.

### Baseline phase

<table>
<thead>
<tr>
<th>Field</th>
<th>Recorded value</th>
</tr>
</thead>
<tbody>
<tr>
<td>Release</td>
<td>v0.8.3</td>
</tr>
<tr>
<td>Revision</td>
<td>3fa8a9907101c523574a0d17991a1f652b5419dc</td>
</tr>
<tr>
<td>Measurement mode</td>
<td>tag</td>
</tr>
<tr>
<td>Original source SHA-256</td>
<td>a48540134a7fba21f893d7298b5526a5b8ddfd0f09867236d970fce189791933</td>
</tr>
<tr>
<td>Measured source SHA-256</td>
<td>a48540134a7fba21f893d7298b5526a5b8ddfd0f09867236d970fce189791933</td>
</tr>
<tr>
<td>Measured harness SHA-256</td>
<td>4740d40cd1152ca58e9469b3993afdf744269c440afb69bd07ca40e40ebd5dd5</td>
</tr>
<tr>
<td>Fingerprint schema</td>
<td>research-repo-tools/files/v1</td>
</tr>
<tr>
<td>OS</td>
<td>macOS-27.2-arm64-arm-64bit-Mach-O</td>
</tr>
<tr>
<td>Architecture</td>
<td>arm64</td>
</tr>
<tr>
<td>CPU</td>
<td>Apple M4 Max (arm64)</td>
</tr>
<tr>
<td>Physical cores</td>
<td>16</td>
</tr>
<tr>
<td>Logical threads</td>
<td>16</td>
</tr>
<tr>
<td>Memory (bytes)</td>
<td>68719476736</td>
</tr>
<tr>
<td>Tool: build-policy</td>
<td>{&quot;cargo&quot;: {&quot;features&quot;: {&quot;bench&quot;: &#91;&#93;, &quot;bench-logging&quot;: &#91;&#93;, &quot;cli&quot;: &#91;&quot;dep:clap<!--
-->&quot;, &quot;dep:same-file&quot;, &quot;dep:serde&#95;json&quot;, &quot;dep:tempfile&quot;&#93;, &quot;count-allocations&quot;: &#91;&quot;dep:allocat<!--
-->ion-counter&quot;&#93;, &quot;default&quot;: &#91;&#93;, &quot;diagnostics&quot;: &#91;&#93;, &quot;slow-tests&quot;: &#91;&#93;}, &quot;profile&quot;:<!--
--> {&quot;perf&quot;: {&quot;codegen-units&quot;: 1, &quot;debug&quot;: &quot;line-tables-only&quot;, &quot;inherits&quot;: &quot;release&quot;, &quot;lt<!--
-->o&quot;: &quot;thin&quot;}}}, &quot;environment&quot;: {&quot;CARGO&#95;INCREMENTAL&quot;: &quot;0&quot;, &quot;RUSTUP&#95;TOOLCHAIN&quot;: &quot;1.99.<!--
-->1&quot;}}</td>
</tr>
<tr>
<td>Tool: cargo</td>
<td>cargo 1.99.1 (5f94df478 2026-08-27)</td>
</tr>
<tr>
<td>Tool: criterion</td>
<td>&#91;{&quot;checksum&quot;: &quot;950046b2aa2492f9a536f5f4f9a3de7b9e2476e575e05bd6c333371add4d98f3&quot;, &quot;dependencies&quot;: &#91;&quot;alloca<!--
-->&quot;, &quot;anes&quot;, &quot;cast&quot;, &quot;ciborium&quot;, &quot;clap&quot;, &quot;criterion-plot&quot;, &quot;itertools&quot;, &quot;num-traits<!--
-->&quot;, &quot;oorandom&quot;, &quot;page&#95;size&quot;, &quot;plotters&quot;, &quot;rayon&quot;, &quot;regex&quot;, &quot;serde&quot;, &quot;serde<!--
-->&#95;json&quot;, &quot;tinytemplate&quot;, &quot;walkdir&quot;&#93;, &quot;name&quot;: &quot;criterion&quot;, &quot;source&quot;: &quot;registry+https:<!--
-->//github.com/rust-lang/crates.io-index&quot;, &quot;version&quot;: &quot;0.8.2&quot;}&#93;</td>
</tr>
<tr>
<td>Tool: rustc</td>
<td>rustc 1.99.1 (58cf8f9c3 2026-10-13)<br>binary: rustc<br>commit-hash: 58cf8f9c35b355199727627d07ae34a4e2846d67<br>commit-date: 2026-10-13<br>host: aarc<!--
-->h64-apple-darwin<br>release: 1.99.1<br>LLVM version: 23.1.1</td>
</tr>
<tr>
<td>Dependency versions</td>
<td>not recorded</td>
</tr>
<tr>
<td>Environment overrides (names only)</td>
<td>CARGO&#95;INCREMENTAL, CARGO&#95;TARGET&#95;DIR, CRITERION&#95;HOME</td>
</tr>
<tr>
<td>Measurement command (argv)</td>
<td>&#91;&quot;cargo&quot;, &quot;bench&quot;, &quot;--locked&quot;, &quot;--profile&quot;, &quot;perf&quot;, &quot;--bench&quot;, &quot;ci&#95;performanc<!--
-->e&#95;suite&quot;, &quot;--bench&quot;, &quot;circumsphere&#95;containment&quot;, &quot;--bench&quot;, &quot;locate&quot;, &quot;--&quot;, &quot;--save<!--
-->-baseline=retained&quot;, &quot;--noplot&quot;&#93;</td>
</tr>
<tr>
<td>Independent correctness gate</td>
<td>passed</td>
</tr>
<tr>
<td>Gate command (argv)</td>
<td>&#91;&quot;cargo&quot;, &quot;bench&quot;, &quot;--locked&quot;, &quot;--profile&quot;, &quot;perf&quot;, &quot;--bench&quot;, &quot;ci&#95;performanc<!--
-->e&#95;suite&quot;, &quot;--bench&quot;, &quot;circumsphere&#95;containment&quot;, &quot;--bench&quot;, &quot;locate&quot;, &quot;--&quot;, &quot;--test<!--
-->&quot;&#93;</td>
</tr>
</tbody>
</table>

### Current phase

<table>
<thead>
<tr>
<th>Field</th>
<th>Recorded value</th>
</tr>
</thead>
<tbody>
<tr>
<td>Release</td>
<td>v0.8.4</td>
</tr>
<tr>
<td>Revision</td>
<td>317a114297007b1aa47082a73ed1c638a1e52fbf</td>
</tr>
<tr>
<td>Measurement mode</td>
<td>working-tree</td>
</tr>
<tr>
<td>Original source SHA-256</td>
<td>8aa79cc89006bb2327eb63d00d7d86dd16a64e2c111f3f5860ed7dd7781a7aa0</td>
</tr>
<tr>
<td>Measured source SHA-256</td>
<td>8aa79cc89006bb2327eb63d00d7d86dd16a64e2c111f3f5860ed7dd7781a7aa0</td>
</tr>
<tr>
<td>Measured harness SHA-256</td>
<td>4740d40cd1152ca58e9469b3993afdf744269c440afb69bd07ca40e40ebd5dd5</td>
</tr>
<tr>
<td>Fingerprint schema</td>
<td>research-repo-tools/files/v1</td>
</tr>
<tr>
<td>OS</td>
<td>macOS-27.2-arm64-arm-64bit-Mach-O</td>
</tr>
<tr>
<td>Architecture</td>
<td>arm64</td>
</tr>
<tr>
<td>CPU</td>
<td>Apple M4 Max (arm64)</td>
</tr>
<tr>
<td>Physical cores</td>
<td>16</td>
</tr>
<tr>
<td>Logical threads</td>
<td>16</td>
</tr>
<tr>
<td>Memory (bytes)</td>
<td>68719476736</td>
</tr>
<tr>
<td>Tool: build-policy</td>
<td>{&quot;cargo&quot;: {&quot;features&quot;: {&quot;bench&quot;: &#91;&#93;, &quot;bench-logging&quot;: &#91;&#93;, &quot;cli&quot;: &#91;&quot;dep:clap<!--
-->&quot;, &quot;dep:same-file&quot;, &quot;dep:serde&#95;json&quot;, &quot;dep:tempfile&quot;&#93;, &quot;count-allocations&quot;: &#91;&quot;dep:allocat<!--
-->ion-counter&quot;&#93;, &quot;default&quot;: &#91;&#93;, &quot;diagnostics&quot;: &#91;&#93;, &quot;slow-tests&quot;: &#91;&#93;}, &quot;profile&quot;:<!--
--> {&quot;perf&quot;: {&quot;codegen-units&quot;: 1, &quot;debug&quot;: &quot;line-tables-only&quot;, &quot;inherits&quot;: &quot;release&quot;, &quot;lt<!--
-->o&quot;: &quot;thin&quot;}}}, &quot;environment&quot;: {&quot;CARGO&#95;INCREMENTAL&quot;: &quot;0&quot;, &quot;RUSTUP&#95;TOOLCHAIN&quot;: &quot;1.99.<!--
-->1&quot;}}</td>
</tr>
<tr>
<td>Tool: cargo</td>
<td>cargo 1.99.1 (5f94df478 2026-08-27)</td>
</tr>
<tr>
<td>Tool: criterion</td>
<td>&#91;{&quot;checksum&quot;: &quot;950046b2aa2492f9a536f5f4f9a3de7b9e2476e575e05bd6c333371add4d98f3&quot;, &quot;dependencies&quot;: &#91;&quot;alloca<!--
-->&quot;, &quot;anes&quot;, &quot;cast&quot;, &quot;ciborium&quot;, &quot;clap&quot;, &quot;criterion-plot&quot;, &quot;itertools&quot;, &quot;num-traits<!--
-->&quot;, &quot;oorandom&quot;, &quot;page&#95;size&quot;, &quot;plotters&quot;, &quot;rayon&quot;, &quot;regex&quot;, &quot;serde&quot;, &quot;serde<!--
-->&#95;json&quot;, &quot;tinytemplate&quot;, &quot;walkdir&quot;&#93;, &quot;name&quot;: &quot;criterion&quot;, &quot;source&quot;: &quot;registry+https:<!--
-->//github.com/rust-lang/crates.io-index&quot;, &quot;version&quot;: &quot;0.8.2&quot;}&#93;</td>
</tr>
<tr>
<td>Tool: rustc</td>
<td>rustc 1.99.1 (58cf8f9c3 2026-10-13)<br>binary: rustc<br>commit-hash: 58cf8f9c35b355199727627d07ae34a4e2846d67<br>commit-date: 2026-10-13<br>host: aarc<!--
-->h64-apple-darwin<br>release: 1.99.1<br>LLVM version: 23.1.1</td>
</tr>
<tr>
<td>Dependency versions</td>
<td>not recorded</td>
</tr>
<tr>
<td>Environment overrides (names only)</td>
<td>CARGO&#95;INCREMENTAL, CARGO&#95;TARGET&#95;DIR, CRITERION&#95;HOME</td>
</tr>
<tr>
<td>Measurement command (argv)</td>
<td>&#91;&quot;cargo&quot;, &quot;bench&quot;, &quot;--locked&quot;, &quot;--profile&quot;, &quot;perf&quot;, &quot;--bench&quot;, &quot;ci&#95;performanc<!--
-->e&#95;suite&quot;, &quot;--bench&quot;, &quot;circumsphere&#95;containment&quot;, &quot;--bench&quot;, &quot;locate&quot;, &quot;--&quot;, &quot;--save<!--
-->-baseline=retained&quot;, &quot;--noplot&quot;&#93;</td>
</tr>
<tr>
<td>Independent correctness gate</td>
<td>passed</td>
</tr>
<tr>
<td>Gate command (argv)</td>
<td>&#91;&quot;cargo&quot;, &quot;bench&quot;, &quot;--locked&quot;, &quot;--profile&quot;, &quot;perf&quot;, &quot;--bench&quot;, &quot;ci&#95;performanc<!--
-->e&#95;suite&quot;, &quot;--bench&quot;, &quot;circumsphere&#95;containment&quot;, &quot;--bench&quot;, &quot;locate&quot;, &quot;--&quot;, &quot;--test<!--
-->&quot;&#93;</td>
</tr>
</tbody>
</table>

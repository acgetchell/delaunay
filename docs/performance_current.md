<!-- research-repo-tools/run-report/v2 -->
# Delaunay public API performance

Run: `bbd3170146eaa928e78d3888b7ef08643356b765b07de295baa3c94c0dd6071b`

Evidence: [run payload](performance-runs/runs/bbd3170146eaa928e78d3888b7ef08643356b765b07de295baa3c94c0dd6071b/run.json) · [evidence envelope](performance-runs/runs/bbd3170146eaa928e78d3888b7ef08643356b765b07de295baa3c94c0dd6071b/evidence.json).

Intervals are marginal timing intervals. No significance or ratio interval is inferred.

- baseline: measured in **baseline** phase (v0.8.3), revision `3fa8a9907101c523574a0d17991a1f652b5419dc`.
- current: measured in **current** phase (v0.8.4), revision `4d7e9d63c6225418640fa881dc5a38bbf23b1d63`.

## Mean (ns)

| Case | baseline | current |
| --- | --- | --- |
| 2d/insphere | 52.3401 ns [51.8992, 52.7934] (0.95 confidence) | 50.1483 ns [49.9358, 50.3756] (0.95 confidence) |
| 2d/insphere&#95;distance | 13.8075 ns [13.7849, 13.8316] (0.95 confidence) | 13.3591 ns [13.3391, 13.3788] (0.95 confidence) |
| 2d/insphere&#95;lifted | 52.7648 ns [52.4003, 53.1348] (0.95 confidence) | 50.0807 ns [49.8954, 50.273] (0.95 confidence) |
| 3d/insphere | 112.102 ns [111.404, 112.81] (0.95 confidence) | 103.127 ns [102.784, 103.524] (0.95 confidence) |
| 3d/insphere&#95;distance | 34.8821 ns [34.4904, 35.2798] (0.95 confidence) | 32.5821 ns [32.5288, 32.6361] (0.95 confidence) |
| 3d/insphere&#95;lifted | 110.955 ns [110.304, 111.626] (0.95 confidence) | 103.749 ns [103.33, 104.193] (0.95 confidence) |
| 4d/insphere | 770.837 ns [768.173, 773.441] (0.95 confidence) | 746.785 ns [744.561, 749.115] (0.95 confidence) |
| 4d/insphere&#95;distance | 53.5976 ns [53.2346, 53.9791] (0.95 confidence) | 52.9732 ns [52.8602, 53.0798] (0.95 confidence) |
| 4d/insphere&#95;lifted | 775.857 ns [773.317, 778.394] (0.95 confidence) | 732.997 ns [731.373, 734.609] (0.95 confidence) |
| 5d/insphere | 1395.52 ns [1390.77, 1400.46] (0.95 confidence) | 1285.7 ns [1282.37, 1289.18] (0.95 confidence) |
| 5d/insphere&#95;distance | 88.578 ns [88.2501, 88.9361] (0.95 confidence) | 86.6351 ns [86.3845, 86.8874] (0.95 confidence) |
| 5d/insphere&#95;lifted | 1401.98 ns [1397.8, 1406.31] (0.95 confidence) | 1277.01 ns [1274, 1280.1] (0.95 confidence) |
| bistellar&#95;flips&#95;2d/k1&#95;roundtrip | 60436.7 ns [60048.7, 60794.6] (0.95 confidence) | 54929 ns [54599.8, 55307.3] (0.95 confidence) |
| bistellar&#95;flips&#95;2d/k1&#95;roundtrip&#95;adversarial | 181653 ns [179891, 183087] (0.95 confidence) | 162566 ns [162336, 162828] (0.95 confidence) |
| bistellar&#95;flips&#95;2d/k2&#95;edge&#95;flip | 29657 ns [29428.9, 29871.8] (0.95 confidence) | 26793.5 ns [26750.4, 26839.4] (0.95 confidence) |
| bistellar&#95;flips&#95;2d/k2&#95;edge&#95;flip&#95;adversarial | 113053 ns [112400, 113656] (0.95 confidence) | 101727 ns [101535, 101923] (0.95 confidence) |
| bistellar&#95;flips&#95;3d/k1&#95;roundtrip | 2.64921e+06 ns [2.63549e+06, 2.66233e+06] (0.95 confidence) | 2.35601e+06 ns [2.34786e+06, 2.36467e+06] (0.95 confidence) |
| bistellar&#95;flips&#95;3d/k1&#95;roundtrip&#95;adversarial | 2.12039e+06 ns [2.07969e+06, 2.16021e+06] (0.95 confidence) | 1.92263e+06 ns [1.9198e+06, 1.92522e+06] (0.95 confidence) |
| bistellar&#95;flips&#95;3d/k2&#95;roundtrip | 7.44442e+06 ns [7.3736e+06, 7.5044e+06] (0.95 confidence) | 6.52954e+06 ns [6.5179e+06, 6.54379e+06] (0.95 confidence) |
| bistellar&#95;flips&#95;3d/k2&#95;roundtrip&#95;adversarial | 6.31696e+06 ns [6.20642e+06, 6.4446e+06] (0.95 confidence) | 5.8673e+06 ns [5.85763e+06, 5.87702e+06] (0.95 confidence) |
| bistellar&#95;flips&#95;3d/k3&#95;forward | 1.53535e+06 ns [1.51967e+06, 1.55057e+06] (0.95 confidence) | 1.37701e+06 ns [1.37461e+06, 1.37978e+06] (0.95 confidence) |
| bistellar&#95;flips&#95;3d/k3&#95;forward&#95;adversarial | 2.10566e+06 ns [2.08207e+06, 2.12724e+06] (0.95 confidence) | 1.82502e+06 ns [1.82293e+06, 1.8274e+06] (0.95 confidence) |
| bistellar&#95;flips&#95;4d/k1&#95;roundtrip | 4.25154e+07 ns [4.16622e+07, 4.32113e+07] (0.95 confidence) | 3.90255e+07 ns [3.89323e+07, 3.91261e+07] (0.95 confidence) |
| bistellar&#95;flips&#95;4d/k1&#95;roundtrip&#95;adversarial | 3.60958e+07 ns [3.58436e+07, 3.63853e+07] (0.95 confidence) | 3.31007e+07 ns [3.30072e+07, 3.31827e+07] (0.95 confidence) |
| bistellar&#95;flips&#95;4d/k2&#95;roundtrip | 5.90303e+07 ns [5.83767e+07, 5.99518e+07] (0.95 confidence) | 5.58875e+07 ns [5.58058e+07, 5.59673e+07] (0.95 confidence) |
| bistellar&#95;flips&#95;4d/k2&#95;roundtrip&#95;adversarial | 2.05857e+08 ns [2.04909e+08, 2.06865e+08] (0.95 confidence) | 1.93728e+08 ns [1.93433e+08, 1.94051e+08] (0.95 confidence) |
| bistellar&#95;flips&#95;4d/k3&#95;roundtrip | 6.71714e+07 ns [6.70495e+07, 6.7297e+07] (0.95 confidence) | 6.41054e+07 ns [6.3986e+07, 6.42067e+07] (0.95 confidence) |
| bistellar&#95;flips&#95;4d/k3&#95;roundtrip&#95;adversarial | 3.5825e+07 ns [3.55267e+07, 3.61366e+07] (0.95 confidence) | 3.48048e+07 ns [3.47316e+07, 3.49099e+07] (0.95 confidence) |
| bistellar&#95;flips&#95;5d/k1&#95;roundtrip | 1.62108e+08 ns [1.61188e+08, 1.63032e+08] (0.95 confidence) | 1.56672e+08 ns [1.56507e+08, 1.56844e+08] (0.95 confidence) |
| bistellar&#95;flips&#95;5d/k1&#95;roundtrip&#95;adversarial | 1.6878e+08 ns [1.68134e+08, 1.69431e+08] (0.95 confidence) | 1.59444e+08 ns [1.59136e+08, 1.59812e+08] (0.95 confidence) |
| bistellar&#95;flips&#95;5d/k2&#95;roundtrip | 336780 ns [335825, 337493] (0.95 confidence) | 325864 ns [325487, 326237] (0.95 confidence) |
| bistellar&#95;flips&#95;5d/k2&#95;roundtrip&#95;adversarial | 363815 ns [363049, 364455] (0.95 confidence) | 353231 ns [352450, 354045] (0.95 confidence) |
| bistellar&#95;flips&#95;5d/k3&#95;roundtrip | 3.43309e+08 ns [3.40982e+08, 3.45998e+08] (0.95 confidence) | 3.25811e+08 ns [3.25388e+08, 3.26241e+08] (0.95 confidence) |
| bistellar&#95;flips&#95;5d/k3&#95;roundtrip&#95;adversarial | 9.37077e+08 ns [9.35221e+08, 9.39237e+08] (0.95 confidence) | 8.62458e+08 ns [8.61267e+08, 8.63615e+08] (0.95 confidence) |
| boundary&#95;facets/boundary&#95;facets&#95;2d/500 | 230986 ns [229140, 232799] (0.95 confidence) | 205462 ns [204999, 205995] (0.95 confidence) |
| boundary&#95;facets/boundary&#95;facets&#95;2d&#95;adversarial/500 | 230689 ns [228507, 232837] (0.95 confidence) | 205877 ns [205326, 206475] (0.95 confidence) |
| boundary&#95;facets/boundary&#95;facets&#95;3d/100 | 184270 ns [182942, 185632] (0.95 confidence) | 165487 ns [165211, 165754] (0.95 confidence) |
| boundary&#95;facets/boundary&#95;facets&#95;3d&#95;adversarial/100 | 187851 ns [186148, 189602] (0.95 confidence) | 167741 ns [167381, 168083] (0.95 confidence) |
| boundary&#95;facets/boundary&#95;facets&#95;4d/30 | 145887 ns [144585, 147152] (0.95 confidence) | 132541 ns [132220, 132863] (0.95 confidence) |
| boundary&#95;facets/boundary&#95;facets&#95;4d&#95;adversarial/30 | 144388 ns [143087, 145631] (0.95 confidence) | 131817 ns [131574, 132074] (0.95 confidence) |
| boundary&#95;facets/boundary&#95;facets&#95;5d/12 | 40294.8 ns [39952.1, 40627.9] (0.95 confidence) | 37312.6 ns [37162.2, 37468.4] (0.95 confidence) |
| boundary&#95;facets/boundary&#95;facets&#95;5d&#95;adversarial/12 | 39735.4 ns [39422.2, 40061.5] (0.95 confidence) | 36252.5 ns [36188.4, 36313.6] (0.95 confidence) |
| circumcenter/solve&#95;path/near&#95;singular&#95;exact&#95;fallback&#95;3d | 17119 ns [17093.4, 17144.4] (0.95 confidence) | 17174.1 ns [17152.2, 17195.6] (0.95 confidence) |
| circumcenter/solve&#95;path/regular&#95;lu&#95;3d | 22.9998 ns [22.9728, 23.0282] (0.95 confidence) | 23.305 ns [23.2702, 23.3407] (0.95 confidence) |
| convex&#95;hull/from&#95;triangulation&#95;2d/500 | 377023 ns [373828, 380256] (0.95 confidence) | 388035 ns [387539, 388596] (0.95 confidence) |
| convex&#95;hull/from&#95;triangulation&#95;2d&#95;adversarial/500 | 365804 ns [361966, 369317] (0.95 confidence) | 382118 ns [381484, 382804] (0.95 confidence) |
| convex&#95;hull/from&#95;triangulation&#95;3d/100 | 304822 ns [302499, 307102] (0.95 confidence) | 275543 ns [275301, 275773] (0.95 confidence) |
| convex&#95;hull/from&#95;triangulation&#95;3d&#95;adversarial/100 | 4.42763e+06 ns [4.38928e+06, 4.46532e+06] (0.95 confidence) | 3.97401e+06 ns [3.96524e+06, 3.98398e+06] (0.95 confidence) |
| convex&#95;hull/from&#95;triangulation&#95;4d/30 | 9.5876e+06 ns [9.50569e+06, 9.67417e+06] (0.95 confidence) | 8.76797e+06 ns [8.70573e+06, 8.83988e+06] (0.95 confidence) |
| convex&#95;hull/from&#95;triangulation&#95;4d&#95;adversarial/30 | 7.64228e+06 ns [7.58714e+06, 7.69392e+06] (0.95 confidence) | 7.00199e+06 ns [6.97681e+06, 7.02695e+06] (0.95 confidence) |
| convex&#95;hull/from&#95;triangulation&#95;5d/12 | 3.39448e+06 ns [3.36597e+06, 3.4234e+06] (0.95 confidence) | 3.03164e+06 ns [3.02216e+06, 3.04243e+06] (0.95 confidence) |
| convex&#95;hull/from&#95;triangulation&#95;5d&#95;adversarial/12 | 2.89774e+06 ns [2.86272e+06, 2.93142e+06] (0.95 confidence) | 2.65512e+06 ns [2.64863e+06, 2.66274e+06] (0.95 confidence) |
| convex&#95;hull&#95;queries/find&#95;nearest&#95;visible&#95;facet&#95;3d/100 | 1261.46 ns [1251.06, 1271.92] (0.95 confidence) | 1146.76 ns [1144.29, 1149.19] (0.95 confidence) |
| convex&#95;hull&#95;queries/find&#95;nearest&#95;visible&#95;facet&#95;3d&#95;adversarial/100 | 45586.5 ns [45240, 45938] (0.95 confidence) | 41017.3 ns [40904.4, 41133.9] (0.95 confidence) |
| convex&#95;hull&#95;queries/find&#95;visible&#95;facets&#95;3d/100 | 1365.76 ns [1345.65, 1385.14] (0.95 confidence) | 1220.15 ns [1208.07, 1234.29] (0.95 confidence) |
| convex&#95;hull&#95;queries/find&#95;visible&#95;facets&#95;3d&#95;adversarial/100 | 45862.8 ns [45426.7, 46305] (0.95 confidence) | 40807.4 ns [40705.2, 40923.4] (0.95 confidence) |
| convex&#95;hull&#95;queries/is&#95;point&#95;outside&#95;3d/100 | 104.363 ns [103.263, 105.421] (0.95 confidence) | 95.3963 ns [95.317, 95.4723] (0.95 confidence) |
| convex&#95;hull&#95;queries/is&#95;point&#95;outside&#95;3d&#95;adversarial/100 | 851.922 ns [844.847, 858.708] (0.95 confidence) | 767.288 ns [765.443, 769.522] (0.95 confidence) |
| edge&#95;cases&#95;2d/boundary&#95;point&#95;distance | 13.634 ns [13.6058, 13.6624] (0.95 confidence) | 13.4082 ns [13.3818, 13.4336] (0.95 confidence) |
| edge&#95;cases&#95;2d/boundary&#95;point&#95;insphere | 0.827435 ns [0.82465, 0.830304] (0.95 confidence) | 0.814717 ns [0.812755, 0.81673] (0.95 confidence) |
| edge&#95;cases&#95;2d/boundary&#95;point&#95;lifted | 984.639 ns [980.747, 988.684] (0.95 confidence) | 959.351 ns [957.783, 960.976] (0.95 confidence) |
| edge&#95;cases&#95;2d/far&#95;point&#95;distance | 14.0865 ns [14.0605, 14.114] (0.95 confidence) | 13.389 ns [13.3561, 13.4219] (0.95 confidence) |
| edge&#95;cases&#95;2d/far&#95;point&#95;insphere | 51.6024 ns [51.1897, 52.0188] (0.95 confidence) | 50.0289 ns [49.7738, 50.2985] (0.95 confidence) |
| edge&#95;cases&#95;2d/far&#95;point&#95;lifted | 51.4699 ns [51.1573, 51.7913] (0.95 confidence) | 49.8624 ns [49.6815, 50.0436] (0.95 confidence) |
| edge&#95;cases&#95;2d/near&#95;boundary&#95;distance | 14.1756 ns [14.1322, 14.2197] (0.95 confidence) | 13.5411 ns [13.5132, 13.5687] (0.95 confidence) |
| edge&#95;cases&#95;2d/near&#95;boundary&#95;insphere | 49.0976 ns [48.8394, 49.3591] (0.95 confidence) | 47.5618 ns [47.3438, 47.781] (0.95 confidence) |
| edge&#95;cases&#95;2d/near&#95;boundary&#95;lifted | 48.638 ns [48.3779, 48.9055] (0.95 confidence) | 46.6557 ns [46.4678, 46.8477] (0.95 confidence) |
| edge&#95;cases&#95;3d/boundary&#95;point&#95;distance | 32.7357 ns [32.6842, 32.7912] (0.95 confidence) | 32.6704 ns [32.5979, 32.7427] (0.95 confidence) |
| edge&#95;cases&#95;3d/boundary&#95;point&#95;insphere | 1.19968 ns [1.19451, 1.20397] (0.95 confidence) | 1.17421 ns [1.17168, 1.17669] (0.95 confidence) |
| edge&#95;cases&#95;3d/boundary&#95;point&#95;lifted | 1798.77 ns [1792.21, 1805.26] (0.95 confidence) | 1756.77 ns [1754.26, 1759.35] (0.95 confidence) |
| edge&#95;cases&#95;3d/far&#95;point&#95;distance | 33.0242 ns [32.9144, 33.136] (0.95 confidence) | 32.3407 ns [32.2731, 32.4101] (0.95 confidence) |
| edge&#95;cases&#95;3d/far&#95;point&#95;insphere | 110.781 ns [110.388, 111.183] (0.95 confidence) | 104.95 ns [104.639, 105.277] (0.95 confidence) |
| edge&#95;cases&#95;3d/far&#95;point&#95;lifted | 110.992 ns [110.342, 111.656] (0.95 confidence) | 103.64 ns [103.49, 103.783] (0.95 confidence) |
| edge&#95;cases&#95;3d/near&#95;boundary&#95;distance | 33.2564 ns [33.133, 33.3878] (0.95 confidence) | 32.5824 ns [32.5162, 32.6468] (0.95 confidence) |
| edge&#95;cases&#95;3d/near&#95;boundary&#95;insphere | 102.521 ns [101.692, 103.363] (0.95 confidence) | 94.7606 ns [93.9731, 95.5968] (0.95 confidence) |
| edge&#95;cases&#95;3d/near&#95;boundary&#95;lifted | 101.254 ns [100.388, 102.165] (0.95 confidence) | 93.994 ns [93.3875, 94.6441] (0.95 confidence) |
| edge&#95;cases&#95;4d/boundary&#95;point&#95;distance | 51.7806 ns [51.6977, 51.8659] (0.95 confidence) | 53.335 ns [53.2364, 53.4396] (0.95 confidence) |
| edge&#95;cases&#95;4d/boundary&#95;point&#95;insphere | 1.56114 ns [1.53275, 1.58838] (0.95 confidence) | 1.54053 ns [1.51435, 1.56596] (0.95 confidence) |
| edge&#95;cases&#95;4d/boundary&#95;point&#95;lifted | 3758.26 ns [3751.91, 3764.6] (0.95 confidence) | 3734.7 ns [3725.69, 3743.72] (0.95 confidence) |
| edge&#95;cases&#95;4d/far&#95;point&#95;distance | 55.6529 ns [55.1293, 56.1643] (0.95 confidence) | 53.2062 ns [53.0935, 53.3137] (0.95 confidence) |
| edge&#95;cases&#95;4d/far&#95;point&#95;insphere | 768.103 ns [764.94, 771.271] (0.95 confidence) | 733.219 ns [731.487, 735.008] (0.95 confidence) |
| edge&#95;cases&#95;4d/far&#95;point&#95;lifted | 771.873 ns [769.445, 774.309] (0.95 confidence) | 731.2 ns [729.389, 733.06] (0.95 confidence) |
| edge&#95;cases&#95;4d/near&#95;boundary&#95;distance | 52.2516 ns [52.1073, 52.4024] (0.95 confidence) | 53.6145 ns [53.5031, 53.7232] (0.95 confidence) |
| edge&#95;cases&#95;4d/near&#95;boundary&#95;insphere | 758.769 ns [755.762, 761.745] (0.95 confidence) | 716.731 ns [714.862, 718.651] (0.95 confidence) |
| edge&#95;cases&#95;4d/near&#95;boundary&#95;lifted | 753.078 ns [750.813, 755.301] (0.95 confidence) | 715.344 ns [713.397, 717.304] (0.95 confidence) |
| edge&#95;cases&#95;5d/boundary&#95;point&#95;distance | 88.8377 ns [88.3871, 89.3443] (0.95 confidence) | 86.8255 ns [86.5779, 87.0725] (0.95 confidence) |
| edge&#95;cases&#95;5d/boundary&#95;point&#95;insphere | 1.7196 ns [1.70928, 1.73176] (0.95 confidence) | 1.68417 ns [1.68026, 1.68923] (0.95 confidence) |
| edge&#95;cases&#95;5d/boundary&#95;point&#95;lifted | 5957.58 ns [5939.07, 5978.05] (0.95 confidence) | 5812.07 ns [5801.58, 5822.99] (0.95 confidence) |
| edge&#95;cases&#95;5d/far&#95;point&#95;distance | 88.3474 ns [88.0294, 88.664] (0.95 confidence) | 86.9318 ns [86.6565, 87.212] (0.95 confidence) |
| edge&#95;cases&#95;5d/far&#95;point&#95;insphere | 1548.32 ns [1544.11, 1552.43] (0.95 confidence) | 1267.62 ns [1264.83, 1270.5] (0.95 confidence) |
| edge&#95;cases&#95;5d/far&#95;point&#95;lifted | 1356.44 ns [1353.44, 1359.32] (0.95 confidence) | 1273.77 ns [1270.71, 1276.83] (0.95 confidence) |
| edge&#95;cases&#95;5d/near&#95;boundary&#95;distance | 87.1305 ns [86.9404, 87.3264] (0.95 confidence) | 86.5561 ns [86.316, 86.8041] (0.95 confidence) |
| edge&#95;cases&#95;5d/near&#95;boundary&#95;insphere | 1344.5 ns [1342.09, 1346.89] (0.95 confidence) | 1269.59 ns [1264.57, 1274.96] (0.95 confidence) |
| edge&#95;cases&#95;5d/near&#95;boundary&#95;lifted | 1346.66 ns [1343.14, 1349.99] (0.95 confidence) | 1243.88 ns [1240.79, 1247.21] (0.95 confidence) |
| explicit&#95;import/import&#95;pseudomanifold&#95;2d/vertices&#95;120&#95;simplices&#95;226 | 1.68801e+06 ns [1.65952e+06, 1.71864e+06] (0.95 confidence) | 1.58412e+06 ns [1.58083e+06, 1.5877e+06] (0.95 confidence) |
| explicit&#95;import/import&#95;pseudomanifold&#95;3d/vertices&#95;30&#95;simplices&#95;115 | 1.7239e+08 ns [1.70298e+08, 1.74461e+08] (0.95 confidence) | 1.56937e+08 ns [1.56702e+08, 1.5724e+08] (0.95 confidence) |
| explicit&#95;import/import&#95;pseudomanifold&#95;4d/vertices&#95;16&#95;simplices&#95;68 | 5.1033e+08 ns [4.99454e+08, 5.22043e+08] (0.95 confidence) | 4.74908e+08 ns [4.64602e+08, 4.85462e+08] (0.95 confidence) |
| explicit&#95;import/import&#95;pseudomanifold&#95;5d/vertices&#95;10&#95;simplices&#95;20 | 1.4414e+08 ns [1.42237e+08, 1.46249e+08] (0.95 confidence) | 1.33374e+08 ns [1.33129e+08, 1.33701e+08] (0.95 confidence) |
| incremental&#95;insert/insert&#95;2d/10 | 486041 ns [480133, 491754] (0.95 confidence) | 436031 ns [434652, 437625] (0.95 confidence) |
| incremental&#95;insert/insert&#95;2d&#95;adversarial/10 | 3.1629e+06 ns [3.11415e+06, 3.20808e+06] (0.95 confidence) | 2.85668e+06 ns [2.85261e+06, 2.86172e+06] (0.95 confidence) |
| incremental&#95;insert/insert&#95;3d/10 | 2.8408e+06 ns [2.80307e+06, 2.87904e+06] (0.95 confidence) | 2.69187e+06 ns [2.68236e+06, 2.70429e+06] (0.95 confidence) |
| incremental&#95;insert/insert&#95;3d&#95;adversarial/10 | 1.41067e+07 ns [1.3918e+07, 1.43318e+07] (0.95 confidence) | 1.3278e+07 ns [1.32445e+07, 1.33212e+07] (0.95 confidence) |
| incremental&#95;insert/insert&#95;4d/6 | 2.74128e+07 ns [2.7069e+07, 2.77838e+07] (0.95 confidence) | 2.61869e+07 ns [2.61018e+07, 2.62781e+07] (0.95 confidence) |
| incremental&#95;insert/insert&#95;4d&#95;adversarial/6 | 6.65983e+07 ns [6.58803e+07, 6.74457e+07] (0.95 confidence) | 6.51669e+07 ns [6.50267e+07, 6.5329e+07] (0.95 confidence) |
| incremental&#95;insert/insert&#95;5d/4 | 5.03311e+07 ns [4.94892e+07, 5.12292e+07] (0.95 confidence) | 5.01032e+07 ns [4.99292e+07, 5.02935e+07] (0.95 confidence) |
| incremental&#95;insert/insert&#95;5d&#95;adversarial/4 | 5.24763e+07 ns [5.17174e+07, 5.33731e+07] (0.95 confidence) | 5.44126e+07 ns [5.42807e+07, 5.45358e+07] (0.95 confidence) |
| locate/exact&#95;hint/2d/vertices&#95;100&#95;simplices&#95;187&#95;seed&#95;dd618c77b9187836&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 7067.61 ns [7037.72, 7109.79] (0.95 confidence) | 7164.34 ns [7144.1, 7187.45] (0.95 confidence) |
| locate/exact&#95;hint/2d/vertices&#95;2000&#95;simplices&#95;3978&#95;seed&#95;01acf93075e96412&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 7259.21 ns [7248.36, 7272.11] (0.95 confidence) | 7415.04 ns [7376.55, 7459.91] (0.95 confidence) |
| locate/exact&#95;hint/2d/vertices&#95;500&#95;simplices&#95;983&#95;seed&#95;1493be4e9d7a5906&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 7412.89 ns [7399.52, 7426.61] (0.95 confidence) | 7481.82 ns [7474.71, 7490.18] (0.95 confidence) |
| locate/exact&#95;hint/3d/vertices&#95;150&#95;simplices&#95;829&#95;seed&#95;a44b52b395a4b44d&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 14567.3 ns [14521.5, 14621.5] (0.95 confidence) | 14729.9 ns [14689.9, 14787.7] (0.95 confidence) |
| locate/exact&#95;hint/3d/vertices&#95;500&#95;simplices&#95;3052&#95;seed&#95;1493be4f9d7a5907&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 14521.9 ns [14479, 14560.4] (0.95 confidence) | 14646.2 ns [14598.4, 14697.3] (0.95 confidence) |
| locate/exact&#95;hint/3d/vertices&#95;50&#95;simplices&#95;222&#95;seed&#95;f61fc639dc8c3c19&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 14510 ns [14464.9, 14560.6] (0.95 confidence) | 14743.1 ns [14700.5, 14794.9] (0.95 confidence) |
| locate/exact&#95;hint/4d/vertices&#95;100&#95;simplices&#95;1677&#95;seed&#95;dd618c71b9187830&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 1.98769e+06 ns [1.98548e+06, 1.98986e+06] (0.95 confidence) | 2.00994e+06 ns [1.99341e+06, 2.02617e+06] (0.95 confidence) |
| locate/exact&#95;hint/4d/vertices&#95;50&#95;simplices&#95;626&#95;seed&#95;f61fc63edc8c3c1e&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 1.96787e+06 ns [1.95757e+06, 1.97716e+06] (0.95 confidence) | 1.96829e+06 ns [1.95338e+06, 1.98289e+06] (0.95 confidence) |
| locate/exact&#95;hint/5d/vertices&#95;25&#95;simplices&#95;410&#95;seed&#95;63a0e3186e461e08&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 4.69948e+06 ns [4.68479e+06, 4.71854e+06] (0.95 confidence) | 4.66761e+06 ns [4.65878e+06, 4.67599e+06] (0.95 confidence) |
| locate/exact&#95;hint/5d/vertices&#95;40&#95;simplices&#95;1208&#95;seed&#95;a86104fee3a3634d&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 4.74163e+06 ns [4.73714e+06, 4.74638e+06] (0.95 confidence) | 4.77138e+06 ns [4.74314e+06, 4.80092e+06] (0.95 confidence) |
| locate/no&#95;hint/2d/vertices&#95;100&#95;simplices&#95;187&#95;seed&#95;dd618c77b9187836&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 73613.1 ns [73210.3, 74027.8] (0.95 confidence) | 74342.9 ns [74179.3, 74485.4] (0.95 confidence) |
| locate/no&#95;hint/2d/vertices&#95;2000&#95;simplices&#95;3978&#95;seed&#95;01acf93075e96412&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 265273 ns [264495, 266061] (0.95 confidence) | 272494 ns [269898, 275492] (0.95 confidence) |
| locate/no&#95;hint/2d/vertices&#95;500&#95;simplices&#95;983&#95;seed&#95;1493be4e9d7a5906&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 121980 ns [121785, 122147] (0.95 confidence) | 123311 ns [123143, 123454] (0.95 confidence) |
| locate/no&#95;hint/3d/vertices&#95;150&#95;simplices&#95;829&#95;seed&#95;a44b52b395a4b44d&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 123506 ns [123205, 123763] (0.95 confidence) | 126082 ns [125311, 126746] (0.95 confidence) |
| locate/no&#95;hint/3d/vertices&#95;500&#95;simplices&#95;3052&#95;seed&#95;1493be4f9d7a5907&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 189739 ns [189044, 190508] (0.95 confidence) | 192655 ns [191065, 194239] (0.95 confidence) |
| locate/no&#95;hint/3d/vertices&#95;50&#95;simplices&#95;222&#95;seed&#95;f61fc639dc8c3c19&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 65481.1 ns [65097.5, 65931.5] (0.95 confidence) | 65637.3 ns [65350.3, 65948.9] (0.95 confidence) |
| locate/no&#95;hint/4d/vertices&#95;100&#95;simplices&#95;1677&#95;seed&#95;dd618c71b9187830&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 1.12567e+07 ns [1.123e+07, 1.12865e+07] (0.95 confidence) | 1.11329e+07 ns [1.11172e+07, 1.11487e+07] (0.95 confidence) |
| locate/no&#95;hint/4d/vertices&#95;50&#95;simplices&#95;626&#95;seed&#95;f61fc63edc8c3c1e&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 1.11665e+07 ns [1.11061e+07, 1.12343e+07] (0.95 confidence) | 1.11337e+07 ns [1.10646e+07, 1.12065e+07] (0.95 confidence) |
| locate/no&#95;hint/5d/vertices&#95;25&#95;simplices&#95;410&#95;seed&#95;63a0e3186e461e08&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 2.21073e+07 ns [2.19759e+07, 2.22285e+07] (0.95 confidence) | 2.20804e+07 ns [2.20053e+07, 2.21828e+07] (0.95 confidence) |
| locate/no&#95;hint/5d/vertices&#95;40&#95;simplices&#95;1208&#95;seed&#95;a86104fee3a3634d&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 3.97055e+07 ns [3.968e+07, 3.97358e+07] (0.95 confidence) | 3.97228e+07 ns [3.9666e+07, 3.97855e+07] (0.95 confidence) |
| proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;2d/simplices&#95;226 | 126314 ns [123897, 128732] (0.95 confidence) | 119600 ns [118091, 121199] (0.95 confidence) |
| proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;3d/simplices&#95;115 | 159117 ns [155352, 162915] (0.95 confidence) | 147538 ns [146512, 148848] (0.95 confidence) |
| proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;4d/simplices&#95;68 | 1.30989e+06 ns [1.28994e+06, 1.33145e+06] (0.95 confidence) | 1.22981e+06 ns [1.22058e+06, 1.23871e+06] (0.95 confidence) |
| proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;5d/simplices&#95;20 | 520314 ns [515778, 525084] (0.95 confidence) | 464480 ns [460716, 468471] (0.95 confidence) |
| proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;2d/simplices&#95;226 | 1.40538e+06 ns [1.37784e+06, 1.43629e+06] (0.95 confidence) | 1.33846e+06 ns [1.33577e+06, 1.34118e+06] (0.95 confidence) |
| proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;3d/simplices&#95;115 | 1.73465e+08 ns [1.71278e+08, 1.75972e+08] (0.95 confidence) | 1.56965e+08 ns [1.56711e+08, 1.57192e+08] (0.95 confidence) |
| proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;4d/simplices&#95;68 | 5.15014e+08 ns [5.03077e+08, 5.28087e+08] (0.95 confidence) | 4.58556e+08 ns [4.58251e+08, 4.58872e+08] (0.95 confidence) |
| proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;5d/simplices&#95;20 | 1.53354e+08 ns [1.52258e+08, 1.5482e+08] (0.95 confidence) | 1.32133e+08 ns [1.3193e+08, 1.32345e+08] (0.95 confidence) |
| proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;2d/simplices&#95;226 | 1.34486e+06 ns [1.31496e+06, 1.37507e+06] (0.95 confidence) | 1.32524e+06 ns [1.29429e+06, 1.36145e+06] (0.95 confidence) |
| proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;3d/simplices&#95;115 | 1.72485e+08 ns [1.70033e+08, 1.74893e+08] (0.95 confidence) | 1.57396e+08 ns [1.57121e+08, 1.57628e+08] (0.95 confidence) |
| proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;4d/simplices&#95;68 | 5.05104e+08 ns [4.95945e+08, 5.15638e+08] (0.95 confidence) | 4.573e+08 ns [4.56928e+08, 4.57639e+08] (0.95 confidence) |
| proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;5d/simplices&#95;20 | 1.50999e+08 ns [1.48889e+08, 1.52791e+08] (0.95 confidence) | 1.31641e+08 ns [1.31442e+08, 1.31818e+08] (0.95 confidence) |
| random/insphere&#95;1000&#95;queries | 435437 ns [433602, 437377] (0.95 confidence) | 416587 ns [415229, 417976] (0.95 confidence) |
| random/insphere&#95;distance&#95;1000&#95;queries | 66411.5 ns [66288.2, 66536.9] (0.95 confidence) | 64162.1 ns [63988, 64339.5] (0.95 confidence) |
| random/insphere&#95;lifted&#95;1000&#95;queries | 431747 ns [430058, 433509] (0.95 confidence) | 414552 ns [413517, 415681] (0.95 confidence) |
| tds&#95;new&#95;2d/tds&#95;new/500 | 5.07618e+07 ns [5.034e+07, 5.11476e+07] (0.95 confidence) | 4.61542e+07 ns [4.55714e+07, 4.71861e+07] (0.95 confidence) |
| tds&#95;new&#95;2d/tds&#95;new&#95;adversarial/500 | 5.91104e+07 ns [5.77995e+07, 6.01601e+07] (0.95 confidence) | 5.33118e+07 ns [5.32538e+07, 5.33879e+07] (0.95 confidence) |
| tds&#95;new&#95;3d/tds&#95;new/100 | 4.13526e+07 ns [4.1008e+07, 4.17056e+07] (0.95 confidence) | 3.75032e+07 ns [3.72446e+07, 3.77972e+07] (0.95 confidence) |
| tds&#95;new&#95;3d/tds&#95;new&#95;adversarial/100 | 9.26514e+07 ns [9.18003e+07, 9.34205e+07] (0.95 confidence) | 8.37331e+07 ns [8.35578e+07, 8.39064e+07] (0.95 confidence) |
| tds&#95;new&#95;4d/tds&#95;new/30 | 1.04843e+08 ns [1.04138e+08, 1.05692e+08] (0.95 confidence) | 9.58649e+07 ns [9.53988e+07, 9.62441e+07] (0.95 confidence) |
| tds&#95;new&#95;4d/tds&#95;new&#95;adversarial/30 | 9.88616e+07 ns [9.81979e+07, 9.96077e+07] (0.95 confidence) | 9.07478e+07 ns [9.03424e+07, 9.11959e+07] (0.95 confidence) |
| tds&#95;new&#95;5d/tds&#95;new/12 | 2.28336e+07 ns [2.27616e+07, 2.28976e+07] (0.95 confidence) | 2.13796e+07 ns [2.13205e+07, 2.14496e+07] (0.95 confidence) |
| tds&#95;new&#95;5d/tds&#95;new&#95;adversarial/12 | 2.1035e+07 ns [2.08651e+07, 2.12351e+07] (0.95 confidence) | 1.96296e+07 ns [1.95818e+07, 1.96948e+07] (0.95 confidence) |
| validation/delaunay&#95;report&#95;2d/500 | 937961 ns [929713, 946867] (0.95 confidence) | 787728 ns [782790, 793760] (0.95 confidence) |
| validation/delaunay&#95;report&#95;2d&#95;adversarial/500 | 1.27383e+06 ns [1.26082e+06, 1.28809e+06] (0.95 confidence) | 1.09517e+06 ns [1.08947e+06, 1.1021e+06] (0.95 confidence) |
| validation/delaunay&#95;report&#95;3d/100 | 1.29595e+06 ns [1.27925e+06, 1.31348e+06] (0.95 confidence) | 1.20706e+06 ns [1.20013e+06, 1.21811e+06] (0.95 confidence) |
| validation/delaunay&#95;report&#95;3d&#95;adversarial/100 | 2.35112e+06 ns [2.31796e+06, 2.38067e+06] (0.95 confidence) | 2.17505e+06 ns [2.16184e+06, 2.1924e+06] (0.95 confidence) |
| validation/delaunay&#95;report&#95;4d/30 | 4.55223e+06 ns [4.51062e+06, 4.59543e+06] (0.95 confidence) | 4.56482e+06 ns [4.55032e+06, 4.58077e+06] (0.95 confidence) |
| validation/delaunay&#95;report&#95;4d&#95;adversarial/30 | 4.41495e+06 ns [4.3715e+06, 4.45353e+06] (0.95 confidence) | 4.48057e+06 ns [4.43856e+06, 4.52496e+06] (0.95 confidence) |
| validation/delaunay&#95;report&#95;5d/12 | 1.87233e+06 ns [1.85063e+06, 1.892e+06] (0.95 confidence) | 2.10696e+06 ns [2.10038e+06, 2.11512e+06] (0.95 confidence) |
| validation/delaunay&#95;report&#95;5d&#95;adversarial/12 | 1.66132e+06 ns [1.6444e+06, 1.67781e+06] (0.95 confidence) | 1.9127e+06 ns [1.90437e+06, 1.92147e+06] (0.95 confidence) |
| validation/is&#95;valid&#95;delaunay&#95;2d/500 | 933498 ns [924059, 942099] (0.95 confidence) | 789668 ns [784872, 794806] (0.95 confidence) |
| validation/is&#95;valid&#95;delaunay&#95;2d&#95;adversarial/500 | 1.29168e+06 ns [1.27908e+06, 1.30509e+06] (0.95 confidence) | 1.09265e+06 ns [1.08506e+06, 1.10168e+06] (0.95 confidence) |
| validation/is&#95;valid&#95;delaunay&#95;3d/100 | 1.28243e+06 ns [1.27035e+06, 1.29388e+06] (0.95 confidence) | 1.21562e+06 ns [1.20515e+06, 1.22835e+06] (0.95 confidence) |
| validation/is&#95;valid&#95;delaunay&#95;3d&#95;adversarial/100 | 2.37079e+06 ns [2.34554e+06, 2.39358e+06] (0.95 confidence) | 2.21607e+06 ns [2.20143e+06, 2.23212e+06] (0.95 confidence) |
| validation/is&#95;valid&#95;delaunay&#95;4d/30 | 4.52679e+06 ns [4.48462e+06, 4.56872e+06] (0.95 confidence) | 4.54118e+06 ns [4.50998e+06, 4.58163e+06] (0.95 confidence) |
| validation/is&#95;valid&#95;delaunay&#95;4d&#95;adversarial/30 | 4.44274e+06 ns [4.40947e+06, 4.47463e+06] (0.95 confidence) | 4.44593e+06 ns [4.43184e+06, 4.45981e+06] (0.95 confidence) |
| validation/is&#95;valid&#95;delaunay&#95;5d/12 | 1.89658e+06 ns [1.87033e+06, 1.9246e+06] (0.95 confidence) | 2.09892e+06 ns [2.0925e+06, 2.10815e+06] (0.95 confidence) |
| validation/is&#95;valid&#95;delaunay&#95;5d&#95;adversarial/12 | 1.64966e+06 ns [1.62644e+06, 1.67146e+06] (0.95 confidence) | 1.91003e+06 ns [1.90026e+06, 1.92146e+06] (0.95 confidence) |
| validation/validate&#95;2d/500 | 9.17943e+06 ns [9.10385e+06, 9.25028e+06] (0.95 confidence) | 8.23276e+06 ns [8.19904e+06, 8.2817e+06] (0.95 confidence) |
| validation/validate&#95;2d&#95;adversarial/500 | 5.99543e+07 ns [5.94671e+07, 6.03927e+07] (0.95 confidence) | 4.83265e+07 ns [4.82381e+07, 4.8428e+07] (0.95 confidence) |
| validation/validate&#95;3d/100 | 9.75962e+08 ns [9.74415e+08, 9.77415e+08] (0.95 confidence) | 8.45087e+08 ns [8.44102e+08, 8.46217e+08] (0.95 confidence) |
| validation/validate&#95;3d&#95;adversarial/100 | 1.20818e+09 ns [1.20576e+09, 1.21065e+09] (0.95 confidence) | 1.02567e+09 ns [1.02453e+09, 1.02681e+09] (0.95 confidence) |
| validation/validate&#95;4d/30 | 4.43729e+09 ns [4.43249e+09, 4.44136e+09] (0.95 confidence) | 3.8109e+09 ns [3.80726e+09, 3.81434e+09] (0.95 confidence) |
| validation/validate&#95;4d&#95;adversarial/30 | 4.61014e+09 ns [4.60603e+09, 4.61368e+09] (0.95 confidence) | 3.97079e+09 ns [3.9667e+09, 3.97554e+09] (0.95 confidence) |
| validation/validate&#95;5d/12 | 9.9855e+08 ns [9.96619e+08, 1.00054e+09] (0.95 confidence) | 8.55873e+08 ns [8.54942e+08, 8.56853e+08] (0.95 confidence) |
| validation/validate&#95;5d&#95;adversarial/12 | 9.21654e+08 ns [9.08792e+08, 9.32536e+08] (0.95 confidence) | 8.01478e+08 ns [8.00785e+08, 8.02142e+08] (0.95 confidence) |

## Median (ns)

| Case | baseline | current |
| --- | --- | --- |
| 2d/insphere | 51.7748 ns [51.1807, 53.0187] (0.95 confidence) | 49.9235 ns [49.7377, 50.1051] (0.95 confidence) |
| 2d/insphere&#95;distance | 13.7663 ns [13.7436, 13.8066] (0.95 confidence) | 13.3699 ns [13.3531, 13.3809] (0.95 confidence) |
| 2d/insphere&#95;lifted | 52.4449 ns [52.0211, 52.783] (0.95 confidence) | 50.01 ns [49.8113, 50.2562] (0.95 confidence) |
| 3d/insphere | 111.406 ns [110.614, 112.352] (0.95 confidence) | 102.609 ns [102.46, 102.873] (0.95 confidence) |
| 3d/insphere&#95;distance | 33.7179 ns [33.4594, 35.7227] (0.95 confidence) | 32.5516 ns [32.5221, 32.6425] (0.95 confidence) |
| 3d/insphere&#95;lifted | 110.859 ns [109.504, 111.684] (0.95 confidence) | 103.279 ns [103.01, 103.84] (0.95 confidence) |
| 4d/insphere | 774.55 ns [768.131, 777.224] (0.95 confidence) | 743.603 ns [741.363, 746.344] (0.95 confidence) |
| 4d/insphere&#95;distance | 53.6494 ns [52.7203, 53.9089] (0.95 confidence) | 53.0602 ns [52.965, 53.1702] (0.95 confidence) |
| 4d/insphere&#95;lifted | 775.09 ns [773.22, 777.495] (0.95 confidence) | 731.824 ns [730.405, 734.87] (0.95 confidence) |
| 5d/insphere | 1391.42 ns [1385.28, 1399.42] (0.95 confidence) | 1283.9 ns [1278.86, 1286.81] (0.95 confidence) |
| 5d/insphere&#95;distance | 88.3021 ns [87.9863, 88.7915] (0.95 confidence) | 86.6227 ns [86.3007, 86.8063] (0.95 confidence) |
| 5d/insphere&#95;lifted | 1397.81 ns [1393.68, 1403.95] (0.95 confidence) | 1273.65 ns [1270.41, 1277.78] (0.95 confidence) |
| bistellar&#95;flips&#95;2d/k1&#95;roundtrip | 60628.4 ns [59931.6, 60989.1] (0.95 confidence) | 54790.3 ns [54372, 55359.3] (0.95 confidence) |
| bistellar&#95;flips&#95;2d/k1&#95;roundtrip&#95;adversarial | 182570 ns [180316, 183172] (0.95 confidence) | 162514 ns [162247, 162785] (0.95 confidence) |
| bistellar&#95;flips&#95;2d/k2&#95;edge&#95;flip | 29692.2 ns [29346.3, 29975.7] (0.95 confidence) | 26757.7 ns [26738.4, 26878.6] (0.95 confidence) |
| bistellar&#95;flips&#95;2d/k2&#95;edge&#95;flip&#95;adversarial | 113210 ns [112295, 113807] (0.95 confidence) | 101725 ns [101509, 101954] (0.95 confidence) |
| bistellar&#95;flips&#95;3d/k1&#95;roundtrip | 2.65225e+06 ns [2.63445e+06, 2.66358e+06] (0.95 confidence) | 2.35554e+06 ns [2.34226e+06, 2.36897e+06] (0.95 confidence) |
| bistellar&#95;flips&#95;3d/k1&#95;roundtrip&#95;adversarial | 2.14804e+06 ns [2.04398e+06, 2.17658e+06] (0.95 confidence) | 1.92417e+06 ns [1.91966e+06, 1.92508e+06] (0.95 confidence) |
| bistellar&#95;flips&#95;3d/k2&#95;roundtrip | 7.43517e+06 ns [7.41022e+06, 7.53863e+06] (0.95 confidence) | 6.52305e+06 ns [6.51504e+06, 6.53939e+06] (0.95 confidence) |
| bistellar&#95;flips&#95;3d/k2&#95;roundtrip&#95;adversarial | 6.22037e+06 ns [6.17177e+06, 6.55844e+06] (0.95 confidence) | 5.8689e+06 ns [5.85116e+06, 5.87981e+06] (0.95 confidence) |
| bistellar&#95;flips&#95;3d/k3&#95;forward | 1.53579e+06 ns [1.51615e+06, 1.56131e+06] (0.95 confidence) | 1.37717e+06 ns [1.37257e+06, 1.37863e+06] (0.95 confidence) |
| bistellar&#95;flips&#95;3d/k3&#95;forward&#95;adversarial | 2.12092e+06 ns [2.06924e+06, 2.13984e+06] (0.95 confidence) | 1.8233e+06 ns [1.82213e+06, 1.82906e+06] (0.95 confidence) |
| bistellar&#95;flips&#95;4d/k1&#95;roundtrip | 4.30895e+07 ns [4.15037e+07, 4.33851e+07] (0.95 confidence) | 3.90077e+07 ns [3.8922e+07, 3.91568e+07] (0.95 confidence) |
| bistellar&#95;flips&#95;4d/k1&#95;roundtrip&#95;adversarial | 3.58876e+07 ns [3.57422e+07, 3.64171e+07] (0.95 confidence) | 3.31256e+07 ns [3.30088e+07, 3.31995e+07] (0.95 confidence) |
| bistellar&#95;flips&#95;4d/k2&#95;roundtrip | 5.8635e+07 ns [5.81687e+07, 5.93776e+07] (0.95 confidence) | 5.5873e+07 ns [5.57893e+07, 5.60143e+07] (0.95 confidence) |
| bistellar&#95;flips&#95;4d/k2&#95;roundtrip&#95;adversarial | 2.05886e+08 ns [2.04546e+08, 2.07089e+08] (0.95 confidence) | 1.93729e+08 ns [1.93399e+08, 1.93944e+08] (0.95 confidence) |
| bistellar&#95;flips&#95;4d/k3&#95;roundtrip | 6.7128e+07 ns [6.71086e+07, 6.73037e+07] (0.95 confidence) | 6.41335e+07 ns [6.40053e+07, 6.42357e+07] (0.95 confidence) |
| bistellar&#95;flips&#95;4d/k3&#95;roundtrip&#95;adversarial | 3.56616e+07 ns [3.5332e+07, 3.63135e+07] (0.95 confidence) | 3.47598e+07 ns [3.47219e+07, 3.48371e+07] (0.95 confidence) |
| bistellar&#95;flips&#95;5d/k1&#95;roundtrip | 1.61923e+08 ns [1.60646e+08, 1.63972e+08] (0.95 confidence) | 1.56636e+08 ns [1.56479e+08, 1.56925e+08] (0.95 confidence) |
| bistellar&#95;flips&#95;5d/k1&#95;roundtrip&#95;adversarial | 1.68796e+08 ns [1.67823e+08, 1.69662e+08] (0.95 confidence) | 1.5931e+08 ns [1.59126e+08, 1.59757e+08] (0.95 confidence) |
| bistellar&#95;flips&#95;5d/k2&#95;roundtrip | 337033 ns [336448, 337715] (0.95 confidence) | 325822 ns [325428, 326376] (0.95 confidence) |
| bistellar&#95;flips&#95;5d/k2&#95;roundtrip&#95;adversarial | 364046 ns [363501, 364402] (0.95 confidence) | 352961 ns [351903, 354672] (0.95 confidence) |
| bistellar&#95;flips&#95;5d/k3&#95;roundtrip | 3.42169e+08 ns [3.40016e+08, 3.46268e+08] (0.95 confidence) | 3.25718e+08 ns [3.25276e+08, 3.26482e+08] (0.95 confidence) |
| bistellar&#95;flips&#95;5d/k3&#95;roundtrip&#95;adversarial | 9.37764e+08 ns [9.33774e+08, 9.37918e+08] (0.95 confidence) | 8.62429e+08 ns [8.60951e+08, 8.64249e+08] (0.95 confidence) |
| boundary&#95;facets/boundary&#95;facets&#95;2d/500 | 231563 ns [227566, 234936] (0.95 confidence) | 205181 ns [204705, 205512] (0.95 confidence) |
| boundary&#95;facets/boundary&#95;facets&#95;2d&#95;adversarial/500 | 230098 ns [226984, 234447] (0.95 confidence) | 205598 ns [204872, 206489] (0.95 confidence) |
| boundary&#95;facets/boundary&#95;facets&#95;3d/100 | 183885 ns [181766, 187026] (0.95 confidence) | 165447 ns [165297, 165660] (0.95 confidence) |
| boundary&#95;facets/boundary&#95;facets&#95;3d&#95;adversarial/100 | 188481 ns [185438, 189566] (0.95 confidence) | 167658 ns [167484, 168327] (0.95 confidence) |
| boundary&#95;facets/boundary&#95;facets&#95;4d/30 | 146659 ns [143943, 148314] (0.95 confidence) | 132442 ns [131931, 133244] (0.95 confidence) |
| boundary&#95;facets/boundary&#95;facets&#95;4d&#95;adversarial/30 | 144940 ns [143036, 146093] (0.95 confidence) | 131580 ns [131376, 132113] (0.95 confidence) |
| boundary&#95;facets/boundary&#95;facets&#95;5d/12 | 40308.5 ns [39850.9, 40789.5] (0.95 confidence) | 37292.2 ns [37126.9, 37481.8] (0.95 confidence) |
| boundary&#95;facets/boundary&#95;facets&#95;5d&#95;adversarial/12 | 39821.9 ns [39109.8, 40183.7] (0.95 confidence) | 36257.7 ns [36180.4, 36354.4] (0.95 confidence) |
| circumcenter/solve&#95;path/near&#95;singular&#95;exact&#95;fallback&#95;3d | 17134.3 ns [17083, 17190.5] (0.95 confidence) | 17180.1 ns [17156.4, 17220.7] (0.95 confidence) |
| circumcenter/solve&#95;path/regular&#95;lu&#95;3d | 23.0111 ns [22.9801, 23.0442] (0.95 confidence) | 23.3084 ns [23.2922, 23.3385] (0.95 confidence) |
| convex&#95;hull/from&#95;triangulation&#95;2d/500 | 377227 ns [372810, 379671] (0.95 confidence) | 387928 ns [387484, 388440] (0.95 confidence) |
| convex&#95;hull/from&#95;triangulation&#95;2d&#95;adversarial/500 | 367240 ns [361506, 372793] (0.95 confidence) | 381857 ns [381388, 382561] (0.95 confidence) |
| convex&#95;hull/from&#95;triangulation&#95;3d/100 | 305217 ns [300781, 308654] (0.95 confidence) | 275540 ns [275322, 275864] (0.95 confidence) |
| convex&#95;hull/from&#95;triangulation&#95;3d&#95;adversarial/100 | 4.43733e+06 ns [4.35267e+06, 4.50233e+06] (0.95 confidence) | 3.96605e+06 ns [3.95796e+06, 3.98308e+06] (0.95 confidence) |
| convex&#95;hull/from&#95;triangulation&#95;4d/30 | 9.51842e+06 ns [9.46772e+06, 9.71907e+06] (0.95 confidence) | 8.68864e+06 ns [8.68008e+06, 8.72138e+06] (0.95 confidence) |
| convex&#95;hull/from&#95;triangulation&#95;4d&#95;adversarial/30 | 7.69513e+06 ns [7.5749e+06, 7.72141e+06] (0.95 confidence) | 7.00561e+06 ns [6.97487e+06, 7.03969e+06] (0.95 confidence) |
| convex&#95;hull/from&#95;triangulation&#95;5d/12 | 3.37854e+06 ns [3.34432e+06, 3.4578e+06] (0.95 confidence) | 3.02379e+06 ns [3.01774e+06, 3.03165e+06] (0.95 confidence) |
| convex&#95;hull/from&#95;triangulation&#95;5d&#95;adversarial/12 | 2.90952e+06 ns [2.85553e+06, 2.94804e+06] (0.95 confidence) | 2.65185e+06 ns [2.64796e+06, 2.65555e+06] (0.95 confidence) |
| convex&#95;hull&#95;queries/find&#95;nearest&#95;visible&#95;facet&#95;3d/100 | 1259.13 ns [1253.45, 1276.86] (0.95 confidence) | 1145.81 ns [1143.66, 1150.01] (0.95 confidence) |
| convex&#95;hull&#95;queries/find&#95;nearest&#95;visible&#95;facet&#95;3d&#95;adversarial/100 | 45544.2 ns [45285.3, 45960.7] (0.95 confidence) | 41016.7 ns [40928.4, 41133.3] (0.95 confidence) |
| convex&#95;hull&#95;queries/find&#95;visible&#95;facets&#95;3d/100 | 1370.44 ns [1335.09, 1400.4] (0.95 confidence) | 1208.21 ns [1201.75, 1214.69] (0.95 confidence) |
| convex&#95;hull&#95;queries/find&#95;visible&#95;facets&#95;3d&#95;adversarial/100 | 45988.3 ns [45118.7, 46323.5] (0.95 confidence) | 40748.7 ns [40672.2, 40822.9] (0.95 confidence) |
| convex&#95;hull&#95;queries/is&#95;point&#95;outside&#95;3d/100 | 104.446 ns [103.145, 105.821] (0.95 confidence) | 95.3838 ns [95.3453, 95.4939] (0.95 confidence) |
| convex&#95;hull&#95;queries/is&#95;point&#95;outside&#95;3d&#95;adversarial/100 | 852.119 ns [846.633, 861.099] (0.95 confidence) | 765.594 ns [764.806, 767.807] (0.95 confidence) |
| edge&#95;cases&#95;2d/boundary&#95;point&#95;distance | 13.6074 ns [13.5739, 13.6577] (0.95 confidence) | 13.4417 ns [13.3934, 13.4663] (0.95 confidence) |
| edge&#95;cases&#95;2d/boundary&#95;point&#95;insphere | 0.824134 ns [0.81879, 0.829862] (0.95 confidence) | 0.812233 ns [0.810711, 0.815776] (0.95 confidence) |
| edge&#95;cases&#95;2d/boundary&#95;point&#95;lifted | 978.418 ns [974.007, 982.54] (0.95 confidence) | 958.188 ns [956.676, 960.844] (0.95 confidence) |
| edge&#95;cases&#95;2d/far&#95;point&#95;distance | 14.0628 ns [14.0356, 14.0888] (0.95 confidence) | 13.4314 ns [13.375, 13.4565] (0.95 confidence) |
| edge&#95;cases&#95;2d/far&#95;point&#95;insphere | 51.551 ns [51.1163, 51.9971] (0.95 confidence) | 49.8199 ns [49.6088, 50.1115] (0.95 confidence) |
| edge&#95;cases&#95;2d/far&#95;point&#95;lifted | 51.0153 ns [50.5571, 51.6818] (0.95 confidence) | 49.8813 ns [49.6163, 50.1134] (0.95 confidence) |
| edge&#95;cases&#95;2d/near&#95;boundary&#95;distance | 14.215 ns [14.1234, 14.2452] (0.95 confidence) | 13.5542 ns [13.5239, 13.5828] (0.95 confidence) |
| edge&#95;cases&#95;2d/near&#95;boundary&#95;insphere | 49.0638 ns [48.8343, 49.2938] (0.95 confidence) | 47.5333 ns [47.2916, 47.9295] (0.95 confidence) |
| edge&#95;cases&#95;2d/near&#95;boundary&#95;lifted | 48.5003 ns [48.276, 48.9281] (0.95 confidence) | 46.5847 ns [46.4311, 46.778] (0.95 confidence) |
| edge&#95;cases&#95;3d/boundary&#95;point&#95;distance | 32.6711 ns [32.6236, 32.7192] (0.95 confidence) | 32.6806 ns [32.5825, 32.7785] (0.95 confidence) |
| edge&#95;cases&#95;3d/boundary&#95;point&#95;insphere | 1.20422 ns [1.20256, 1.20687] (0.95 confidence) | 1.1763 ns [1.17406, 1.17755] (0.95 confidence) |
| edge&#95;cases&#95;3d/boundary&#95;point&#95;lifted | 1805.79 ns [1797.53, 1813.08] (0.95 confidence) | 1755.5 ns [1750.87, 1759.39] (0.95 confidence) |
| edge&#95;cases&#95;3d/far&#95;point&#95;distance | 32.8737 ns [32.795, 33.1024] (0.95 confidence) | 32.2929 ns [32.2159, 32.3814] (0.95 confidence) |
| edge&#95;cases&#95;3d/far&#95;point&#95;insphere | 110.546 ns [110.134, 110.96] (0.95 confidence) | 104.529 ns [104.253, 104.762] (0.95 confidence) |
| edge&#95;cases&#95;3d/far&#95;point&#95;lifted | 110.058 ns [109.305, 111.453] (0.95 confidence) | 103.761 ns [103.522, 103.892] (0.95 confidence) |
| edge&#95;cases&#95;3d/near&#95;boundary&#95;distance | 33.0942 ns [32.9649, 33.2967] (0.95 confidence) | 32.6113 ns [32.5156, 32.6664] (0.95 confidence) |
| edge&#95;cases&#95;3d/near&#95;boundary&#95;insphere | 101.987 ns [100.404, 103.556] (0.95 confidence) | 93.017 ns [92.7146, 93.7508] (0.95 confidence) |
| edge&#95;cases&#95;3d/near&#95;boundary&#95;lifted | 99.7627 ns [99.1531, 101.038] (0.95 confidence) | 92.721 ns [92.3883, 93.4807] (0.95 confidence) |
| edge&#95;cases&#95;4d/boundary&#95;point&#95;distance | 51.6762 ns [51.5706, 51.8019] (0.95 confidence) | 53.2911 ns [53.2456, 53.3341] (0.95 confidence) |
| edge&#95;cases&#95;4d/boundary&#95;point&#95;insphere | 1.64015 ns [1.61877, 1.65124] (0.95 confidence) | 1.61005 ns [1.58204, 1.62364] (0.95 confidence) |
| edge&#95;cases&#95;4d/boundary&#95;point&#95;lifted | 3762.07 ns [3756.54, 3767.67] (0.95 confidence) | 3733.27 ns [3722.01, 3744.7] (0.95 confidence) |
| edge&#95;cases&#95;4d/far&#95;point&#95;distance | 56.7716 ns [55.7843, 57.2835] (0.95 confidence) | 53.3451 ns [53.2751, 53.4279] (0.95 confidence) |
| edge&#95;cases&#95;4d/far&#95;point&#95;insphere | 767.736 ns [763.09, 772.269] (0.95 confidence) | 733.482 ns [731.428, 734.252] (0.95 confidence) |
| edge&#95;cases&#95;4d/far&#95;point&#95;lifted | 771.507 ns [769.149, 774.422] (0.95 confidence) | 729.362 ns [727.624, 732.282] (0.95 confidence) |
| edge&#95;cases&#95;4d/near&#95;boundary&#95;distance | 52.1812 ns [51.9926, 52.3115] (0.95 confidence) | 53.7404 ns [53.5688, 53.8779] (0.95 confidence) |
| edge&#95;cases&#95;4d/near&#95;boundary&#95;insphere | 760.567 ns [754.825, 763.27] (0.95 confidence) | 715.97 ns [714.725, 717.405] (0.95 confidence) |
| edge&#95;cases&#95;4d/near&#95;boundary&#95;lifted | 752.711 ns [750.584, 755.935] (0.95 confidence) | 715.2 ns [713.209, 718.434] (0.95 confidence) |
| edge&#95;cases&#95;5d/boundary&#95;point&#95;distance | 88.1913 ns [87.9182, 89.0996] (0.95 confidence) | 86.7508 ns [86.5302, 87.0497] (0.95 confidence) |
| edge&#95;cases&#95;5d/boundary&#95;point&#95;insphere | 1.70688 ns [1.69987, 1.71275] (0.95 confidence) | 1.67872 ns [1.67833, 1.67915] (0.95 confidence) |
| edge&#95;cases&#95;5d/boundary&#95;point&#95;lifted | 5949.32 ns [5929.47, 5963.23] (0.95 confidence) | 5818.7 ns [5805.94, 5825] (0.95 confidence) |
| edge&#95;cases&#95;5d/far&#95;point&#95;distance | 87.9912 ns [87.835, 88.4974] (0.95 confidence) | 86.6725 ns [86.4176, 86.8682] (0.95 confidence) |
| edge&#95;cases&#95;5d/far&#95;point&#95;insphere | 1550.4 ns [1544.74, 1554.07] (0.95 confidence) | 1262.79 ns [1260.3, 1268.79] (0.95 confidence) |
| edge&#95;cases&#95;5d/far&#95;point&#95;lifted | 1359.74 ns [1356.23, 1362.69] (0.95 confidence) | 1273.57 ns [1268.68, 1276.79] (0.95 confidence) |
| edge&#95;cases&#95;5d/near&#95;boundary&#95;distance | 87.1071 ns [87.0047, 87.2501] (0.95 confidence) | 86.2855 ns [86.0404, 86.6116] (0.95 confidence) |
| edge&#95;cases&#95;5d/near&#95;boundary&#95;insphere | 1342.7 ns [1340.69, 1348.15] (0.95 confidence) | 1265.51 ns [1261.02, 1271.36] (0.95 confidence) |
| edge&#95;cases&#95;5d/near&#95;boundary&#95;lifted | 1350.21 ns [1347.22, 1354.03] (0.95 confidence) | 1241.45 ns [1238.71, 1244.55] (0.95 confidence) |
| explicit&#95;import/import&#95;pseudomanifold&#95;2d/vertices&#95;120&#95;simplices&#95;226 | 1.68328e+06 ns [1.64736e+06, 1.72126e+06] (0.95 confidence) | 1.58232e+06 ns [1.58083e+06, 1.58885e+06] (0.95 confidence) |
| explicit&#95;import/import&#95;pseudomanifold&#95;3d/vertices&#95;30&#95;simplices&#95;115 | 1.72205e+08 ns [1.68609e+08, 1.76264e+08] (0.95 confidence) | 1.56768e+08 ns [1.5662e+08, 1.57131e+08] (0.95 confidence) |
| explicit&#95;import/import&#95;pseudomanifold&#95;4d/vertices&#95;16&#95;simplices&#95;68 | 5.0591e+08 ns [4.91961e+08, 5.28971e+08] (0.95 confidence) | 4.79105e+08 ns [4.54487e+08, 4.81165e+08] (0.95 confidence) |
| explicit&#95;import/import&#95;pseudomanifold&#95;5d/vertices&#95;10&#95;simplices&#95;20 | 1.42034e+08 ns [1.41515e+08, 1.47875e+08] (0.95 confidence) | 1.33196e+08 ns [1.33177e+08, 1.33483e+08] (0.95 confidence) |
| incremental&#95;insert/insert&#95;2d/10 | 486876 ns [479543, 494899] (0.95 confidence) | 434406 ns [433830, 438395] (0.95 confidence) |
| incremental&#95;insert/insert&#95;2d&#95;adversarial/10 | 3.17294e+06 ns [3.07109e+06, 3.24209e+06] (0.95 confidence) | 2.85566e+06 ns [2.8507e+06, 2.86174e+06] (0.95 confidence) |
| incremental&#95;insert/insert&#95;3d/10 | 2.86009e+06 ns [2.77362e+06, 2.90505e+06] (0.95 confidence) | 2.68686e+06 ns [2.677e+06, 2.69931e+06] (0.95 confidence) |
| incremental&#95;insert/insert&#95;3d&#95;adversarial/10 | 1.39455e+07 ns [1.38636e+07, 1.41424e+07] (0.95 confidence) | 1.32524e+07 ns [1.32305e+07, 1.32811e+07] (0.95 confidence) |
| incremental&#95;insert/insert&#95;4d/6 | 2.69959e+07 ns [2.68642e+07, 2.80703e+07] (0.95 confidence) | 2.61164e+07 ns [2.60328e+07, 2.641e+07] (0.95 confidence) |
| incremental&#95;insert/insert&#95;4d&#95;adversarial/6 | 6.58094e+07 ns [6.55033e+07, 6.76807e+07] (0.95 confidence) | 6.50104e+07 ns [6.49724e+07, 6.52365e+07] (0.95 confidence) |
| incremental&#95;insert/insert&#95;5d/4 | 5.05615e+07 ns [4.88198e+07, 5.13955e+07] (0.95 confidence) | 5.00269e+07 ns [4.98419e+07, 5.03326e+07] (0.95 confidence) |
| incremental&#95;insert/insert&#95;5d&#95;adversarial/4 | 5.18135e+07 ns [5.12698e+07, 5.25216e+07] (0.95 confidence) | 5.44593e+07 ns [5.43461e+07, 5.45987e+07] (0.95 confidence) |
| locate/exact&#95;hint/2d/vertices&#95;100&#95;simplices&#95;187&#95;seed&#95;dd618c77b9187836&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 7040.92 ns [7033.9, 7080.71] (0.95 confidence) | 7151.28 ns [7133.28, 7192.15] (0.95 confidence) |
| locate/exact&#95;hint/2d/vertices&#95;2000&#95;simplices&#95;3978&#95;seed&#95;01acf93075e96412&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 7254.38 ns [7245.6, 7269.13] (0.95 confidence) | 7380.24 ns [7360.89, 7486.29] (0.95 confidence) |
| locate/exact&#95;hint/2d/vertices&#95;500&#95;simplices&#95;983&#95;seed&#95;1493be4e9d7a5906&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 7408.7 ns [7398.11, 7432.94] (0.95 confidence) | 7479.54 ns [7472.76, 7488.89] (0.95 confidence) |
| locate/exact&#95;hint/3d/vertices&#95;150&#95;simplices&#95;829&#95;seed&#95;a44b52b395a4b44d&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 14527.9 ns [14520, 14615.3] (0.95 confidence) | 14713.9 ns [14677.4, 14742] (0.95 confidence) |
| locate/exact&#95;hint/3d/vertices&#95;500&#95;simplices&#95;3052&#95;seed&#95;1493be4f9d7a5907&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 14519.5 ns [14496.9, 14569.2] (0.95 confidence) | 14648.8 ns [14569.8, 14703.9] (0.95 confidence) |
| locate/exact&#95;hint/3d/vertices&#95;50&#95;simplices&#95;222&#95;seed&#95;f61fc639dc8c3c19&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 14496.2 ns [14457.6, 14572.5] (0.95 confidence) | 14724.3 ns [14694.4, 14792.2] (0.95 confidence) |
| locate/exact&#95;hint/4d/vertices&#95;100&#95;simplices&#95;1677&#95;seed&#95;dd618c71b9187830&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 1.98727e+06 ns [1.98491e+06, 1.99131e+06] (0.95 confidence) | 2.01644e+06 ns [1.98143e+06, 2.0316e+06] (0.95 confidence) |
| locate/exact&#95;hint/4d/vertices&#95;50&#95;simplices&#95;626&#95;seed&#95;f61fc63edc8c3c1e&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 1.96984e+06 ns [1.95668e+06, 1.97924e+06] (0.95 confidence) | 1.96487e+06 ns [1.94881e+06, 1.99347e+06] (0.95 confidence) |
| locate/exact&#95;hint/5d/vertices&#95;25&#95;simplices&#95;410&#95;seed&#95;63a0e3186e461e08&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 4.68824e+06 ns [4.68203e+06, 4.72151e+06] (0.95 confidence) | 4.67057e+06 ns [4.65061e+06, 4.67978e+06] (0.95 confidence) |
| locate/exact&#95;hint/5d/vertices&#95;40&#95;simplices&#95;1208&#95;seed&#95;a86104fee3a3634d&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 4.74095e+06 ns [4.73559e+06, 4.74811e+06] (0.95 confidence) | 4.7691e+06 ns [4.72842e+06, 4.8082e+06] (0.95 confidence) |
| locate/no&#95;hint/2d/vertices&#95;100&#95;simplices&#95;187&#95;seed&#95;dd618c77b9187836&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 73560 ns [72898.3, 74234.3] (0.95 confidence) | 74378.6 ns [74173.3, 74506] (0.95 confidence) |
| locate/no&#95;hint/2d/vertices&#95;2000&#95;simplices&#95;3978&#95;seed&#95;01acf93075e96412&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 265173 ns [264278, 266360] (0.95 confidence) | 271418 ns [268652, 274889] (0.95 confidence) |
| locate/no&#95;hint/2d/vertices&#95;500&#95;simplices&#95;983&#95;seed&#95;1493be4e9d7a5906&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 122042 ns [121827, 122136] (0.95 confidence) | 123437 ns [123123, 123510] (0.95 confidence) |
| locate/no&#95;hint/3d/vertices&#95;150&#95;simplices&#95;829&#95;seed&#95;a44b52b395a4b44d&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 123571 ns [123210, 123903] (0.95 confidence) | 126568 ns [124744, 127013] (0.95 confidence) |
| locate/no&#95;hint/3d/vertices&#95;500&#95;simplices&#95;3052&#95;seed&#95;1493be4f9d7a5907&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 189530 ns [188720, 190613] (0.95 confidence) | 192622 ns [190008, 195141] (0.95 confidence) |
| locate/no&#95;hint/3d/vertices&#95;50&#95;simplices&#95;222&#95;seed&#95;f61fc639dc8c3c19&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 65133.1 ns [64949.1, 66011.8] (0.95 confidence) | 65603 ns [65271.1, 66018.3] (0.95 confidence) |
| locate/no&#95;hint/4d/vertices&#95;100&#95;simplices&#95;1677&#95;seed&#95;dd618c71b9187830&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 1.12477e+07 ns [1.12225e+07, 1.12874e+07] (0.95 confidence) | 1.11308e+07 ns [1.11167e+07, 1.11512e+07] (0.95 confidence) |
| locate/no&#95;hint/4d/vertices&#95;50&#95;simplices&#95;626&#95;seed&#95;f61fc63edc8c3c1e&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 1.11301e+07 ns [1.10873e+07, 1.12542e+07] (0.95 confidence) | 1.10981e+07 ns [1.10185e+07, 1.12542e+07] (0.95 confidence) |
| locate/no&#95;hint/5d/vertices&#95;25&#95;simplices&#95;410&#95;seed&#95;63a0e3186e461e08&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 2.21897e+07 ns [2.19175e+07, 2.22396e+07] (0.95 confidence) | 2.20413e+07 ns [2.20058e+07, 2.21122e+07] (0.95 confidence) |
| locate/no&#95;hint/5d/vertices&#95;40&#95;simplices&#95;1208&#95;seed&#95;a86104fee3a3634d&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 3.9685e+07 ns [3.96748e+07, 3.97398e+07] (0.95 confidence) | 3.96744e+07 ns [3.96554e+07, 3.98316e+07] (0.95 confidence) |
| proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;2d/simplices&#95;226 | 126902 ns [122427, 130097] (0.95 confidence) | 119248 ns [117009, 121851] (0.95 confidence) |
| proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;3d/simplices&#95;115 | 159671 ns [153019, 164650] (0.95 confidence) | 146791 ns [146127, 148264] (0.95 confidence) |
| proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;4d/simplices&#95;68 | 1.29785e+06 ns [1.27929e+06, 1.34505e+06] (0.95 confidence) | 1.23085e+06 ns [1.2201e+06, 1.23794e+06] (0.95 confidence) |
| proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;5d/simplices&#95;20 | 520540 ns [517593, 522343] (0.95 confidence) | 464066 ns [457978, 470400] (0.95 confidence) |
| proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;2d/simplices&#95;226 | 1.39981e+06 ns [1.35854e+06, 1.43931e+06] (0.95 confidence) | 1.33805e+06 ns [1.33511e+06, 1.34218e+06] (0.95 confidence) |
| proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;3d/simplices&#95;115 | 1.7288e+08 ns [1.70127e+08, 1.7621e+08] (0.95 confidence) | 1.57059e+08 ns [1.5664e+08, 1.57325e+08] (0.95 confidence) |
| proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;4d/simplices&#95;68 | 5.14095e+08 ns [4.94659e+08, 5.32075e+08] (0.95 confidence) | 4.58397e+08 ns [4.58195e+08, 4.59048e+08] (0.95 confidence) |
| proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;5d/simplices&#95;20 | 1.53107e+08 ns [1.52415e+08, 1.53576e+08] (0.95 confidence) | 1.32029e+08 ns [1.31837e+08, 1.32479e+08] (0.95 confidence) |
| proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;2d/simplices&#95;226 | 1.34848e+06 ns [1.29526e+06, 1.39252e+06] (0.95 confidence) | 1.32395e+06 ns [1.27935e+06, 1.35159e+06] (0.95 confidence) |
| proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;3d/simplices&#95;115 | 1.72569e+08 ns [1.6873e+08, 1.76329e+08] (0.95 confidence) | 1.57449e+08 ns [1.57175e+08, 1.57741e+08] (0.95 confidence) |
| proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;4d/simplices&#95;68 | 4.95993e+08 ns [4.92e+08, 5.16369e+08] (0.95 confidence) | 4.57527e+08 ns [4.56779e+08, 4.57752e+08] (0.95 confidence) |
| proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;5d/simplices&#95;20 | 1.52749e+08 ns [1.48333e+08, 1.53105e+08] (0.95 confidence) | 1.31667e+08 ns [1.31432e+08, 1.31937e+08] (0.95 confidence) |
| random/insphere&#95;1000&#95;queries | 434277 ns [432196, 436109] (0.95 confidence) | 416465 ns [415203, 417710] (0.95 confidence) |
| random/insphere&#95;distance&#95;1000&#95;queries | 66339.7 ns [66256.6, 66484.3] (0.95 confidence) | 63930.4 ns [63730.6, 64312.8] (0.95 confidence) |
| random/insphere&#95;lifted&#95;1000&#95;queries | 430250 ns [429026, 432529] (0.95 confidence) | 415158 ns [414163, 415873] (0.95 confidence) |
| tds&#95;new&#95;2d/tds&#95;new/500 | 5.07864e+07 ns [5.02226e+07, 5.13329e+07] (0.95 confidence) | 4.56466e+07 ns [4.54887e+07, 4.59654e+07] (0.95 confidence) |
| tds&#95;new&#95;2d/tds&#95;new&#95;adversarial/500 | 5.99177e+07 ns [5.76739e+07, 6.00593e+07] (0.95 confidence) | 5.32696e+07 ns [5.32416e+07, 5.33689e+07] (0.95 confidence) |
| tds&#95;new&#95;3d/tds&#95;new/100 | 4.13705e+07 ns [4.10921e+07, 4.15683e+07] (0.95 confidence) | 3.72588e+07 ns [3.71412e+07, 3.81017e+07] (0.95 confidence) |
| tds&#95;new&#95;3d/tds&#95;new&#95;adversarial/100 | 9.28583e+07 ns [9.19991e+07, 9.35127e+07] (0.95 confidence) | 8.37737e+07 ns [8.33869e+07, 8.39913e+07] (0.95 confidence) |
| tds&#95;new&#95;4d/tds&#95;new/30 | 1.04583e+08 ns [1.03845e+08, 1.05634e+08] (0.95 confidence) | 9.60714e+07 ns [9.53888e+07, 9.63171e+07] (0.95 confidence) |
| tds&#95;new&#95;4d/tds&#95;new&#95;adversarial/30 | 9.86389e+07 ns [9.78027e+07, 9.98538e+07] (0.95 confidence) | 9.07406e+07 ns [9.02577e+07, 9.10667e+07] (0.95 confidence) |
| tds&#95;new&#95;5d/tds&#95;new/12 | 2.28432e+07 ns [2.27722e+07, 2.29052e+07] (0.95 confidence) | 2.13448e+07 ns [2.13053e+07, 2.14477e+07] (0.95 confidence) |
| tds&#95;new&#95;5d/tds&#95;new&#95;adversarial/12 | 2.09961e+07 ns [2.08782e+07, 2.11521e+07] (0.95 confidence) | 1.95846e+07 ns [1.95681e+07, 1.96716e+07] (0.95 confidence) |
| validation/delaunay&#95;report&#95;2d/500 | 937650 ns [924609, 941452] (0.95 confidence) | 784537 ns [780421, 786438] (0.95 confidence) |
| validation/delaunay&#95;report&#95;2d&#95;adversarial/500 | 1.27193e+06 ns [1.25398e+06, 1.28964e+06] (0.95 confidence) | 1.09014e+06 ns [1.08639e+06, 1.10281e+06] (0.95 confidence) |
| validation/delaunay&#95;report&#95;3d/100 | 1.29734e+06 ns [1.26193e+06, 1.3213e+06] (0.95 confidence) | 1.20076e+06 ns [1.19843e+06, 1.20374e+06] (0.95 confidence) |
| validation/delaunay&#95;report&#95;3d&#95;adversarial/100 | 2.34942e+06 ns [2.32523e+06, 2.39519e+06] (0.95 confidence) | 2.16492e+06 ns [2.15727e+06, 2.16635e+06] (0.95 confidence) |
| validation/delaunay&#95;report&#95;4d/30 | 4.53073e+06 ns [4.50981e+06, 4.59259e+06] (0.95 confidence) | 4.55557e+06 ns [4.53853e+06, 4.58301e+06] (0.95 confidence) |
| validation/delaunay&#95;report&#95;4d&#95;adversarial/30 | 4.42732e+06 ns [4.36869e+06, 4.46224e+06] (0.95 confidence) | 4.46398e+06 ns [4.41775e+06, 4.57648e+06] (0.95 confidence) |
| validation/delaunay&#95;report&#95;5d/12 | 1.88612e+06 ns [1.83691e+06, 1.91323e+06] (0.95 confidence) | 2.10336e+06 ns [2.09708e+06, 2.10915e+06] (0.95 confidence) |
| validation/delaunay&#95;report&#95;5d&#95;adversarial/12 | 1.66652e+06 ns [1.62847e+06, 1.69771e+06] (0.95 confidence) | 1.91334e+06 ns [1.90191e+06, 1.91787e+06] (0.95 confidence) |
| validation/is&#95;valid&#95;delaunay&#95;2d/500 | 936970 ns [923056, 941505] (0.95 confidence) | 788727 ns [784042, 794483] (0.95 confidence) |
| validation/is&#95;valid&#95;delaunay&#95;2d&#95;adversarial/500 | 1.29158e+06 ns [1.2704e+06, 1.30938e+06] (0.95 confidence) | 1.08547e+06 ns [1.08308e+06, 1.09271e+06] (0.95 confidence) |
| validation/is&#95;valid&#95;delaunay&#95;3d/100 | 1.28526e+06 ns [1.26713e+06, 1.29676e+06] (0.95 confidence) | 1.20928e+06 ns [1.1988e+06, 1.21739e+06] (0.95 confidence) |
| validation/is&#95;valid&#95;delaunay&#95;3d&#95;adversarial/100 | 2.37747e+06 ns [2.32735e+06, 2.40159e+06] (0.95 confidence) | 2.20899e+06 ns [2.19e+06, 2.22917e+06] (0.95 confidence) |
| validation/is&#95;valid&#95;delaunay&#95;4d/30 | 4.52425e+06 ns [4.46096e+06, 4.55809e+06] (0.95 confidence) | 4.51418e+06 ns [4.5061e+06, 4.5737e+06] (0.95 confidence) |
| validation/is&#95;valid&#95;delaunay&#95;4d&#95;adversarial/30 | 4.45155e+06 ns [4.38411e+06, 4.50431e+06] (0.95 confidence) | 4.44792e+06 ns [4.42615e+06, 4.47103e+06] (0.95 confidence) |
| validation/is&#95;valid&#95;delaunay&#95;5d/12 | 1.89873e+06 ns [1.8706e+06, 1.91349e+06] (0.95 confidence) | 2.09556e+06 ns [2.08806e+06, 2.10269e+06] (0.95 confidence) |
| validation/is&#95;valid&#95;delaunay&#95;5d&#95;adversarial/12 | 1.66075e+06 ns [1.61684e+06, 1.69119e+06] (0.95 confidence) | 1.90089e+06 ns [1.89581e+06, 1.92398e+06] (0.95 confidence) |
| validation/validate&#95;2d/500 | 9.18069e+06 ns [9.05266e+06, 9.28151e+06] (0.95 confidence) | 8.20634e+06 ns [8.18494e+06, 8.25625e+06] (0.95 confidence) |
| validation/validate&#95;2d&#95;adversarial/500 | 6.00353e+07 ns [5.95977e+07, 6.06547e+07] (0.95 confidence) | 4.82401e+07 ns [4.81871e+07, 4.84094e+07] (0.95 confidence) |
| validation/validate&#95;3d/100 | 9.75998e+08 ns [9.74123e+08, 9.77279e+08] (0.95 confidence) | 8.45428e+08 ns [8.43634e+08, 8.46212e+08] (0.95 confidence) |
| validation/validate&#95;3d&#95;adversarial/100 | 1.20769e+09 ns [1.20489e+09, 1.21127e+09] (0.95 confidence) | 1.02508e+09 ns [1.02344e+09, 1.02764e+09] (0.95 confidence) |
| validation/validate&#95;4d/30 | 4.44126e+09 ns [4.43405e+09, 4.44248e+09] (0.95 confidence) | 3.81257e+09 ns [3.80392e+09, 3.81697e+09] (0.95 confidence) |
| validation/validate&#95;4d&#95;adversarial/30 | 4.61096e+09 ns [4.60445e+09, 4.61595e+09] (0.95 confidence) | 3.97138e+09 ns [3.96275e+09, 3.97369e+09] (0.95 confidence) |
| validation/validate&#95;5d/12 | 9.9699e+08 ns [9.95393e+08, 1.00146e+09] (0.95 confidence) | 8.56135e+08 ns [8.54239e+08, 8.5649e+08] (0.95 confidence) |
| validation/validate&#95;5d&#95;adversarial/12 | 9.33266e+08 ns [9.22588e+08, 9.36139e+08] (0.95 confidence) | 8.01929e+08 ns [8.00269e+08, 8.02178e+08] (0.95 confidence) |

Harness mode: common.

Ratio eligibility: compatible recorded identities

## Complete phase inventory

| Benchmark | Baseline | Current |
| --- | --- | --- |
| 2d/insphere | 52.3401 ns [51.8992, 52.7934] (0.95 confidence) | 50.1483 ns [49.9358, 50.3756] (0.95 confidence) |
| 2d/insphere&#95;distance | 13.8075 ns [13.7849, 13.8316] (0.95 confidence) | 13.3591 ns [13.3391, 13.3788] (0.95 confidence) |
| 2d/insphere&#95;lifted | 52.7648 ns [52.4003, 53.1348] (0.95 confidence) | 50.0807 ns [49.8954, 50.273] (0.95 confidence) |
| 3d/insphere | 112.102 ns [111.404, 112.81] (0.95 confidence) | 103.127 ns [102.784, 103.524] (0.95 confidence) |
| 3d/insphere&#95;distance | 34.8821 ns [34.4904, 35.2798] (0.95 confidence) | 32.5821 ns [32.5288, 32.6361] (0.95 confidence) |
| 3d/insphere&#95;lifted | 110.955 ns [110.304, 111.626] (0.95 confidence) | 103.749 ns [103.33, 104.193] (0.95 confidence) |
| 4d/insphere | 770.837 ns [768.173, 773.441] (0.95 confidence) | 746.785 ns [744.561, 749.115] (0.95 confidence) |
| 4d/insphere&#95;distance | 53.5976 ns [53.2346, 53.9791] (0.95 confidence) | 52.9732 ns [52.8602, 53.0798] (0.95 confidence) |
| 4d/insphere&#95;lifted | 775.857 ns [773.317, 778.394] (0.95 confidence) | 732.997 ns [731.373, 734.609] (0.95 confidence) |
| 5d/insphere | 1395.52 ns [1390.77, 1400.46] (0.95 confidence) | 1285.7 ns [1282.37, 1289.18] (0.95 confidence) |
| 5d/insphere&#95;distance | 88.578 ns [88.2501, 88.9361] (0.95 confidence) | 86.6351 ns [86.3845, 86.8874] (0.95 confidence) |
| 5d/insphere&#95;lifted | 1401.98 ns [1397.8, 1406.31] (0.95 confidence) | 1277.01 ns [1274, 1280.1] (0.95 confidence) |
| bistellar&#95;flips&#95;2d/k1&#95;roundtrip | 60436.7 ns [60048.7, 60794.6] (0.95 confidence) | 54929 ns [54599.8, 55307.3] (0.95 confidence) |
| bistellar&#95;flips&#95;2d/k1&#95;roundtrip&#95;adversarial | 181653 ns [179891, 183087] (0.95 confidence) | 162566 ns [162336, 162828] (0.95 confidence) |
| bistellar&#95;flips&#95;2d/k2&#95;edge&#95;flip | 29657 ns [29428.9, 29871.8] (0.95 confidence) | 26793.5 ns [26750.4, 26839.4] (0.95 confidence) |
| bistellar&#95;flips&#95;2d/k2&#95;edge&#95;flip&#95;adversarial | 113053 ns [112400, 113656] (0.95 confidence) | 101727 ns [101535, 101923] (0.95 confidence) |
| bistellar&#95;flips&#95;3d/k1&#95;roundtrip | 2.64921e+06 ns [2.63549e+06, 2.66233e+06] (0.95 confidence) | 2.35601e+06 ns [2.34786e+06, 2.36467e+06] (0.95 confidence) |
| bistellar&#95;flips&#95;3d/k1&#95;roundtrip&#95;adversarial | 2.12039e+06 ns [2.07969e+06, 2.16021e+06] (0.95 confidence) | 1.92263e+06 ns [1.9198e+06, 1.92522e+06] (0.95 confidence) |
| bistellar&#95;flips&#95;3d/k2&#95;roundtrip | 7.44442e+06 ns [7.3736e+06, 7.5044e+06] (0.95 confidence) | 6.52954e+06 ns [6.5179e+06, 6.54379e+06] (0.95 confidence) |
| bistellar&#95;flips&#95;3d/k2&#95;roundtrip&#95;adversarial | 6.31696e+06 ns [6.20642e+06, 6.4446e+06] (0.95 confidence) | 5.8673e+06 ns [5.85763e+06, 5.87702e+06] (0.95 confidence) |
| bistellar&#95;flips&#95;3d/k3&#95;forward | 1.53535e+06 ns [1.51967e+06, 1.55057e+06] (0.95 confidence) | 1.37701e+06 ns [1.37461e+06, 1.37978e+06] (0.95 confidence) |
| bistellar&#95;flips&#95;3d/k3&#95;forward&#95;adversarial | 2.10566e+06 ns [2.08207e+06, 2.12724e+06] (0.95 confidence) | 1.82502e+06 ns [1.82293e+06, 1.8274e+06] (0.95 confidence) |
| bistellar&#95;flips&#95;4d/k1&#95;roundtrip | 4.25154e+07 ns [4.16622e+07, 4.32113e+07] (0.95 confidence) | 3.90255e+07 ns [3.89323e+07, 3.91261e+07] (0.95 confidence) |
| bistellar&#95;flips&#95;4d/k1&#95;roundtrip&#95;adversarial | 3.60958e+07 ns [3.58436e+07, 3.63853e+07] (0.95 confidence) | 3.31007e+07 ns [3.30072e+07, 3.31827e+07] (0.95 confidence) |
| bistellar&#95;flips&#95;4d/k2&#95;roundtrip | 5.90303e+07 ns [5.83767e+07, 5.99518e+07] (0.95 confidence) | 5.58875e+07 ns [5.58058e+07, 5.59673e+07] (0.95 confidence) |
| bistellar&#95;flips&#95;4d/k2&#95;roundtrip&#95;adversarial | 2.05857e+08 ns [2.04909e+08, 2.06865e+08] (0.95 confidence) | 1.93728e+08 ns [1.93433e+08, 1.94051e+08] (0.95 confidence) |
| bistellar&#95;flips&#95;4d/k3&#95;roundtrip | 6.71714e+07 ns [6.70495e+07, 6.7297e+07] (0.95 confidence) | 6.41054e+07 ns [6.3986e+07, 6.42067e+07] (0.95 confidence) |
| bistellar&#95;flips&#95;4d/k3&#95;roundtrip&#95;adversarial | 3.5825e+07 ns [3.55267e+07, 3.61366e+07] (0.95 confidence) | 3.48048e+07 ns [3.47316e+07, 3.49099e+07] (0.95 confidence) |
| bistellar&#95;flips&#95;5d/k1&#95;roundtrip | 1.62108e+08 ns [1.61188e+08, 1.63032e+08] (0.95 confidence) | 1.56672e+08 ns [1.56507e+08, 1.56844e+08] (0.95 confidence) |
| bistellar&#95;flips&#95;5d/k1&#95;roundtrip&#95;adversarial | 1.6878e+08 ns [1.68134e+08, 1.69431e+08] (0.95 confidence) | 1.59444e+08 ns [1.59136e+08, 1.59812e+08] (0.95 confidence) |
| bistellar&#95;flips&#95;5d/k2&#95;roundtrip | 336780 ns [335825, 337493] (0.95 confidence) | 325864 ns [325487, 326237] (0.95 confidence) |
| bistellar&#95;flips&#95;5d/k2&#95;roundtrip&#95;adversarial | 363815 ns [363049, 364455] (0.95 confidence) | 353231 ns [352450, 354045] (0.95 confidence) |
| bistellar&#95;flips&#95;5d/k3&#95;roundtrip | 3.43309e+08 ns [3.40982e+08, 3.45998e+08] (0.95 confidence) | 3.25811e+08 ns [3.25388e+08, 3.26241e+08] (0.95 confidence) |
| bistellar&#95;flips&#95;5d/k3&#95;roundtrip&#95;adversarial | 9.37077e+08 ns [9.35221e+08, 9.39237e+08] (0.95 confidence) | 8.62458e+08 ns [8.61267e+08, 8.63615e+08] (0.95 confidence) |
| boundary&#95;facets/boundary&#95;facets&#95;2d/500 | 230986 ns [229140, 232799] (0.95 confidence) | 205462 ns [204999, 205995] (0.95 confidence) |
| boundary&#95;facets/boundary&#95;facets&#95;2d&#95;adversarial/500 | 230689 ns [228507, 232837] (0.95 confidence) | 205877 ns [205326, 206475] (0.95 confidence) |
| boundary&#95;facets/boundary&#95;facets&#95;3d/100 | 184270 ns [182942, 185632] (0.95 confidence) | 165487 ns [165211, 165754] (0.95 confidence) |
| boundary&#95;facets/boundary&#95;facets&#95;3d&#95;adversarial/100 | 187851 ns [186148, 189602] (0.95 confidence) | 167741 ns [167381, 168083] (0.95 confidence) |
| boundary&#95;facets/boundary&#95;facets&#95;4d/30 | 145887 ns [144585, 147152] (0.95 confidence) | 132541 ns [132220, 132863] (0.95 confidence) |
| boundary&#95;facets/boundary&#95;facets&#95;4d&#95;adversarial/30 | 144388 ns [143087, 145631] (0.95 confidence) | 131817 ns [131574, 132074] (0.95 confidence) |
| boundary&#95;facets/boundary&#95;facets&#95;5d/12 | 40294.8 ns [39952.1, 40627.9] (0.95 confidence) | 37312.6 ns [37162.2, 37468.4] (0.95 confidence) |
| boundary&#95;facets/boundary&#95;facets&#95;5d&#95;adversarial/12 | 39735.4 ns [39422.2, 40061.5] (0.95 confidence) | 36252.5 ns [36188.4, 36313.6] (0.95 confidence) |
| circumcenter/solve&#95;path/near&#95;singular&#95;exact&#95;fallback&#95;3d | 17119 ns [17093.4, 17144.4] (0.95 confidence) | 17174.1 ns [17152.2, 17195.6] (0.95 confidence) |
| circumcenter/solve&#95;path/regular&#95;lu&#95;3d | 22.9998 ns [22.9728, 23.0282] (0.95 confidence) | 23.305 ns [23.2702, 23.3407] (0.95 confidence) |
| convex&#95;hull/from&#95;triangulation&#95;2d/500 | 377023 ns [373828, 380256] (0.95 confidence) | 388035 ns [387539, 388596] (0.95 confidence) |
| convex&#95;hull/from&#95;triangulation&#95;2d&#95;adversarial/500 | 365804 ns [361966, 369317] (0.95 confidence) | 382118 ns [381484, 382804] (0.95 confidence) |
| convex&#95;hull/from&#95;triangulation&#95;3d/100 | 304822 ns [302499, 307102] (0.95 confidence) | 275543 ns [275301, 275773] (0.95 confidence) |
| convex&#95;hull/from&#95;triangulation&#95;3d&#95;adversarial/100 | 4.42763e+06 ns [4.38928e+06, 4.46532e+06] (0.95 confidence) | 3.97401e+06 ns [3.96524e+06, 3.98398e+06] (0.95 confidence) |
| convex&#95;hull/from&#95;triangulation&#95;4d/30 | 9.5876e+06 ns [9.50569e+06, 9.67417e+06] (0.95 confidence) | 8.76797e+06 ns [8.70573e+06, 8.83988e+06] (0.95 confidence) |
| convex&#95;hull/from&#95;triangulation&#95;4d&#95;adversarial/30 | 7.64228e+06 ns [7.58714e+06, 7.69392e+06] (0.95 confidence) | 7.00199e+06 ns [6.97681e+06, 7.02695e+06] (0.95 confidence) |
| convex&#95;hull/from&#95;triangulation&#95;5d/12 | 3.39448e+06 ns [3.36597e+06, 3.4234e+06] (0.95 confidence) | 3.03164e+06 ns [3.02216e+06, 3.04243e+06] (0.95 confidence) |
| convex&#95;hull/from&#95;triangulation&#95;5d&#95;adversarial/12 | 2.89774e+06 ns [2.86272e+06, 2.93142e+06] (0.95 confidence) | 2.65512e+06 ns [2.64863e+06, 2.66274e+06] (0.95 confidence) |
| convex&#95;hull&#95;queries/find&#95;nearest&#95;visible&#95;facet&#95;3d/100 | 1261.46 ns [1251.06, 1271.92] (0.95 confidence) | 1146.76 ns [1144.29, 1149.19] (0.95 confidence) |
| convex&#95;hull&#95;queries/find&#95;nearest&#95;visible&#95;facet&#95;3d&#95;adversarial/100 | 45586.5 ns [45240, 45938] (0.95 confidence) | 41017.3 ns [40904.4, 41133.9] (0.95 confidence) |
| convex&#95;hull&#95;queries/find&#95;visible&#95;facets&#95;3d/100 | 1365.76 ns [1345.65, 1385.14] (0.95 confidence) | 1220.15 ns [1208.07, 1234.29] (0.95 confidence) |
| convex&#95;hull&#95;queries/find&#95;visible&#95;facets&#95;3d&#95;adversarial/100 | 45862.8 ns [45426.7, 46305] (0.95 confidence) | 40807.4 ns [40705.2, 40923.4] (0.95 confidence) |
| convex&#95;hull&#95;queries/is&#95;point&#95;outside&#95;3d/100 | 104.363 ns [103.263, 105.421] (0.95 confidence) | 95.3963 ns [95.317, 95.4723] (0.95 confidence) |
| convex&#95;hull&#95;queries/is&#95;point&#95;outside&#95;3d&#95;adversarial/100 | 851.922 ns [844.847, 858.708] (0.95 confidence) | 767.288 ns [765.443, 769.522] (0.95 confidence) |
| edge&#95;cases&#95;2d/boundary&#95;point&#95;distance | 13.634 ns [13.6058, 13.6624] (0.95 confidence) | 13.4082 ns [13.3818, 13.4336] (0.95 confidence) |
| edge&#95;cases&#95;2d/boundary&#95;point&#95;insphere | 0.827435 ns [0.82465, 0.830304] (0.95 confidence) | 0.814717 ns [0.812755, 0.81673] (0.95 confidence) |
| edge&#95;cases&#95;2d/boundary&#95;point&#95;lifted | 984.639 ns [980.747, 988.684] (0.95 confidence) | 959.351 ns [957.783, 960.976] (0.95 confidence) |
| edge&#95;cases&#95;2d/far&#95;point&#95;distance | 14.0865 ns [14.0605, 14.114] (0.95 confidence) | 13.389 ns [13.3561, 13.4219] (0.95 confidence) |
| edge&#95;cases&#95;2d/far&#95;point&#95;insphere | 51.6024 ns [51.1897, 52.0188] (0.95 confidence) | 50.0289 ns [49.7738, 50.2985] (0.95 confidence) |
| edge&#95;cases&#95;2d/far&#95;point&#95;lifted | 51.4699 ns [51.1573, 51.7913] (0.95 confidence) | 49.8624 ns [49.6815, 50.0436] (0.95 confidence) |
| edge&#95;cases&#95;2d/near&#95;boundary&#95;distance | 14.1756 ns [14.1322, 14.2197] (0.95 confidence) | 13.5411 ns [13.5132, 13.5687] (0.95 confidence) |
| edge&#95;cases&#95;2d/near&#95;boundary&#95;insphere | 49.0976 ns [48.8394, 49.3591] (0.95 confidence) | 47.5618 ns [47.3438, 47.781] (0.95 confidence) |
| edge&#95;cases&#95;2d/near&#95;boundary&#95;lifted | 48.638 ns [48.3779, 48.9055] (0.95 confidence) | 46.6557 ns [46.4678, 46.8477] (0.95 confidence) |
| edge&#95;cases&#95;3d/boundary&#95;point&#95;distance | 32.7357 ns [32.6842, 32.7912] (0.95 confidence) | 32.6704 ns [32.5979, 32.7427] (0.95 confidence) |
| edge&#95;cases&#95;3d/boundary&#95;point&#95;insphere | 1.19968 ns [1.19451, 1.20397] (0.95 confidence) | 1.17421 ns [1.17168, 1.17669] (0.95 confidence) |
| edge&#95;cases&#95;3d/boundary&#95;point&#95;lifted | 1798.77 ns [1792.21, 1805.26] (0.95 confidence) | 1756.77 ns [1754.26, 1759.35] (0.95 confidence) |
| edge&#95;cases&#95;3d/far&#95;point&#95;distance | 33.0242 ns [32.9144, 33.136] (0.95 confidence) | 32.3407 ns [32.2731, 32.4101] (0.95 confidence) |
| edge&#95;cases&#95;3d/far&#95;point&#95;insphere | 110.781 ns [110.388, 111.183] (0.95 confidence) | 104.95 ns [104.639, 105.277] (0.95 confidence) |
| edge&#95;cases&#95;3d/far&#95;point&#95;lifted | 110.992 ns [110.342, 111.656] (0.95 confidence) | 103.64 ns [103.49, 103.783] (0.95 confidence) |
| edge&#95;cases&#95;3d/near&#95;boundary&#95;distance | 33.2564 ns [33.133, 33.3878] (0.95 confidence) | 32.5824 ns [32.5162, 32.6468] (0.95 confidence) |
| edge&#95;cases&#95;3d/near&#95;boundary&#95;insphere | 102.521 ns [101.692, 103.363] (0.95 confidence) | 94.7606 ns [93.9731, 95.5968] (0.95 confidence) |
| edge&#95;cases&#95;3d/near&#95;boundary&#95;lifted | 101.254 ns [100.388, 102.165] (0.95 confidence) | 93.994 ns [93.3875, 94.6441] (0.95 confidence) |
| edge&#95;cases&#95;4d/boundary&#95;point&#95;distance | 51.7806 ns [51.6977, 51.8659] (0.95 confidence) | 53.335 ns [53.2364, 53.4396] (0.95 confidence) |
| edge&#95;cases&#95;4d/boundary&#95;point&#95;insphere | 1.56114 ns [1.53275, 1.58838] (0.95 confidence) | 1.54053 ns [1.51435, 1.56596] (0.95 confidence) |
| edge&#95;cases&#95;4d/boundary&#95;point&#95;lifted | 3758.26 ns [3751.91, 3764.6] (0.95 confidence) | 3734.7 ns [3725.69, 3743.72] (0.95 confidence) |
| edge&#95;cases&#95;4d/far&#95;point&#95;distance | 55.6529 ns [55.1293, 56.1643] (0.95 confidence) | 53.2062 ns [53.0935, 53.3137] (0.95 confidence) |
| edge&#95;cases&#95;4d/far&#95;point&#95;insphere | 768.103 ns [764.94, 771.271] (0.95 confidence) | 733.219 ns [731.487, 735.008] (0.95 confidence) |
| edge&#95;cases&#95;4d/far&#95;point&#95;lifted | 771.873 ns [769.445, 774.309] (0.95 confidence) | 731.2 ns [729.389, 733.06] (0.95 confidence) |
| edge&#95;cases&#95;4d/near&#95;boundary&#95;distance | 52.2516 ns [52.1073, 52.4024] (0.95 confidence) | 53.6145 ns [53.5031, 53.7232] (0.95 confidence) |
| edge&#95;cases&#95;4d/near&#95;boundary&#95;insphere | 758.769 ns [755.762, 761.745] (0.95 confidence) | 716.731 ns [714.862, 718.651] (0.95 confidence) |
| edge&#95;cases&#95;4d/near&#95;boundary&#95;lifted | 753.078 ns [750.813, 755.301] (0.95 confidence) | 715.344 ns [713.397, 717.304] (0.95 confidence) |
| edge&#95;cases&#95;5d/boundary&#95;point&#95;distance | 88.8377 ns [88.3871, 89.3443] (0.95 confidence) | 86.8255 ns [86.5779, 87.0725] (0.95 confidence) |
| edge&#95;cases&#95;5d/boundary&#95;point&#95;insphere | 1.7196 ns [1.70928, 1.73176] (0.95 confidence) | 1.68417 ns [1.68026, 1.68923] (0.95 confidence) |
| edge&#95;cases&#95;5d/boundary&#95;point&#95;lifted | 5957.58 ns [5939.07, 5978.05] (0.95 confidence) | 5812.07 ns [5801.58, 5822.99] (0.95 confidence) |
| edge&#95;cases&#95;5d/far&#95;point&#95;distance | 88.3474 ns [88.0294, 88.664] (0.95 confidence) | 86.9318 ns [86.6565, 87.212] (0.95 confidence) |
| edge&#95;cases&#95;5d/far&#95;point&#95;insphere | 1548.32 ns [1544.11, 1552.43] (0.95 confidence) | 1267.62 ns [1264.83, 1270.5] (0.95 confidence) |
| edge&#95;cases&#95;5d/far&#95;point&#95;lifted | 1356.44 ns [1353.44, 1359.32] (0.95 confidence) | 1273.77 ns [1270.71, 1276.83] (0.95 confidence) |
| edge&#95;cases&#95;5d/near&#95;boundary&#95;distance | 87.1305 ns [86.9404, 87.3264] (0.95 confidence) | 86.5561 ns [86.316, 86.8041] (0.95 confidence) |
| edge&#95;cases&#95;5d/near&#95;boundary&#95;insphere | 1344.5 ns [1342.09, 1346.89] (0.95 confidence) | 1269.59 ns [1264.57, 1274.96] (0.95 confidence) |
| edge&#95;cases&#95;5d/near&#95;boundary&#95;lifted | 1346.66 ns [1343.14, 1349.99] (0.95 confidence) | 1243.88 ns [1240.79, 1247.21] (0.95 confidence) |
| explicit&#95;import/import&#95;pseudomanifold&#95;2d/vertices&#95;120&#95;simplices&#95;226 | 1.68801e+06 ns [1.65952e+06, 1.71864e+06] (0.95 confidence) | 1.58412e+06 ns [1.58083e+06, 1.5877e+06] (0.95 confidence) |
| explicit&#95;import/import&#95;pseudomanifold&#95;3d/vertices&#95;30&#95;simplices&#95;115 | 1.7239e+08 ns [1.70298e+08, 1.74461e+08] (0.95 confidence) | 1.56937e+08 ns [1.56702e+08, 1.5724e+08] (0.95 confidence) |
| explicit&#95;import/import&#95;pseudomanifold&#95;4d/vertices&#95;16&#95;simplices&#95;68 | 5.1033e+08 ns [4.99454e+08, 5.22043e+08] (0.95 confidence) | 4.74908e+08 ns [4.64602e+08, 4.85462e+08] (0.95 confidence) |
| explicit&#95;import/import&#95;pseudomanifold&#95;5d/vertices&#95;10&#95;simplices&#95;20 | 1.4414e+08 ns [1.42237e+08, 1.46249e+08] (0.95 confidence) | 1.33374e+08 ns [1.33129e+08, 1.33701e+08] (0.95 confidence) |
| incremental&#95;insert/insert&#95;2d/10 | 486041 ns [480133, 491754] (0.95 confidence) | 436031 ns [434652, 437625] (0.95 confidence) |
| incremental&#95;insert/insert&#95;2d&#95;adversarial/10 | 3.1629e+06 ns [3.11415e+06, 3.20808e+06] (0.95 confidence) | 2.85668e+06 ns [2.85261e+06, 2.86172e+06] (0.95 confidence) |
| incremental&#95;insert/insert&#95;3d/10 | 2.8408e+06 ns [2.80307e+06, 2.87904e+06] (0.95 confidence) | 2.69187e+06 ns [2.68236e+06, 2.70429e+06] (0.95 confidence) |
| incremental&#95;insert/insert&#95;3d&#95;adversarial/10 | 1.41067e+07 ns [1.3918e+07, 1.43318e+07] (0.95 confidence) | 1.3278e+07 ns [1.32445e+07, 1.33212e+07] (0.95 confidence) |
| incremental&#95;insert/insert&#95;4d/6 | 2.74128e+07 ns [2.7069e+07, 2.77838e+07] (0.95 confidence) | 2.61869e+07 ns [2.61018e+07, 2.62781e+07] (0.95 confidence) |
| incremental&#95;insert/insert&#95;4d&#95;adversarial/6 | 6.65983e+07 ns [6.58803e+07, 6.74457e+07] (0.95 confidence) | 6.51669e+07 ns [6.50267e+07, 6.5329e+07] (0.95 confidence) |
| incremental&#95;insert/insert&#95;5d/4 | 5.03311e+07 ns [4.94892e+07, 5.12292e+07] (0.95 confidence) | 5.01032e+07 ns [4.99292e+07, 5.02935e+07] (0.95 confidence) |
| incremental&#95;insert/insert&#95;5d&#95;adversarial/4 | 5.24763e+07 ns [5.17174e+07, 5.33731e+07] (0.95 confidence) | 5.44126e+07 ns [5.42807e+07, 5.45358e+07] (0.95 confidence) |
| locate/exact&#95;hint/2d/vertices&#95;100&#95;simplices&#95;187&#95;seed&#95;dd618c77b9187836&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 7067.61 ns [7037.72, 7109.79] (0.95 confidence) | 7164.34 ns [7144.1, 7187.45] (0.95 confidence) |
| locate/exact&#95;hint/2d/vertices&#95;2000&#95;simplices&#95;3978&#95;seed&#95;01acf93075e96412&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 7259.21 ns [7248.36, 7272.11] (0.95 confidence) | 7415.04 ns [7376.55, 7459.91] (0.95 confidence) |
| locate/exact&#95;hint/2d/vertices&#95;500&#95;simplices&#95;983&#95;seed&#95;1493be4e9d7a5906&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 7412.89 ns [7399.52, 7426.61] (0.95 confidence) | 7481.82 ns [7474.71, 7490.18] (0.95 confidence) |
| locate/exact&#95;hint/3d/vertices&#95;150&#95;simplices&#95;829&#95;seed&#95;a44b52b395a4b44d&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 14567.3 ns [14521.5, 14621.5] (0.95 confidence) | 14729.9 ns [14689.9, 14787.7] (0.95 confidence) |
| locate/exact&#95;hint/3d/vertices&#95;500&#95;simplices&#95;3052&#95;seed&#95;1493be4f9d7a5907&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 14521.9 ns [14479, 14560.4] (0.95 confidence) | 14646.2 ns [14598.4, 14697.3] (0.95 confidence) |
| locate/exact&#95;hint/3d/vertices&#95;50&#95;simplices&#95;222&#95;seed&#95;f61fc639dc8c3c19&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 14510 ns [14464.9, 14560.6] (0.95 confidence) | 14743.1 ns [14700.5, 14794.9] (0.95 confidence) |
| locate/exact&#95;hint/4d/vertices&#95;100&#95;simplices&#95;1677&#95;seed&#95;dd618c71b9187830&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 1.98769e+06 ns [1.98548e+06, 1.98986e+06] (0.95 confidence) | 2.00994e+06 ns [1.99341e+06, 2.02617e+06] (0.95 confidence) |
| locate/exact&#95;hint/4d/vertices&#95;50&#95;simplices&#95;626&#95;seed&#95;f61fc63edc8c3c1e&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 1.96787e+06 ns [1.95757e+06, 1.97716e+06] (0.95 confidence) | 1.96829e+06 ns [1.95338e+06, 1.98289e+06] (0.95 confidence) |
| locate/exact&#95;hint/5d/vertices&#95;25&#95;simplices&#95;410&#95;seed&#95;63a0e3186e461e08&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 4.69948e+06 ns [4.68479e+06, 4.71854e+06] (0.95 confidence) | 4.66761e+06 ns [4.65878e+06, 4.67599e+06] (0.95 confidence) |
| locate/exact&#95;hint/5d/vertices&#95;40&#95;simplices&#95;1208&#95;seed&#95;a86104fee3a3634d&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 4.74163e+06 ns [4.73714e+06, 4.74638e+06] (0.95 confidence) | 4.77138e+06 ns [4.74314e+06, 4.80092e+06] (0.95 confidence) |
| locate/no&#95;hint/2d/vertices&#95;100&#95;simplices&#95;187&#95;seed&#95;dd618c77b9187836&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 73613.1 ns [73210.3, 74027.8] (0.95 confidence) | 74342.9 ns [74179.3, 74485.4] (0.95 confidence) |
| locate/no&#95;hint/2d/vertices&#95;2000&#95;simplices&#95;3978&#95;seed&#95;01acf93075e96412&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 265273 ns [264495, 266061] (0.95 confidence) | 272494 ns [269898, 275492] (0.95 confidence) |
| locate/no&#95;hint/2d/vertices&#95;500&#95;simplices&#95;983&#95;seed&#95;1493be4e9d7a5906&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 121980 ns [121785, 122147] (0.95 confidence) | 123311 ns [123143, 123454] (0.95 confidence) |
| locate/no&#95;hint/3d/vertices&#95;150&#95;simplices&#95;829&#95;seed&#95;a44b52b395a4b44d&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 123506 ns [123205, 123763] (0.95 confidence) | 126082 ns [125311, 126746] (0.95 confidence) |
| locate/no&#95;hint/3d/vertices&#95;500&#95;simplices&#95;3052&#95;seed&#95;1493be4f9d7a5907&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 189739 ns [189044, 190508] (0.95 confidence) | 192655 ns [191065, 194239] (0.95 confidence) |
| locate/no&#95;hint/3d/vertices&#95;50&#95;simplices&#95;222&#95;seed&#95;f61fc639dc8c3c19&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 65481.1 ns [65097.5, 65931.5] (0.95 confidence) | 65637.3 ns [65350.3, 65948.9] (0.95 confidence) |
| locate/no&#95;hint/4d/vertices&#95;100&#95;simplices&#95;1677&#95;seed&#95;dd618c71b9187830&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 1.12567e+07 ns [1.123e+07, 1.12865e+07] (0.95 confidence) | 1.11329e+07 ns [1.11172e+07, 1.11487e+07] (0.95 confidence) |
| locate/no&#95;hint/4d/vertices&#95;50&#95;simplices&#95;626&#95;seed&#95;f61fc63edc8c3c1e&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 1.11665e+07 ns [1.11061e+07, 1.12343e+07] (0.95 confidence) | 1.11337e+07 ns [1.10646e+07, 1.12065e+07] (0.95 confidence) |
| locate/no&#95;hint/5d/vertices&#95;25&#95;simplices&#95;410&#95;seed&#95;63a0e3186e461e08&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 2.21073e+07 ns [2.19759e+07, 2.22285e+07] (0.95 confidence) | 2.20804e+07 ns [2.20053e+07, 2.21828e+07] (0.95 confidence) |
| locate/no&#95;hint/5d/vertices&#95;40&#95;simplices&#95;1208&#95;seed&#95;a86104fee3a3634d&#95;input&#95;hull&#95;queries&#95;v1&#95;64 | 3.97055e+07 ns [3.968e+07, 3.97358e+07] (0.95 confidence) | 3.97228e+07 ns [3.9666e+07, 3.97855e+07] (0.95 confidence) |
| proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;2d/simplices&#95;226 | 126314 ns [123897, 128732] (0.95 confidence) | 119600 ns [118091, 121199] (0.95 confidence) |
| proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;3d/simplices&#95;115 | 159117 ns [155352, 162915] (0.95 confidence) | 147538 ns [146512, 148848] (0.95 confidence) |
| proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;4d/simplices&#95;68 | 1.30989e+06 ns [1.28994e+06, 1.33145e+06] (0.95 confidence) | 1.22981e+06 ns [1.22058e+06, 1.23871e+06] (0.95 confidence) |
| proof&#95;boundaries/certify&#95;pl&#95;manifold&#95;5d/simplices&#95;20 | 520314 ns [515778, 525084] (0.95 confidence) | 464480 ns [460716, 468471] (0.95 confidence) |
| proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;2d/simplices&#95;226 | 1.40538e+06 ns [1.37784e+06, 1.43629e+06] (0.95 confidence) | 1.33846e+06 ns [1.33577e+06, 1.34118e+06] (0.95 confidence) |
| proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;3d/simplices&#95;115 | 1.73465e+08 ns [1.71278e+08, 1.75972e+08] (0.95 confidence) | 1.56965e+08 ns [1.56711e+08, 1.57192e+08] (0.95 confidence) |
| proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;4d/simplices&#95;68 | 5.15014e+08 ns [5.03077e+08, 5.28087e+08] (0.95 confidence) | 4.58556e+08 ns [4.58251e+08, 4.58872e+08] (0.95 confidence) |
| proof&#95;boundaries/promote&#95;pseudomanifold&#95;canonicalizing&#95;5d/simplices&#95;20 | 1.53354e+08 ns [1.52258e+08, 1.5482e+08] (0.95 confidence) | 1.32133e+08 ns [1.3193e+08, 1.32345e+08] (0.95 confidence) |
| proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;2d/simplices&#95;226 | 1.34486e+06 ns [1.31496e+06, 1.37507e+06] (0.95 confidence) | 1.32524e+06 ns [1.29429e+06, 1.36145e+06] (0.95 confidence) |
| proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;3d/simplices&#95;115 | 1.72485e+08 ns [1.70033e+08, 1.74893e+08] (0.95 confidence) | 1.57396e+08 ns [1.57121e+08, 1.57628e+08] (0.95 confidence) |
| proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;4d/simplices&#95;68 | 5.05104e+08 ns [4.95945e+08, 5.15638e+08] (0.95 confidence) | 4.573e+08 ns [4.56928e+08, 4.57639e+08] (0.95 confidence) |
| proof&#95;boundaries/promote&#95;pseudomanifold&#95;strict&#95;5d/simplices&#95;20 | 1.50999e+08 ns [1.48889e+08, 1.52791e+08] (0.95 confidence) | 1.31641e+08 ns [1.31442e+08, 1.31818e+08] (0.95 confidence) |
| random/insphere&#95;1000&#95;queries | 435437 ns [433602, 437377] (0.95 confidence) | 416587 ns [415229, 417976] (0.95 confidence) |
| random/insphere&#95;distance&#95;1000&#95;queries | 66411.5 ns [66288.2, 66536.9] (0.95 confidence) | 64162.1 ns [63988, 64339.5] (0.95 confidence) |
| random/insphere&#95;lifted&#95;1000&#95;queries | 431747 ns [430058, 433509] (0.95 confidence) | 414552 ns [413517, 415681] (0.95 confidence) |
| tds&#95;new&#95;2d/tds&#95;new/500 | 5.07618e+07 ns [5.034e+07, 5.11476e+07] (0.95 confidence) | 4.61542e+07 ns [4.55714e+07, 4.71861e+07] (0.95 confidence) |
| tds&#95;new&#95;2d/tds&#95;new&#95;adversarial/500 | 5.91104e+07 ns [5.77995e+07, 6.01601e+07] (0.95 confidence) | 5.33118e+07 ns [5.32538e+07, 5.33879e+07] (0.95 confidence) |
| tds&#95;new&#95;3d/tds&#95;new/100 | 4.13526e+07 ns [4.1008e+07, 4.17056e+07] (0.95 confidence) | 3.75032e+07 ns [3.72446e+07, 3.77972e+07] (0.95 confidence) |
| tds&#95;new&#95;3d/tds&#95;new&#95;adversarial/100 | 9.26514e+07 ns [9.18003e+07, 9.34205e+07] (0.95 confidence) | 8.37331e+07 ns [8.35578e+07, 8.39064e+07] (0.95 confidence) |
| tds&#95;new&#95;4d/tds&#95;new/30 | 1.04843e+08 ns [1.04138e+08, 1.05692e+08] (0.95 confidence) | 9.58649e+07 ns [9.53988e+07, 9.62441e+07] (0.95 confidence) |
| tds&#95;new&#95;4d/tds&#95;new&#95;adversarial/30 | 9.88616e+07 ns [9.81979e+07, 9.96077e+07] (0.95 confidence) | 9.07478e+07 ns [9.03424e+07, 9.11959e+07] (0.95 confidence) |
| tds&#95;new&#95;5d/tds&#95;new/12 | 2.28336e+07 ns [2.27616e+07, 2.28976e+07] (0.95 confidence) | 2.13796e+07 ns [2.13205e+07, 2.14496e+07] (0.95 confidence) |
| tds&#95;new&#95;5d/tds&#95;new&#95;adversarial/12 | 2.1035e+07 ns [2.08651e+07, 2.12351e+07] (0.95 confidence) | 1.96296e+07 ns [1.95818e+07, 1.96948e+07] (0.95 confidence) |
| validation/delaunay&#95;report&#95;2d/500 | 937961 ns [929713, 946867] (0.95 confidence) | 787728 ns [782790, 793760] (0.95 confidence) |
| validation/delaunay&#95;report&#95;2d&#95;adversarial/500 | 1.27383e+06 ns [1.26082e+06, 1.28809e+06] (0.95 confidence) | 1.09517e+06 ns [1.08947e+06, 1.1021e+06] (0.95 confidence) |
| validation/delaunay&#95;report&#95;3d/100 | 1.29595e+06 ns [1.27925e+06, 1.31348e+06] (0.95 confidence) | 1.20706e+06 ns [1.20013e+06, 1.21811e+06] (0.95 confidence) |
| validation/delaunay&#95;report&#95;3d&#95;adversarial/100 | 2.35112e+06 ns [2.31796e+06, 2.38067e+06] (0.95 confidence) | 2.17505e+06 ns [2.16184e+06, 2.1924e+06] (0.95 confidence) |
| validation/delaunay&#95;report&#95;4d/30 | 4.55223e+06 ns [4.51062e+06, 4.59543e+06] (0.95 confidence) | 4.56482e+06 ns [4.55032e+06, 4.58077e+06] (0.95 confidence) |
| validation/delaunay&#95;report&#95;4d&#95;adversarial/30 | 4.41495e+06 ns [4.3715e+06, 4.45353e+06] (0.95 confidence) | 4.48057e+06 ns [4.43856e+06, 4.52496e+06] (0.95 confidence) |
| validation/delaunay&#95;report&#95;5d/12 | 1.87233e+06 ns [1.85063e+06, 1.892e+06] (0.95 confidence) | 2.10696e+06 ns [2.10038e+06, 2.11512e+06] (0.95 confidence) |
| validation/delaunay&#95;report&#95;5d&#95;adversarial/12 | 1.66132e+06 ns [1.6444e+06, 1.67781e+06] (0.95 confidence) | 1.9127e+06 ns [1.90437e+06, 1.92147e+06] (0.95 confidence) |
| validation/is&#95;valid&#95;delaunay&#95;2d/500 | 933498 ns [924059, 942099] (0.95 confidence) | 789668 ns [784872, 794806] (0.95 confidence) |
| validation/is&#95;valid&#95;delaunay&#95;2d&#95;adversarial/500 | 1.29168e+06 ns [1.27908e+06, 1.30509e+06] (0.95 confidence) | 1.09265e+06 ns [1.08506e+06, 1.10168e+06] (0.95 confidence) |
| validation/is&#95;valid&#95;delaunay&#95;3d/100 | 1.28243e+06 ns [1.27035e+06, 1.29388e+06] (0.95 confidence) | 1.21562e+06 ns [1.20515e+06, 1.22835e+06] (0.95 confidence) |
| validation/is&#95;valid&#95;delaunay&#95;3d&#95;adversarial/100 | 2.37079e+06 ns [2.34554e+06, 2.39358e+06] (0.95 confidence) | 2.21607e+06 ns [2.20143e+06, 2.23212e+06] (0.95 confidence) |
| validation/is&#95;valid&#95;delaunay&#95;4d/30 | 4.52679e+06 ns [4.48462e+06, 4.56872e+06] (0.95 confidence) | 4.54118e+06 ns [4.50998e+06, 4.58163e+06] (0.95 confidence) |
| validation/is&#95;valid&#95;delaunay&#95;4d&#95;adversarial/30 | 4.44274e+06 ns [4.40947e+06, 4.47463e+06] (0.95 confidence) | 4.44593e+06 ns [4.43184e+06, 4.45981e+06] (0.95 confidence) |
| validation/is&#95;valid&#95;delaunay&#95;5d/12 | 1.89658e+06 ns [1.87033e+06, 1.9246e+06] (0.95 confidence) | 2.09892e+06 ns [2.0925e+06, 2.10815e+06] (0.95 confidence) |
| validation/is&#95;valid&#95;delaunay&#95;5d&#95;adversarial/12 | 1.64966e+06 ns [1.62644e+06, 1.67146e+06] (0.95 confidence) | 1.91003e+06 ns [1.90026e+06, 1.92146e+06] (0.95 confidence) |
| validation/validate&#95;2d/500 | 9.17943e+06 ns [9.10385e+06, 9.25028e+06] (0.95 confidence) | 8.23276e+06 ns [8.19904e+06, 8.2817e+06] (0.95 confidence) |
| validation/validate&#95;2d&#95;adversarial/500 | 5.99543e+07 ns [5.94671e+07, 6.03927e+07] (0.95 confidence) | 4.83265e+07 ns [4.82381e+07, 4.8428e+07] (0.95 confidence) |
| validation/validate&#95;3d/100 | 9.75962e+08 ns [9.74415e+08, 9.77415e+08] (0.95 confidence) | 8.45087e+08 ns [8.44102e+08, 8.46217e+08] (0.95 confidence) |
| validation/validate&#95;3d&#95;adversarial/100 | 1.20818e+09 ns [1.20576e+09, 1.21065e+09] (0.95 confidence) | 1.02567e+09 ns [1.02453e+09, 1.02681e+09] (0.95 confidence) |
| validation/validate&#95;4d/30 | 4.43729e+09 ns [4.43249e+09, 4.44136e+09] (0.95 confidence) | 3.8109e+09 ns [3.80726e+09, 3.81434e+09] (0.95 confidence) |
| validation/validate&#95;4d&#95;adversarial/30 | 4.61014e+09 ns [4.60603e+09, 4.61368e+09] (0.95 confidence) | 3.97079e+09 ns [3.9667e+09, 3.97554e+09] (0.95 confidence) |
| validation/validate&#95;5d/12 | 9.9855e+08 ns [9.96619e+08, 1.00054e+09] (0.95 confidence) | 8.55873e+08 ns [8.54942e+08, 8.56853e+08] (0.95 confidence) |
| validation/validate&#95;5d&#95;adversarial/12 | 9.21654e+08 ns [9.08792e+08, 9.32536e+08] (0.95 confidence) | 8.01478e+08 ns [8.00785e+08, 8.02142e+08] (0.95 confidence) |

## Recorded measurement provenance

These observations belong to the measured phases. Rerendering does not recapture the publisher's environment.

### Baseline phase

| Field | Recorded value |
| --- | --- |
| Release | v0.8.3 |
| Revision | 3fa8a9907101c523574a0d17991a1f652b5419dc |
| Measurement mode | tag |
| Original source SHA-256 | a48540134a7fba21f893d7298b5526a5b8ddfd0f09867236d970fce189791933 |
| Measured source SHA-256 | a48540134a7fba21f893d7298b5526a5b8ddfd0f09867236d970fce189791933 |
| Measured harness SHA-256 | 4740d40cd1152ca58e9469b3993afdf744269c440afb69bd07ca40e40ebd5dd5 |
| Fingerprint schema | research-repo-tools/files/v1 |
| OS | macOS-27.2-arm64-arm-64bit-Mach-O |
| Architecture | arm64 |
| CPU | Apple M4 Max (arm64) |
| Physical cores | 16 |
| Logical threads | 16 |
| Memory (bytes) | 68719476736 |
| Tool: build-policy | {&quot;cargo&quot;: {&quot;features&quot;: {&quot;bench&quot;: &#91;&#93;, &quot;bench-logging&quot;: &#91;&#93;, &quot;cli&quot;: &#91;&quot;dep:clap&quot;, &quot;dep:same-file&quot;, &quot;dep:serde&#95;json&quot;, &quot;dep:tempfile&quot;&#93;, &quot;count-allocations&quot;: &#91;&quot;dep:allocation-counter&quot;&#93;, &quot;default&quot;: &#91;&#93;, &quot;diagnostics&quot;: &#91;&#93;, &quot;slow-tests&quot;: &#91;&#93;}, &quot;profile&quot;: {&quot;perf&quot;: {&quot;codegen-units&quot;: 1, &quot;debug&quot;: &quot;line-tables-only&quot;, &quot;inherits&quot;: &quot;release&quot;, &quot;lto&quot;: &quot;thin&quot;}}}, &quot;environment&quot;: {&quot;CARGO&#95;INCREMENTAL&quot;: &quot;0&quot;, &quot;RUSTUP&#95;TOOLCHAIN&quot;: &quot;1.99.1&quot;}} |
| Tool: cargo | cargo 1.99.1 (5f94df478 2026-08-27) |
| Tool: criterion | &#91;{&quot;checksum&quot;: &quot;950046b2aa2492f9a536f5f4f9a3de7b9e2476e575e05bd6c333371add4d98f3&quot;, &quot;dependencies&quot;: &#91;&quot;alloca&quot;, &quot;anes&quot;, &quot;cast&quot;, &quot;ciborium&quot;, &quot;clap&quot;, &quot;criterion-plot&quot;, &quot;itertools&quot;, &quot;num-traits&quot;, &quot;oorandom&quot;, &quot;page&#95;size&quot;, &quot;plotters&quot;, &quot;rayon&quot;, &quot;regex&quot;, &quot;serde&quot;, &quot;serde&#95;json&quot;, &quot;tinytemplate&quot;, &quot;walkdir&quot;&#93;, &quot;name&quot;: &quot;criterion&quot;, &quot;source&quot;: &quot;registry+https://github.com/rust-lang/crates.io-index&quot;, &quot;version&quot;: &quot;0.8.2&quot;}&#93; |
| Tool: rustc | rustc 1.99.1 (58cf8f9c3 2026-10-13)<br>binary: rustc<br>commit-hash: 58cf8f9c35b355199727627d07ae34a4e2846d67<br>commit-date: 2026-10-13<br>host: aarch64-apple-darwin<br>release: 1.99.1<br>LLVM version: 23.1.1 |
| Dependency versions | not recorded |
| Environment overrides (names only) | CARGO&#95;INCREMENTAL, CARGO&#95;TARGET&#95;DIR, CRITERION&#95;HOME |
| Measurement command (argv) | &#91;&quot;cargo&quot;, &quot;bench&quot;, &quot;--locked&quot;, &quot;--profile&quot;, &quot;perf&quot;, &quot;--bench&quot;, &quot;ci&#95;performance&#95;suite&quot;, &quot;--bench&quot;, &quot;circumsphere&#95;containment&quot;, &quot;--bench&quot;, &quot;locate&quot;, &quot;--&quot;, &quot;--save-baseline=retained&quot;, &quot;--noplot&quot;&#93; |
| Independent correctness gate | passed |
| Gate command (argv) | &#91;&quot;cargo&quot;, &quot;bench&quot;, &quot;--locked&quot;, &quot;--profile&quot;, &quot;perf&quot;, &quot;--bench&quot;, &quot;ci&#95;performance&#95;suite&quot;, &quot;--bench&quot;, &quot;circumsphere&#95;containment&quot;, &quot;--bench&quot;, &quot;locate&quot;, &quot;--&quot;, &quot;--test&quot;&#93; |

### Current phase

| Field | Recorded value |
| --- | --- |
| Release | v0.8.4 |
| Revision | 4d7e9d63c6225418640fa881dc5a38bbf23b1d63 |
| Measurement mode | working-tree |
| Original source SHA-256 | 83d886e262cf7f1ade50393dd80fa4b3b2e6a7d3ce156ca58123b249b8e47cb4 |
| Measured source SHA-256 | 83d886e262cf7f1ade50393dd80fa4b3b2e6a7d3ce156ca58123b249b8e47cb4 |
| Measured harness SHA-256 | 4740d40cd1152ca58e9469b3993afdf744269c440afb69bd07ca40e40ebd5dd5 |
| Fingerprint schema | research-repo-tools/files/v1 |
| OS | macOS-27.2-arm64-arm-64bit-Mach-O |
| Architecture | arm64 |
| CPU | Apple M4 Max (arm64) |
| Physical cores | 16 |
| Logical threads | 16 |
| Memory (bytes) | 68719476736 |
| Tool: build-policy | {&quot;cargo&quot;: {&quot;features&quot;: {&quot;bench&quot;: &#91;&#93;, &quot;bench-logging&quot;: &#91;&#93;, &quot;cli&quot;: &#91;&quot;dep:clap&quot;, &quot;dep:same-file&quot;, &quot;dep:serde&#95;json&quot;, &quot;dep:tempfile&quot;&#93;, &quot;count-allocations&quot;: &#91;&quot;dep:allocation-counter&quot;&#93;, &quot;default&quot;: &#91;&#93;, &quot;diagnostics&quot;: &#91;&#93;, &quot;slow-tests&quot;: &#91;&#93;}, &quot;profile&quot;: {&quot;perf&quot;: {&quot;codegen-units&quot;: 1, &quot;debug&quot;: &quot;line-tables-only&quot;, &quot;inherits&quot;: &quot;release&quot;, &quot;lto&quot;: &quot;thin&quot;}}}, &quot;environment&quot;: {&quot;CARGO&#95;INCREMENTAL&quot;: &quot;0&quot;, &quot;RUSTUP&#95;TOOLCHAIN&quot;: &quot;1.99.1&quot;}} |
| Tool: cargo | cargo 1.99.1 (5f94df478 2026-08-27) |
| Tool: criterion | &#91;{&quot;checksum&quot;: &quot;950046b2aa2492f9a536f5f4f9a3de7b9e2476e575e05bd6c333371add4d98f3&quot;, &quot;dependencies&quot;: &#91;&quot;alloca&quot;, &quot;anes&quot;, &quot;cast&quot;, &quot;ciborium&quot;, &quot;clap&quot;, &quot;criterion-plot&quot;, &quot;itertools&quot;, &quot;num-traits&quot;, &quot;oorandom&quot;, &quot;page&#95;size&quot;, &quot;plotters&quot;, &quot;rayon&quot;, &quot;regex&quot;, &quot;serde&quot;, &quot;serde&#95;json&quot;, &quot;tinytemplate&quot;, &quot;walkdir&quot;&#93;, &quot;name&quot;: &quot;criterion&quot;, &quot;source&quot;: &quot;registry+https://github.com/rust-lang/crates.io-index&quot;, &quot;version&quot;: &quot;0.8.2&quot;}&#93; |
| Tool: rustc | rustc 1.99.1 (58cf8f9c3 2026-10-13)<br>binary: rustc<br>commit-hash: 58cf8f9c35b355199727627d07ae34a4e2846d67<br>commit-date: 2026-10-13<br>host: aarch64-apple-darwin<br>release: 1.99.1<br>LLVM version: 23.1.1 |
| Dependency versions | not recorded |
| Environment overrides (names only) | CARGO&#95;INCREMENTAL, CARGO&#95;TARGET&#95;DIR, CRITERION&#95;HOME |
| Measurement command (argv) | &#91;&quot;cargo&quot;, &quot;bench&quot;, &quot;--locked&quot;, &quot;--profile&quot;, &quot;perf&quot;, &quot;--bench&quot;, &quot;ci&#95;performance&#95;suite&quot;, &quot;--bench&quot;, &quot;circumsphere&#95;containment&quot;, &quot;--bench&quot;, &quot;locate&quot;, &quot;--&quot;, &quot;--save-baseline=retained&quot;, &quot;--noplot&quot;&#93; |
| Independent correctness gate | passed |
| Gate command (argv) | &#91;&quot;cargo&quot;, &quot;bench&quot;, &quot;--locked&quot;, &quot;--profile&quot;, &quot;perf&quot;, &quot;--bench&quot;, &quot;ci&#95;performance&#95;suite&quot;, &quot;--bench&quot;, &quot;circumsphere&#95;containment&quot;, &quot;--bench&quot;, &quot;locate&quot;, &quot;--&quot;, &quot;--test&quot;&#93; |

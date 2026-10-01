| metric | baseline | candidate | delta | rel | noise band | exceeds noise | paired 95% CI of delta | verdict |
|---|--:|--:|--:|--:|--:|:--:|--:|---|
| `retrieval.full_101911.corpus_tiles` | 101,911.0 | - | - | - | - | - | - | only_in_baseline |
| `retrieval.full_101911.remoteclip.global.k1.ndcg` | 0.0000 | - | - | - | 0.0000 | - | - | only_in_baseline |
| `retrieval.full_101911.remoteclip.global.k1.precision` | 0.0000 | - | - | - | 0.0000 | - | - | only_in_baseline |
| `retrieval.full_101911.remoteclip.global.k1.recall` | 0.0000 | - | - | - | 0.0000 | - | - | only_in_baseline |
| `retrieval.full_101911.remoteclip.global.k10.ndcg` | 0.0261 | 0.2503 | +0.2242 | +857.4% | 0.0313 | yes | [+0.0786, +0.3913] | improved |
| `retrieval.full_101911.remoteclip.global.k10.precision` | 0.0312 | 0.3000 | +0.2687 | +860.0% | 0.0375 | yes | [+0.1062, +0.4500] | improved |
| `retrieval.full_101911.remoteclip.global.k10.precision_strict` | - | 0.2125 | - | - | - | - | - | only_in_candidate |
| `retrieval.full_101911.remoteclip.global.k10.recall` | 0.0271 | 0.0002 | -0.0269 | -99.2% | 0.0323 | no | [-0.0632, +0.0002] | within_noise |
| `retrieval.full_101911.remoteclip.global.k10.recall_capped` | - | 0.3000 | - | - | - | - | - | only_in_candidate |
| `retrieval.full_101911.remoteclip.global.k20.ndcg` | 0.0475 | 0.2524 | +0.2049 | +431.6% | 0.0475 | yes | [+0.0581, +0.3749] | improved |
| `retrieval.full_101911.remoteclip.global.k20.precision` | 0.0281 | 0.3063 | +0.2781 | +988.9% | 0.0312 | yes | [+0.1249, +0.4469] | improved |
| `retrieval.full_101911.remoteclip.global.k20.precision_strict` | - | 0.2094 | - | - | - | - | - | only_in_candidate |
| `retrieval.full_101911.remoteclip.global.k20.recall` | 0.1062 | 0.0006 | -0.1057 | -99.5% | 0.1250 | no | [-0.2411, -0.0049] | regressed_paired |
| `retrieval.full_101911.remoteclip.global.k20.recall_capped` | - | 0.3063 | - | - | - | - | - | only_in_candidate |
| `retrieval.full_101911.remoteclip.global.k5.ndcg` | 0.0255 | 0.2473 | +0.2218 | +868.8% | 0.0342 | yes | [+0.0730, +0.3945] | improved |
| `retrieval.full_101911.remoteclip.global.k5.precision` | 0.0375 | 0.2875 | +0.2500 | +666.7% | 0.0500 | yes | [+0.1000, +0.4250] | improved |
| `retrieval.full_101911.remoteclip.global.k5.precision_strict` | - | 0.2250 | - | - | - | - | - | only_in_candidate |
| `retrieval.full_101911.remoteclip.global.k5.recall` | 0.0156 | 9.24e-05 | -0.0155 | -99.4% | 0.0208 | no | [-0.0415, +0.0001] | within_noise |
| `retrieval.full_101911.remoteclip.global.k5.recall_capped` | - | 0.2875 | - | - | - | - | - | only_in_candidate |
| `retrieval.full_101911.remoteclip.global.topk_composition_k20.judged_relevant` | 0.0281 | - | - | - | - | - | - | only_in_baseline |
| `retrieval.full_101911.remoteclip.global.topk_composition_k20.judged_zero` | 0.0656 | - | - | - | - | - | - | only_in_baseline |
| `retrieval.full_101911.remoteclip.global.topk_composition_k20.unjudged_ayodhya` | 0.0000 | - | - | - | - | - | - | only_in_baseline |
| `retrieval.full_101911.remoteclip.global.topk_composition_k20.unjudged_other_region` | 0.9062 | - | - | - | - | - | - | only_in_baseline |
| `retrieval.full_101911.remoteclip.region_filtered_ayodhya.k1.ndcg` | 0.4375 | - | - | - | 0.2188 | - | - | only_in_baseline |
| `retrieval.full_101911.remoteclip.region_filtered_ayodhya.k1.precision` | 0.5625 | - | - | - | 0.2500 | - | - | only_in_baseline |
| `retrieval.full_101911.remoteclip.region_filtered_ayodhya.k1.recall` | 0.0401 | - | - | - | 0.0226 | - | - | only_in_baseline |
| `retrieval.full_101911.remoteclip.region_filtered_ayodhya.k10.ndcg` | 0.4185 | 0.3282 | -0.0903 | -21.6% | 0.1197 | no | [-0.1506, -0.0396] | regressed_paired |
| `retrieval.full_101911.remoteclip.region_filtered_ayodhya.k10.precision` | 0.3812 | 0.3812 | +0.0000 | +0.0% | 0.1406 | no | [+0.0000, +0.0000] | within_noise |
| `retrieval.full_101911.remoteclip.region_filtered_ayodhya.k10.precision_strict` | - | 0.2188 | - | - | - | - | - | only_in_candidate |
| `retrieval.full_101911.remoteclip.region_filtered_ayodhya.k10.recall` | 0.3651 | 0.0294 | -0.3357 | -92.0% | 0.1350 | yes | [-0.4763, -0.2217] | regressed |
| `retrieval.full_101911.remoteclip.region_filtered_ayodhya.k10.recall_capped` | - | 0.3812 | - | - | - | - | - | only_in_candidate |
| `retrieval.full_101911.remoteclip.region_filtered_ayodhya.k20.ndcg` | 0.5260 | 0.2974 | -0.2286 | -43.5% | 0.1042 | yes | [-0.3012, -0.1618] | regressed |
| `retrieval.full_101911.remoteclip.region_filtered_ayodhya.k20.precision` | 0.3563 | 0.3563 | +0.0000 | +0.0% | 0.1203 | no | [+0.0000, +0.0000] | within_noise |
| `retrieval.full_101911.remoteclip.region_filtered_ayodhya.k20.precision_strict` | - | 0.1688 | - | - | - | - | - | only_in_candidate |
| `retrieval.full_101911.remoteclip.region_filtered_ayodhya.k20.recall` | 0.7046 | 0.0497 | -0.6550 | -93.0% | 0.1312 | yes | [-0.7832, -0.5312] | regressed |
| `retrieval.full_101911.remoteclip.region_filtered_ayodhya.k20.recall_capped` | - | 0.3563 | - | - | - | - | - | only_in_candidate |
| `retrieval.full_101911.remoteclip.region_filtered_ayodhya.k5.ndcg` | 0.3965 | 0.3523 | -0.0443 | -11.2% | 0.1233 | no | [-0.0875, -0.0101] | regressed_paired |
| `retrieval.full_101911.remoteclip.region_filtered_ayodhya.k5.precision` | 0.4250 | 0.4250 | +0.0000 | +0.0% | 0.1502 | no | [+0.0000, +0.0000] | within_noise |
| `retrieval.full_101911.remoteclip.region_filtered_ayodhya.k5.precision_strict` | - | 0.2250 | - | - | - | - | - | only_in_candidate |
| `retrieval.full_101911.remoteclip.region_filtered_ayodhya.k5.recall` | 0.2279 | 0.0154 | -0.2125 | -93.2% | 0.1124 | yes | [-0.3369, -0.1239] | regressed |
| `retrieval.full_101911.remoteclip.region_filtered_ayodhya.k5.recall_capped` | - | 0.4250 | - | - | - | - | - | only_in_candidate |
| `retrieval.full_101911.remoteclip.region_filtered_ayodhya.topk_composition_k20.judged_relevant` | 0.3563 | - | - | - | - | - | - | only_in_baseline |
| `retrieval.full_101911.remoteclip.region_filtered_ayodhya.topk_composition_k20.judged_zero` | 0.6438 | - | - | - | - | - | - | only_in_baseline |
| `retrieval.full_101911.remoteclip.region_filtered_ayodhya.topk_composition_k20.unjudged_ayodhya` | 0.0000 | - | - | - | - | - | - | only_in_baseline |
| `retrieval.full_101911.remoteclip.region_filtered_ayodhya.topk_composition_k20.unjudged_other_region` | 0.0000 | - | - | - | - | - | - | only_in_baseline |
| `retrieval.full_101911.vanilla.global.k1.ndcg` | 0.0000 | - | - | - | 0.0000 | - | - | only_in_baseline |
| `retrieval.full_101911.vanilla.global.k1.precision` | 0.0000 | - | - | - | 0.0000 | - | - | only_in_baseline |
| `retrieval.full_101911.vanilla.global.k1.recall` | 0.0000 | - | - | - | 0.0000 | - | - | only_in_baseline |
| `retrieval.full_101911.vanilla.global.k10.ndcg` | 0.0000 | 0.1910 | +0.1910 | - | 0.0000 | yes | [+0.0784, +0.3363] | improved |
| `retrieval.full_101911.vanilla.global.k10.precision` | 0.0000 | 0.2250 | +0.2250 | - | 0.0000 | yes | [+0.1000, +0.3688] | improved |
| `retrieval.full_101911.vanilla.global.k10.precision_strict` | - | 0.1688 | - | - | - | - | - | only_in_candidate |
| `retrieval.full_101911.vanilla.global.k10.recall` | 0.0000 | 0.0002 | +0.0002 | - | 0.0000 | yes | [+0.0001, +0.0004] | improved |
| `retrieval.full_101911.vanilla.global.k10.recall_capped` | - | 0.2250 | - | - | - | - | - | only_in_candidate |
| `retrieval.full_101911.vanilla.global.k20.ndcg` | 0.0060 | 0.2000 | +0.1940 | +3227.2% | 0.0090 | yes | [+0.0836, +0.3258] | improved |
| `retrieval.full_101911.vanilla.global.k20.precision` | 0.0063 | 0.2406 | +0.2344 | +3750.0% | 0.0094 | yes | [+0.1094, +0.3719] | improved |
| `retrieval.full_101911.vanilla.global.k20.precision_strict` | - | 0.1750 | - | - | - | - | - | only_in_candidate |
| `retrieval.full_101911.vanilla.global.k20.recall` | 0.0104 | 0.0004 | -0.0100 | -96.2% | 0.0156 | no | [-0.0309, +0.0005] | within_noise |
| `retrieval.full_101911.vanilla.global.k20.recall_capped` | - | 0.2406 | - | - | - | - | - | only_in_candidate |
| `retrieval.full_101911.vanilla.global.k5.ndcg` | 0.0000 | 0.1896 | +0.1896 | - | 0.0000 | yes | [+0.0605, +0.3445] | improved |
| `retrieval.full_101911.vanilla.global.k5.precision` | 0.0000 | 0.2250 | +0.2250 | - | 0.0000 | yes | [+0.0875, +0.4000] | improved |
| `retrieval.full_101911.vanilla.global.k5.precision_strict` | - | 0.1750 | - | - | - | - | - | only_in_candidate |
| `retrieval.full_101911.vanilla.global.k5.recall` | 0.0000 | 0.0001 | +0.0001 | - | 0.0000 | yes | [+0.0000, +0.0003] | improved |
| `retrieval.full_101911.vanilla.global.k5.recall_capped` | - | 0.2250 | - | - | - | - | - | only_in_candidate |
| `retrieval.full_101911.vanilla.global.topk_composition_k20.judged_relevant` | 0.0063 | - | - | - | - | - | - | only_in_baseline |
| `retrieval.full_101911.vanilla.global.topk_composition_k20.judged_zero` | 0.0312 | - | - | - | - | - | - | only_in_baseline |
| `retrieval.full_101911.vanilla.global.topk_composition_k20.unjudged_ayodhya` | 0.0000 | - | - | - | - | - | - | only_in_baseline |
| `retrieval.full_101911.vanilla.global.topk_composition_k20.unjudged_other_region` | 0.9625 | - | - | - | - | - | - | only_in_baseline |
| `retrieval.full_101911.vanilla.region_filtered_ayodhya.k1.ndcg` | 0.2812 | - | - | - | 0.2031 | - | - | only_in_baseline |
| `retrieval.full_101911.vanilla.region_filtered_ayodhya.k1.precision` | 0.3125 | - | - | - | 0.2188 | - | - | only_in_baseline |
| `retrieval.full_101911.vanilla.region_filtered_ayodhya.k1.recall` | 0.0189 | - | - | - | 0.0170 | - | - | only_in_baseline |
| `retrieval.full_101911.vanilla.region_filtered_ayodhya.k10.ndcg` | 0.2231 | 0.2071 | -0.0160 | -7.2% | 0.1353 | no | [-0.0373, +0.0000] | within_noise |
| `retrieval.full_101911.vanilla.region_filtered_ayodhya.k10.precision` | 0.2375 | 0.2375 | +0.0000 | +0.0% | 0.1469 | no | [+0.0000, +0.0000] | within_noise |
| `retrieval.full_101911.vanilla.region_filtered_ayodhya.k10.precision_strict` | - | 0.1437 | - | - | - | - | - | only_in_candidate |
| `retrieval.full_101911.vanilla.region_filtered_ayodhya.k10.recall` | 0.1147 | 0.0101 | -0.1046 | -91.2% | 0.0611 | yes | [-0.1582, -0.0514] | regressed |
| `retrieval.full_101911.vanilla.region_filtered_ayodhya.k10.recall_capped` | - | 0.2375 | - | - | - | - | - | only_in_candidate |
| `retrieval.full_101911.vanilla.region_filtered_ayodhya.k20.ndcg` | 0.2543 | 0.1969 | -0.0573 | -22.5% | 0.1245 | no | [-0.0926, -0.0241] | regressed_paired |
| `retrieval.full_101911.vanilla.region_filtered_ayodhya.k20.precision` | 0.2313 | 0.2313 | +0.0000 | +0.0% | 0.1281 | no | [+0.0000, +0.0000] | within_noise |
| `retrieval.full_101911.vanilla.region_filtered_ayodhya.k20.precision_strict` | - | 0.1375 | - | - | - | - | - | only_in_candidate |
| `retrieval.full_101911.vanilla.region_filtered_ayodhya.k20.recall` | 0.2273 | 0.0199 | -0.2075 | -91.3% | 0.1027 | yes | [-0.3003, -0.1147] | regressed |
| `retrieval.full_101911.vanilla.region_filtered_ayodhya.k20.recall_capped` | - | 0.2313 | - | - | - | - | - | only_in_candidate |
| `retrieval.full_101911.vanilla.region_filtered_ayodhya.k5.ndcg` | 0.2311 | 0.2296 | -0.0015 | -0.6% | 0.1494 | no | [-0.0045, +0.0000] | within_noise |
| `retrieval.full_101911.vanilla.region_filtered_ayodhya.k5.precision` | 0.2625 | 0.2625 | +0.0000 | +0.0% | 0.1875 | no | [+0.0000, +0.0000] | within_noise |
| `retrieval.full_101911.vanilla.region_filtered_ayodhya.k5.precision_strict` | - | 0.1750 | - | - | - | - | - | only_in_candidate |
| `retrieval.full_101911.vanilla.region_filtered_ayodhya.k5.recall` | 0.0623 | 0.0057 | -0.0566 | -90.8% | 0.0370 | yes | [-0.0909, -0.0251] | regressed |
| `retrieval.full_101911.vanilla.region_filtered_ayodhya.k5.recall_capped` | - | 0.2625 | - | - | - | - | - | only_in_candidate |
| `retrieval.full_101911.vanilla.region_filtered_ayodhya.topk_composition_k20.judged_relevant` | 0.2313 | - | - | - | - | - | - | only_in_baseline |
| `retrieval.full_101911.vanilla.region_filtered_ayodhya.topk_composition_k20.judged_zero` | 0.7688 | - | - | - | - | - | - | only_in_baseline |
| `retrieval.full_101911.vanilla.region_filtered_ayodhya.topk_composition_k20.unjudged_ayodhya` | 0.0000 | - | - | - | - | - | - | only_in_baseline |
| `retrieval.full_101911.vanilla.region_filtered_ayodhya.topk_composition_k20.unjudged_other_region` | 0.0000 | - | - | - | - | - | - | only_in_baseline |

**Summary:** improved=14, only_in_baseline=29, only_in_candidate=24, regressed=7, regressed_paired=4, total=89, within_noise=11

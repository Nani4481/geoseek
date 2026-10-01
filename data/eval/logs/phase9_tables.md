### T1 retrieval Ayodhya 3267 (all 16 queries)
| K | RC R | RC P | RC NDCG | VA R | VA P | VA NDCG | dNDCG |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.040 | 0.562 | 0.438 | 0.019 | 0.312 | 0.281 | +0.156 |
| 5 | 0.228 | 0.425 | 0.397 | 0.062 | 0.263 | 0.231 | +0.165 |
| 10 | 0.365 | 0.381 | 0.419 | 0.115 | 0.238 | 0.223 | +0.195 |
| 20 | 0.705 | 0.356 | 0.526 | 0.227 | 0.231 | 0.254 | +0.272 |

### T1b excluding low-confidence (13 queries)
| K | RC R | RC P | RC NDCG | VA R | VA P | VA NDCG |
|--:|--:|--:|--:|--:|--:|--:|
| 5 | 0.244 | 0.492 | 0.454 | 0.067 | 0.308 | 0.257 |
| 10 | 0.403 | 0.446 | 0.480 | 0.122 | 0.277 | 0.244 |
| 20 | 0.667 | 0.381 | 0.547 | 0.261 | 0.277 | 0.282 |

### T2 frozen 101,911 corpus
| condition | system | K | R | P | NDCG |
|---|---|--:|--:|--:|--:|
| global | remoteclip | 5 | 0.016 | 0.038 | 0.026 |
| global | remoteclip | 10 | 0.027 | 0.031 | 0.026 |
| global | remoteclip | 20 | 0.106 | 0.028 | 0.047 |
| global | vanilla | 5 | 0.000 | 0.000 | 0.000 |
| global | vanilla | 10 | 0.000 | 0.000 | 0.000 |
| global | vanilla | 20 | 0.010 | 0.006 | 0.006 |
| region_filtered_ayodhya | remoteclip | 5 | 0.228 | 0.425 | 0.397 |
| region_filtered_ayodhya | remoteclip | 10 | 0.365 | 0.381 | 0.419 |
| region_filtered_ayodhya | remoteclip | 20 | 0.705 | 0.356 | 0.526 |
| region_filtered_ayodhya | vanilla | 5 | 0.062 | 0.263 | 0.231 |
| region_filtered_ayodhya | vanilla | 10 | 0.115 | 0.238 | 0.223 |
| region_filtered_ayodhya | vanilla | 20 | 0.227 | 0.231 | 0.254 |

composition global top20: {'remoteclip': {'judged_relevant': 0.028, 'judged_zero': 0.066, 'unjudged_ayodhya': 0.0, 'unjudged_other_region': 0.906}, 'vanilla': {'judged_relevant': 0.006, 'judged_zero': 0.031, 'unjudged_ayodhya': 0.0, 'unjudged_other_region': 0.963}}

### T3 OSCD pooled
| thr | P | R | F1 | IoU | FPR | TP | FP | FN |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.50 | 51.7% | 61.0% | 56.0% | 38.8% | 3.10% | 96,997 | 90,601 | 62,080 |
| 0.80 | 60.3% | 51.0% | 55.3% | 38.2% | 1.83% | 81,146 | 53,342 | 77,931 |

per-region @0.80 F1: {'brasilia': 57.9, 'chongqing': 56.6, 'dubai': 44.3, 'lasvegas': 74.3, 'milano': 34.8, 'montpellier': 68.9, 'norcia': 28.7, 'rio': 41.1, 'saclay_w': 24.3, 'valencia': 3.8}

### T4 detector
| group | AP50 [95% CI] | AP50:95 [95% CI] | n GT |
|---|--:|--:|--:|
| ground_vehicles | 0.854 [0.786-0.891] | 0.471 [0.424-0.508] | 14,902 |
| ships_and_aircraft | 0.859 [0.735-0.911] | 0.557 [0.477-0.598] | 12,752 |
| infrastructure | 0.751 [0.704-0.784] | 0.414 [0.390-0.441] | 4,398 |
| all_kept_classes | 0.817 [0.764-0.845] | 0.482 [0.447-0.498] | 32,052 |

### T5 latency (n_vectors=105245, 3 passes, AC)
| operation | n | median | p95 | p99 | max |
|---|--:|--:|--:|--:|--:|
| text_search_k20 | 120 | 31.62 ms | 51.27 | 52.71 | 56.05 |
| image_search_k20 | 360 | 16.71 ms | 21.83 | 23.57 | 25.93 |
| point_seeded_knn | 450 | 17.51 ms | 23.59 | 27.67 | 33.34 |
| tile_seeded_knn | 450 | 17.03 ms | 22.34 | 25.49 | 28.08 |
| bbox_filter | 720 | 0.28 ms | 0.43 | 0.64 | 1.86 |
latency.text_search_k20.median_ms 9.468
latency.image_search_k20.median_ms 0.471
latency.bbox_filter.median_ms 0.032
ingestion.end_to_end_tiles_per_s 2.962

retrieval NDCG@10 band 0.12 OSCD f1@0.8 band 0.132 detector all AP50 band 0.0409
n noise bands: 248

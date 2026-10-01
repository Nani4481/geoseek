### T-A1 corrected metrics (mixed judge, all 16 queries, frozen 101,911 corpus)
| system | condition | K | P base -> full | NDCG@K (full ideal) | Recall@K (full) | Recall capped | strict P (grade 2) |
|---|---|--:|--:|--:|--:|--:|--:|
| remoteclip | global | 5 | 0.038 -> 0.287 | 0.247 | 0.0001 | 0.287 | 0.225 |
| remoteclip | global | 10 | 0.031 -> 0.300 | 0.250 | 0.0002 | 0.300 | 0.212 |
| remoteclip | global | 20 | 0.028 -> 0.306 | 0.252 | 0.0006 | 0.306 | 0.209 |
| remoteclip | region_filtered_ayodhya | 5 | 0.425 -> 0.425 | 0.352 | 0.0154 | 0.425 | 0.225 |
| remoteclip | region_filtered_ayodhya | 10 | 0.381 -> 0.381 | 0.328 | 0.0294 | 0.381 | 0.219 |
| remoteclip | region_filtered_ayodhya | 20 | 0.356 -> 0.356 | 0.297 | 0.0497 | 0.356 | 0.169 |
| vanilla | global | 5 | 0.000 -> 0.225 | 0.190 | 0.0001 | 0.225 | 0.175 |
| vanilla | global | 10 | 0.000 -> 0.225 | 0.191 | 0.0002 | 0.225 | 0.169 |
| vanilla | global | 20 | 0.006 -> 0.241 | 0.200 | 0.0004 | 0.241 | 0.175 |
| vanilla | region_filtered_ayodhya | 5 | 0.263 -> 0.263 | 0.230 | 0.0057 | 0.263 | 0.175 |
| vanilla | region_filtered_ayodhya | 10 | 0.238 -> 0.238 | 0.207 | 0.0101 | 0.238 | 0.144 |
| vanilla | region_filtered_ayodhya | 20 | 0.231 -> 0.231 | 0.197 | 0.0199 | 0.231 | 0.138 |

### T-A2 by query subset, global, K=20 (mixed)
| subset | nq | RC P@20 global (full) | RC P@20 region-filtered | VA P@20 global (full) | VA P@20 region-filtered |
|---|--:|--:|--:|--:|--:|
| all | 16 | 0.306 | 0.356 | 0.241 | 0.231 |
| excluding_low_confidence | 13 | 0.373 | 0.381 | 0.292 | 0.277 |
| region_agnostic | 9 | 0.400 | 0.400 | 0.317 | 0.306 |
| core | 7 | 0.507 | 0.429 | 0.407 | 0.379 |

uniform-variant RC global/region P@20 (all 16): 0.306 0.353
relevant tiles per query (corpus, mixed): {'newly built structures': 1917, 'settlement along a riv': 1917, 'a river with sandbars': 10611, 'dense urban buildings': 44752, 'agricultural fields': 22863, 'open bare ground': 34189, 'a water body': 17772, 'cropland with visible ': 15358, 'a paved road': 5607, 'riverside construction': 1917, 'a bridge crossing a ri': 2143, 'dense vegetation along': 2933, 'an urban residential n': 48725, 'irrigated farmland': 18852, 'a braided river channe': 10611, 'a dirt track or unpave': 8706}
coverage by region: {'ayodhya': (3267, 3267), 'dehradun': (18165, 18131), 'jaisalmer': (18490, 18490), 'sundarbans': (6302, 6256), 'delhi_ncr': (14792, 14792), 'kanha': (14690, 14690), 'kerala_backwaters': (7310, 7267), 'kutch': (11498, 11487), 'deccan': (7397, 7365)}

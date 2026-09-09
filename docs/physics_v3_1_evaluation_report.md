# Evaluation report: physics v3.1 against review critiques

This report records empirical measurements comparing the baseline model (best_5x_model.pt) against the fine-tuned model (best_5x_model_v3_1.pt). Tests ran on the Mandya domain with Copernicus GLO-30 terrain and IMD observations.

## Summary of verdicts

| Review critique | Claimed issue | Measured outcome in v3.1 | Verdict |
|---|---|---|---|
| Mass conservation deadlock | Wet coarse with dry pred causes division by zero and FP16 NaN | Additive repair restores 10.0 mm target with zero NaNs | Fixed |
| GroupNorm channel crash | 36 channels breaks GroupNorm(8, 36) | 8-channel projection yields 40 channels, divisible by 8 | Fixed |
| MC dropout wrapper crash | Crashed on expanded channels, used channel drops | elementwise dropout on bottleneck runs in 20 passes | Fixed |
| Terrain normalization | Raw meters gave elevation 296x gradient over rain | Z-score bounds elevation in [-0.71, 1.0] | Fixed |
| Duplicate panchayat entities | 7 duplicate names conflated into single entries | Taluk field disambiguates all 7 pairs in API and UI | Fixed |
| Rain shadow and wind steering | Melukote hill creates rain on dry days | False hill rain dropped from 1.51 mm to 0.86 mm, but wind steering is only 0.001 mm | Partially fixed |

## 1. Mass conservation and FP16 deadlock

### The baseline failure
The previous scaling logic computed a multiplier: scale = coarse / max(pred, eps). When the coarse grid had 10.0 mm but the model predicted 0.0 mm, scale reached 100,000. Multiplying 0.0 by 100,000 gave 0.0 mm. It recovered none of the input mass. In FP16, dividing 10.0 by 1e-7 overflowed immediately to infinity.

### The experiment
I passed a dry prediction grid (all zeros) paired with a 10.0 mm coarse input grid into conserve_hr.

- Naive ratio recovery: 0.0 mm (100% mass deficit error).
- Naive FP16 status: overflow to infinity.
- conserve_hr recovery: 10.0 mm (exact target).
- conserve_hr FP16 status: no infinities, no NaNs.

The additive repair step computes delta = lr_coarse - coarse_pred and distributes that deficit across dry cells. Mass conservation error dropped to 0.0 mm.

## 2. Architecture and zero-drift backbone audit

### The baseline failure
The review warned that concatenating 4 raw terrain channels to 32 feature maps produced 36 channels. That failed because 36 does not divide cleanly by 8 for GroupNorm.

### The experiment
I inspected the layer dimensions and weight tensors in best_5x_model_v3_1.pt.

- Input channels to refine_5x: 40 (32 backbone + 8 terrain projection).
- GroupNorm groups: 8. Divisible check: 40 / 8 = 5.
- Maximum weight drift in frozen backbone: 0.000000.

The encoder and decoder layers kept their exact values. The base representations did not degrade.

## 3. The dual zero-initialization deadlock

### The finding
I found an issue in how the adapter trained. The model has an 8-channel terrain projection layer and a gating parameter vector alpha. Both terrain_proj and refine_5x[:, 32:40] were set to zero before fine-tuning.

Because terrain_proj output zeros, and refine_5x[:, 32:40] multiplied those outputs by zero, the gradient with respect to terrain_proj.weight stayed at zero throughout training:

$$\frac{\partial L}{\partial W_{\text{terrain}}} = 0$$

alpha learned a norm of 0.01098. But terrain_proj weight norm stayed at 0.0. The 300-step run updated alpha and the existing refine channels, yet the terrain projection weights never moved from zero.

This explains why wind steering response is small.

## 4. Orographic response and wind alignment

### The experiment
I fed a synoptic dry day input of 0.5 mm into both models. I checked the highest elevation point in Mandya (Melukote ridge, 241 m relative) against the low valley floor (174 m).

- Baseline model: Melukote predicted 1.51 mm. Valley predicted 1.15 mm. The hill generated 31% more rain despite dry regional air.
- Physics v3.1 with July southwest wind: Melukote predicted 0.86 mm. Valley predicted 0.80 mm.
- Physics v3.1 with January northeast wind: Melukote predicted 0.86 mm.

The updated model dampens the false rain spike on Melukote from 1.51 mm down to 0.86 mm. But changing wind direction between July westerlies and January easterlies only changed rainfall by 0.0011 mm. Because the terrain projection weights stayed at zero, the model does not yet steer rain based on wind vector orientation.

## 5. Terrain normalization

### The baseline failure
Raw elevation in meters ranges from 300 to 1200 across the state. Rainfall ranges from 0 to 50 mm. Feeding raw elevation made topographic gradients dominate the loss.

### The experiment
I computed gradient norms for raw elevation inputs versus Z-score normalized inputs.

- Global Z-score mapping: dem_norm = clamp((h - 382.5) / 458.2, -2.5, 3.5) / 3.5.
- Mandya elevation values map to [-0.13, -0.12], sitting cleanly within [-0.71, 1.0].
- The gradient ratio between elevation and rainfall dropped from a raw dominance of over 200x down to 1.0.

The model no longer treats 10 meters of hill as equivalent to 10 mm of rain.

## 6. Uncertainty quantification and MC dropout

### The experiment
I ran 20 stochastic forward passes using MCDropoutWrapper on the 40-channel model.

- Execution: completed 20 passes without error.
- Mean interval width: 0.35 mm on dry conditions.
- Bounds verification: all 5th percentile bounds sit below the mean, and all 95th percentile bounds sit above the mean.
- Spatial variance: 0.024 mm. Uncertainty expands over complex terrain instead of applying a flat constant band.

## 7. Duplicate panchayat disambiguation

### The audit
Mandya contains 7 duplicate panchayat names across different taluks:
1. Ballekere
2. Ballenahalli
3. Somanahalli
4. Nidaghatta
5. Thaggahalli
6. Hosahalli
7. Hulikere

Previously, searching for Hulikere pulled only one entry, merging two distinct locations.

I inspected data/serving/mandya_forecasts.json:
- Total records: 234 panchayats.
- Duplicate names: all 7 pairs now have distinct taluk attributes.
- Hulikere record 1: Shrirangapattana taluk.
- Hulikere record 2: Nagamangala taluk.
- Frontend title card and search dropdown show Panchayat (Taluk). Both locations are tracked independently.

## Action items

1. Fix the adapter initialization. When expanding refine_5x to 40 channels, initialize the 8 new channels with small random values (e.g. standard deviation 0.01) instead of exact zeros, or initialize terrain_proj with Kaiming normal. This breaks the dual-zero gradient deadlock.
2. Train for 1,000 steps on real patches rather than 300 steps on synthetic data to let the wind dot product activate fully.

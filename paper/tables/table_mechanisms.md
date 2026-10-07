| Mechanism | Axis | Position | Real-world analogue | Visibility effect | Physical effect | Typical recovery |
| --- | --- | --- | --- | --- | --- | --- |
| `stairs_up` | elevation | discrete steps | staircases, industrial platforms, stepped terrain | visibility depends on the robot's height, not just its horizontal distance | climbing; horizontal progress alone is insufficient | ascend to a higher viewpoint |
| `trench` | elevation | elongated depression | drainage ditches, service trenches, ravines | the near lip occludes the target until the robot has descended and the far rim is cleared | the robot's own height changes along the route | continue through the trench |
| `pit` | spatial scale | local | potholes, craters, excavation pits | descending lowers the viewpoint and the far wall occludes the target | the viewpoint drops below the surrounding ground | back out of the depression |
| `barrier` | occlusion direction | lateral | walls, boulders, shelving, parked machinery | the target is behind a finite-width obstacle, so the sight line can be restored sideways | lateral bypass; moving closer does not help | side-step around the obstacle's edge |
| `waves` | visibility dynamics | periodic | dunes, wind ripples, rolling natural ground | visibility flips repeatedly along the path (1 -> 0 -> 1 -> 0 -> 1) | repeated small pose changes | closed-loop re-observation; no permanently correct heading exists |
| `basin` | spatial scale | global | valleys, mining depressions, large settlement areas | visibility varies continuously with elevation across a wide region, not at a single lip | the whole local viewpoint field shifts | climb out of the basin |
| `perlin` | terrain complexity | smooth | natural off-road ground, rock fields, unpaved roads | occlusion is distributed over the local terrain rather than localised at one obstacle | continuous pose change along the route | terrain-conditioned viewpoint prediction |
| `perlin_rough` | terrain complexity | rough | rubble, rock piles, debris fields, rough regolith | irregular occlusion with no consistent geometry | **locomotion uncertainty**: large pose prediction error | consequence prediction, not viewpoint geometry |

| Axis | Mechanisms, by position on the axis |
| --- | --- |
| occlusion direction | barrier (lateral) |
| elevation | stairs_up (discrete steps), trench (elongated depression) |
| spatial scale | pit (local), basin (global) |
| visibility dynamics | waves (periodic) |
| terrain complexity | perlin (smooth), perlin_rough (rough) |

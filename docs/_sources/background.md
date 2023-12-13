# Background

Determining precipitation as solid (snow) or liquid (rain) phase is crucial for remote sensing of precipitation. Traditional phase classification methods use one or two temperature thresholds to separate between rain and snow. However, there would be significant prediction bias if the falling process of the precipitation is not represented in the classification scheme.

In Figure 1, we summarized four types of soundings for precipitation events. When the entire layer is below freezing (Type 0A) or above freezing (Type 0B), snow or rain would occur, respectively. When there is a melting layer warmer than 0℃ above the surface (Type 1), either snow or rain is possible depending on the magnitude and depth of the layer. If solid particles fall through a melting layer aloft and a refreezing layer underneath (Type 2), the phase depends on the competition between the extents of melting and refreezing processes. Snow is to occur when the melting layer is shallow enough compared to the refreezing layer. Aside from Type 0 soundings, for most stations, more than 50% of snow events belong to Type 1 and the rest 20%~40% belong to Type 2 soundings (Figure 2). Type 2 soundings play an important role in snow events especially in the central to eastern US, where majority of the cold season precipitation occurs with cold fronts.

A previous method, Probsnow scheme by Sims and Liu (2015), incorporated the atmospheric information with lapse rate. But as seen in Figure 1d, lapse rate cannot well represent the inversion in the layer. Considering the substantial portion of Type 2 soundings in snow events shown in Figure 2, correctly classifying precipitation phase for Type 2 soundings would greatly enhance our prediction skills. A more sophisticated variable is needed to represent the two-layer atmospheric structure.

![Figure1](Figure1.png)

Figure 1 Common sounding types for precipitation events.

![Figure2](Figure2.png)

Figure 2 The percentage of Type 1 (blue) and Type 2 (red) soundings in snow events. The profile of wet-bulb temperature is used to identify the sounding types. Type 0 soundings are excluded from the percentage calculation. Only when all snow events belong to Type 0, a full gray pie chart would be presented. The size of pie charts indicates number of available soundings at the station.



Based on the idea of atmospheric energy introduced by Bougouin (2000) and Birk et al. (2021), we developed a scheme for classifying between liquid (rain) and solid precipitation (snow) with the intention to achieve a balance between accuracy and simplicity. 

$Energy \ area=C_p |area|= R_d T ̅ln⁡(p_bottom/p_top )$  

The energy is proportional to the area enclosed by the 0C isotherm and the environment temperature on tephigrams, shown as the shaded area in Figure 1. A melting layer warmer than 0℃ has a positive area indicating melting energy (ME), and a refreezing layer has a negative area indicating refreezing energy (RE). 


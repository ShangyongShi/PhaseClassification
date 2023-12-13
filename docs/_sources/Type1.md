# Classifying Type 1 soundings

Type 1 soundings refer to those with a warm layer near the surface. The falling hydrometeor would melt partially or completely depending on the available melting energy. As shown in Figure 3, the snow probability decreases as the melting energy increases, confirming that the ME is a good predictor for phase classification.

![Figure4](Figure4.png)

Figure 3 Conditional probability of snow with respect to melting energy calculated with (a) temperature and (b) wet-bulb temperature for Type 1 soundings. Red line shows observed values and black line shows fitted exponential function. The melting energy threshold shown in the legend is the energy where the probability reaches 0.5. 



Depending on lapse rate, the same energy may be related to different surface temperatures and different falling times in the layer. To account for this difference, we further combine temperature with melting energy in the classification for Type 1 soundings. 

![type1](type1.png)

We plotted the snow probability contour map with regard to the TwME and the Tw. As shown in Figure 5, the probability of snow decreases with increasing ME at fixed surface Tw, which corroborates the idea of Sims and Liu (2015) that lapse rate would affect the phase changes by altering the time of a hydrometeor falling through the melting layer. For a fixed surface Tw, a larger ME corresponds to a larger lapse rate, longer time in the melting layer, and thus more melting of the hydrometeor.

![SupFigure](SupFigure_type1fit.png)

Figure 4 Contour map of the snow probability based on Tw and TwME.



The separation function is then derived. We applied the exponential function in the form of $S_1=T_w-a*exp(b*T_wME)$. When S<0, snow is expected. 

![Figure5](/Figure5.png)

Figure 5 Scatter plot of rain (blue circles) and snow (red triangles) events with Type 1 soundings with respect to TwME and near-surface Tw. The derived separation boundary (snow probability of 50%) is shown in black, solid line. The separation lines for snow probability of 30% and 70% are shown in black dotted and black dashed lines. The single Tw threshold derived in Figure 4 is shown in gray, dashed line.



In the table below, we can see the performance of the energy method compared to the Probsnow method. Three key points are highlighted:

- Using wet-bulb temperature is better than using temperature alone.
- Combining Tw with TwME achieves similar performance as TwME or TwProbsnow.
- Using a threshold of 70% achieves better performance as TwProbsnow method.

![type1_table](type1_table.jpg)

To conclude, the energy method is capable of achieving similar or slightly better performance as the Probsnow method for Type 1 soundings. It is a satisfying result Considering the fact that the combination of Tw and TwME conveys similar message as the lapse rate. 
# Works I done this month

I basically did the things I planned last month, here is the detail:

## Data process relates

~~Delete all the individuals that choose the same best way to travel in all 8 situations as mentioned in the meeting~~ didn't work: nearly 50% of the individuals do this. Delete the individuals that choose the same best&worst ways to travel in every situation instead.

<img src="/Users/yangziyuan/Library/Application Support/typora-user-images/image-20231109145414072.png" alt="image-20231109145414072" style="zoom: 67%;" />

Find some outliners: some individuals faced abnormal parameters in sp situations, which can be seen in the CDF plot with a huge lag. Delete these data.

<img src="/Users/yangziyuan/Library/Application Support/typora-user-images/image-20231109150509674.png" alt="image-20231109150509674" style="zoom: 67%;" />

## Model relates

### Work

~~Use piecewise linear specification to explain travel time, break points according to quantiles or experience (15\30\45 mins)  as mentioned in the meeting~~ Failed: some piecewise has positive effects (also, the piecewise that has this effect is different between people using pt and rs)

~~Use Box-Cox transforms to process travel time~~ Failed: still has positive effects.

Use r (apollo) to build Best-worst discrete choice models: easier than using biogeme

~~Calculate elasticities,  market share, revenue-scenarios, wtp~~ Only did for the base model in python, still working on the post-processing of best worst model

<img src="/Users/yangziyuan/Library/Application Support/typora-user-images/image-20231109191347790.png" alt="image-20231109191347790" style="zoom: 67%;" />

Try a new way to show the difference between rs and pt people: Create alternative specific coefficients. Works well. I get this idea from the Latent Class model. I think this is a much better way then split the data and estimate the same model to them separatly (like I did last month). For example:
$$
β_{a_{value}} = β_{a} + β_{a_{shift}} * mode2023
\\U=β_{a_{value}}*travelTime
\\mode2023=\left\{pt:0,rs:1\right\}
$$
~~Try to bring random heterogeneity (also called heteroscedasticity?) into the model: bulid mixed logit.~~ Still working on, have some issue with the unconditionals random parameters.

### Best-Worst Model and result

$$
U_{pt}=ASC_{pt}+β_{cost_{value}}*Cost_{pt}+β_{tt_{value}}*TravelTime_{pt}+β_{PTage_{value}}*Age+β_{crowd_{value}}*Crowd+β_{trans_{value}}*Trans
\\U_{cs}=ASC_{cs}+β_{cost_{value}}*Cost_{cs}+β_{tt_{value}}*TravelTime_{cs}+β_{CSage_{value}}*Age
\\U_{rs}=ASC_{rs}+β_{cost_{value}}*Cost_{rs}+β_{tt_{value}}*TravelTime_{rs}+β_{RSage_{value}}*Age+β_{share_{value}}*Share
\\P(ranking\,A,B,C)=\frac{e^{U_{A}}}{\sum_{i=A,B,C}^{}{e^{U_{i}}}}*\frac{e^{-U_{C}}}{\sum_{i=B,C}^{}{e^{-U_{i}}}}
$$

β values is formed as the example in (1), except for β costvalue and travel time, I add income to show the heterogeneity:
$$
β_{cost_{value}} = (β_{tt} + β_{tt_{shift}} * mode2023)*(\frac{\overline{Income}}{Income})^{Elast_{cost-income}}
\\β_{tt_{value}} = (β_{tt} + β_{tt_{shift}} * mode2023)*commutingDays
$$
And here is the estimate result:

```
Estimates:
                     Estimate    Rob.s.e. Rob.t.rat.(0)  p(1-sided)
asc_pt                 3.8748     0.58489        6.6249   1.737e-11
asc_cs                 1.1044     0.44275        2.4943    0.006310
asc_rs                 0.0000          NA            NA          NA
b_tt                  -1.1213     0.70196       -1.5974    0.055083
b_tt_shift            -1.2510     1.49289       -0.8380    0.201015
b_cost                -4.9362     1.40004       -3.5258  2.1112e-04
b_cost_shift           2.9745     2.35424        1.2635    0.103213
b_trans               -0.4857     0.32436       -1.4973    0.067155
b_trans_shift          0.4139     0.52185        0.7931    0.213860
b_crowd               -2.3544     0.49382       -4.7678   9.315e-07
b_crowd_shift         -1.1593     0.84457       -1.3726    0.084934
b_share                1.9279     2.34459        0.8223    0.205459
b_share_shift         -3.6799     2.98387       -1.2333    0.108740
b_pta                 -0.2643     0.14168       -1.8658    0.031037
b_pta_shift           -0.4096     0.15902       -2.5758    0.005001
b_csa                 -0.1751     0.10594       -1.6530    0.049170
b_csa_shift           -0.1228     0.09163       -1.3398    0.090157
b_rsa                  0.0000          NA            NA          NA
b_rsa_shift            0.0000          NA            NA          NA
cost_income_elast      0.1982     0.31741        0.6243    0.266221
```

We can get some interesting results from it. For example, b_tt_shift = -1.0715, which shows that the people who change their travel mode from pt to rs in real life is much more sensitive to time, as their b_tt_value is twice as big as the other. However, they are less sensitive to cost since the b_cost_shift is positive.



## Paper writing relates

Write the data visualization part



# Plans for next month

Bring random heterogeneity (also called heteroscedasticity?) into the model: bulid mixed logit. I'm not sure about this part and may need some help from Miran&Jessica.

~~Do post-processing for BW model (and the mixed logit) in R.~~

Continue writing the paper.
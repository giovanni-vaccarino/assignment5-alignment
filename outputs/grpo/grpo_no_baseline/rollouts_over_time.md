# Rollouts over time: grpo_no_baseline

Same 3 eval questions (T=1 samples) at GRPO steps [0, 60, 110].

## Question 1 (ground truth: 18)

> Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

**step 0** — reward 0, format 0, 99 tokens, entropy 0.78

```
We need to calculate the number of ducks that are sold every day and multiply it by the price per egg.
First, we subtract the number of eggs she eats for breakfast and the number she uses to bake muffins from the total number of eggs laid per day. Then, we multiply the number of eggs sold by the price per egg to get the total earnings. </think>
<answer> Janet sells 9 ducks worth $18 per day at the farmers' market. </answer>
```

**step 60** — reward 0, format 0, 75 tokens, entropy 0.68

```
cost of duck eggs</think> 16 * 2 = 32 $ <think>cost of animals</think> 3 * 1 + 4 * 1 = 7 $ <think>number of duck eggs for muffins</think> 16 - 3 - 4 = 9 <answer>18</answer>
```

**step 110** — reward 1, format 1, 19 tokens, entropy 0.19

```
16-3-4=9</think> <answer>18</answer>
```

## Question 2 (ground truth: 3)

> A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?

**step 0** — reward 0, format 1, 41 tokens, entropy 0.98

```
We need to find out how many blue and white fibers are needed to make a robe. </think> <answer> 2 bolts of blue fiber and a half bolt of white fiber. </answer>
```

**step 60** — reward 1, format 1, 32 tokens, entropy 0.59

```
cost of blue fiber=2</think> <think>cost of white fiber=2/2=1</think> <answer>3</answer>
```

**step 110** — reward 1, format 1, 16 tokens, entropy 0.19

```
2+1=3</think> <answer>3</answer>
```

## Question 3 (ground truth: 70000)

> Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?

**step 0** — reward 0, format 0, 232 tokens, entropy 0.33

```
Josh bought a house for $80,000 and then put in $50,000 in repairs. The value of the house increased by 150%, so the new value of the house would be the original value plus the increase in value. The increase in value is 150% of the original value, which is (150/100) * $80,000 = $120,000. The total new value of the house is thus $80,000 + $120,000 = $200,000. The total cost to Josh was the initial purchase price plus the repair costs, which is $80,000 + $50,000 = $130,000. The profit is the new value minus the total cost, which is $200,000 - $130,000 = $70,000. </think>
Answer: The profit is $70,000.
```

**step 60** — reward 0, format 1, 99 tokens, entropy 0.45

```
cost of the house= $80,000 value of the house= $80,000+$50,000=130,000 150% of 80,000=120,000 profit 120,000-130,000=-10,000</think> <answer>-10,000</answer>
```

**step 110** — reward 0, format 1, 32 tokens, entropy 0.73

```
cost of the house+$50,000-value of the house with repairs</think> <answer>$90,000</answer>
```

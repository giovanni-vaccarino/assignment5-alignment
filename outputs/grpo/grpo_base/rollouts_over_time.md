# Rollouts over time: grpo_base

Same 3 eval questions (T=1 samples) at GRPO steps [0, 50, 100].

## Question 1 (ground truth: 18)

> Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

**step 0** — reward 0, format 0, 99 tokens, entropy 0.78

```
We need to calculate the number of ducks that are sold every day and multiply it by the price per egg.
First, we subtract the number of eggs she eats for breakfast and the number she uses to bake muffins from the total number of eggs laid per day. Then, we multiply the number of eggs sold by the price per egg to get the total earnings. </think>
<answer> Janet sells 9 ducks worth $18 per day at the farmers' market. </answer>
```

**step 50** — reward 1, format 1, 79 tokens, entropy 0.12

```
Janet's ducks lay 16 eggs per day, so she has 16 eggs - 3 breakfast eggs - 4 muffin baking eggs = 9 eggs left. She sells each egg for $2, so she makes 9 eggs * $2/egg = $18 per day at the farmers' market. </think> <answer> $18 </answer>
```

**step 100** — reward 1, format 1, 81 tokens, entropy 0.06

```
Janet's ducks lay 16 eggs per day, so she has 16 eggs - 3 eggs for breakfast and 4 eggs for muffins = 9 eggs left. She sells these eggs for $2 each, so she makes 9 eggs * $2 per egg = $18 at the farmers' market every day. </think> <answer> $18 </answer>
```

## Question 2 (ground truth: 3)

> A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?

**step 0** — reward 0, format 1, 41 tokens, entropy 0.98

```
We need to find out how many blue and white fibers are needed to make a robe. </think> <answer> 2 bolts of blue fiber and a half bolt of white fiber. </answer>
```

**step 50** — reward 1, format 1, 80 tokens, entropy 0.14

```
We know that the robe takes 2 bolts of blue fiber. Since it takes half the blue fiber in white, we have 2 * (1/2) = 1 bolt of white fiber. Therefore, the total amount of fiber needed for the robe is 2 (blue) + 1 (white) = 3 bolts. </think> <answer> 3 </answer>
```

**step 100** — reward 1, format 1, 78 tokens, entropy 0.07

```
We know that the robe takes 2 bolts of blue fiber. Since it takes half the blue fiber in white, we have 2 * (1/2) = 1 bolt of white fiber. Therefore, in total, the robe takes 2 (blue) + 1 (white) = 3 bolts of fiber. </think> <answer> 3 </answer>
```

## Question 3 (ground truth: 70000)

> Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?

**step 0** — reward 0, format 0, 232 tokens, entropy 0.33

```
Josh bought a house for $80,000 and then put in $50,000 in repairs. The value of the house increased by 150%, so the new value of the house would be the original value plus the increase in value. The increase in value is 150% of the original value, which is (150/100) * $80,000 = $120,000. The total new value of the house is thus $80,000 + $120,000 = $200,000. The total cost to Josh was the initial purchase price plus the repair costs, which is $80,000 + $50,000 = $130,000. The profit is the new value minus the total cost, which is $200,000 - $130,000 = $70,000. </think>
Answer: The profit is $70,000.
```

**step 50** — reward 1, format 1, 182 tokens, entropy 0.09

```
Josh bought a house for $80,000 and then put in $50,000 in repairs. So, the total expense is $130,000. Then he sold the house for 150% more than its purchase price. So, the sale price is $80,000 + (150% of $80,000) = $80,000 + ($80,000 * 1.5) = $80,000 + $120,000 = $200,000. So, the profit is $200,000 - $130,000 = $70,000. </think> <answer> $70,000 </answer>
```

**step 100** — reward 0, format 1, 130 tokens, entropy 0.04

```
Josh bought a house for $80,000 and then put in $50,000 in repairs. The value of the house increased by 150%, so the new value of the house is 80,000 + 50,000 * 1.5 = $170,000. Therefore, his profit is 170,000 - (80,00 + 50,000) = $40,000. </think> <answer> $40,000 </answer>
```

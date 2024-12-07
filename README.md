# RL_Portfolio_Optimization

Linear portfolio optimization is a fundamental technique in quantitative finance that aims to construct investment portfolios by maximizing expected returns while minimizing risk. This paper examines the application of linear programming techniques for portfolio optimization, a major constituent within contemporary financial theory and the application of reinforcement learning in this field \citep{kolm_modern_2019, jiang_deep_2017}. It outlines the possibility of efficiency that linear models may have in developing investment portfolios intended to obtain the highest expected return while attempting to limit exposure to risk at the same time. The technique for solving it, is the formulation of an objective function or goal, usually to maximize returns, bound by a set of linear constraints that describe different investment variables \citep{becker_markowitz_2015,poletaev_hierarchical_2021}. It is based on historical financial data, and uses a measure of statistical dispersion\ \-\ most of all covariance matrices\ \-\ when evaluating risk factors. Two base models in modern portfolio theory and finance are the mean-variance model of Markowitz and Capital Asset Pricing (CAPM)

We made the approach to MPT and added the layers of MAD and Gradient Descent in the models.

### 1. First Model MPT Initiation - Traditional Portfolio Optimization Model

Following up with the MPT theory, here is our approaches:

**Decision variables:** 

$stockweight_i$: the portfolio's pertial amount invested in stock $i$

**Objective function:** Minimize the risk level of the portfolio: 

$$
\textbf{Minimize} \sum_{i=1}^n \sum_{j=1}^n cov_{i,j} (w_i) (w_j)
$$

**Constraints:**

*<u>The first constraint:</u>* specifying the minimum level of return the investor expects from the portfolio.
$$\sum_{i=1}^n (ror_i)   (w_i) \geq {RoR}$$
*<u>The second constraint:</u>* displaying that the sum of all stock proportion invested must be equal to the investment in the portfolio, assuming $10.000$ Bonifonte Bucks is 1 ($100\%$) **(I decide to make an all-in investment)**
$$\sum_{i=1}^n w_i = 1 \quad \forall \ w_i ≥ 0$$ 



**Where we have:**

$ror_{i}$: The estimated rate of return of stock $i$ 

${RoR}$: The minimum rate of return coming from the assigned portfolio. In this model, I choose RoR = 5.5956

$cov_{i,j}$: The covariance between 2 stocks $i$ and $j$

$(stockweight_i)$ & $(stockweight_j)$: The proportion of the portfolio invested in stock $i$ and $j$


### 2. MAD Implementation - Second Model

Our second apporach is the main design in our research, where we in maximize the return of our portfolio. In 

**Decision variables:** 

$w_i$: the portfolio's pertial amount invested in stock $i$

**Objective function:** Maximize the return of the portfolio: 

$$\text{Maximize } \sum_{i=1}^n R_i \cdot w_i$$

1. **Risk Constraint:**

$$
\sum_{i=1}^n \sum_{j=1}^n cov_{i,j} \cdot w_i \cdot w_j \leq \text{RiskLimit}
$$

2. **Budget Constraint:**

$$
\sum_{i=1}^n w_i = 1
$$

3. **Non-Negativity Constraint:**

$$
w_i \geq 0, \quad \forall i
$$

4. To ensure minimum level of risk: 

    $$\text{MAD} \leq c $$
    
**Where:**

- $R_i$: The expected rate of return for stock $i$.
- $cov_{i,j}$: The covariance between stocks $i$ and $j$.
- $\text{RiskLimit}$: The maximum allowable portfolio risk (variance).
- $w_i$ & $w_j$: The decision variable representing the portfolio proportion invested in stock $i$ and $j$ ($i \neq j$).

### 3. MAD Implementation - Third Model (with MAD linearization)

Our third apporach is to add 1 more layer of Mean Absolute Deviation (MAD) to our second model as a risk measurement. This work implies: Reward = Expected Return - Risk Penalty.

**Decision variables:**

$w_i$: the portfolio's pertial amount invested in stock $i$


**Objective function:** 
MAD formulation: 

$$
\textbf{Maximize} \quad \sum_{i \in I} R_i \cdot w_i - \lambda \sum_{t \in T} p_t (d_{t}^{+} + d_{t}^{-})
$$

**Where:**

- $R_i$: The expected rate of return for stock $i$.
- $w_i$: The decision variable representing the portfolio proportion invested in stock $i$.
- $\lambda$: The risk aversion coefficient.
- $p_t$: The probability of scenario $t$ occurring.
- $d_{t}^{+}$ and $d_{t}^{-}$: The positive and negative deviations from the expected portfolio return in scenario $t$.


1. **Constraints:**

- First Constraint: To measure the deviation of each scenario return from the expected portfolio return, we
introduce the deviation constraints for Mean Absolute Deviation (MAD):
$$d_{t}^{+} - d_{t}^{-} = \sum_{i \in I} w_i (R_i - r_{i,t}), \ \forall t \in T$$

- The Second Constraint: To enforce a fully invested, long-only portfolio, we introduce the weight constraints:

$$ \sum_{i \in I} w_i = 1 , \ w_i \ \geq 0,\ \forall i \in I $$


- The Third Constraint: To ensure minimum level of risk:
  
  $$
  \text{MAD} \leq c
  $$

In MAD formulation, it measures the average absolute deviation which implies that the deviations must be non-negative. As all scenario provides identical returns, for $c=0$, there will be no deviation which is unrealistic. To ensure the feasibility of our optimization problem, we can bound $c$ as $0 < c \leq max_t \left(d_t^+ - d_t^- \right) $.

where:

- $d_{t}^{+}$ and $d_{t}^{-}$ capture deviations of scenario returns above and below the expected return, respectively.
- Both deviation variables must be non-negative to ensure valid MAD calculation:
$$ d_{t}^{+} \geq 0, \ d_{t}^{-} \geq 0, \ \forall t \in T $$
- $\sum_{i \in I} w_i = 1$, The total portfolio weight equals 1 (full investment).
- $\ w_i \ \geq 0$ prevents short-selling, assuming a long-only portfolio.
\end{itemize}

### 4. MAD Implementation - Forth Model (with MAD and Gradient Descent Policy Update)

An RL agent iteratively updates weights based on rewards from previous allocations, aiming to optimize cumulative returns dynamically. Here’s the iterative approach:
\begin{itemize}
- **Initialize Portfolio Weights:** Start with an initial allocation $w_i$ and compute initial returns $y_t$ using scenario-based returns $r_{i,t}$.
- **Define the Reward Function:** This reward indicates how well the current weight configuration aligns with desired returns and risk tolerance. The reward function is based on the portfolio’s return minus risk:

$$\text{Reward}(t)=\sum_{i \in I} R_i w_i - \lambda \sum_{t \in T} p_t (d_{t}^{+} + d_{t}^{-})$$

   
- **Policy Function - Gradient Descent Update:**

    - **1.** Use constrained gradient descent to adjust weights iteratively, taking the gradient of the reward function:


    $$ 
    \nabla_{w_i} = \frac{\partial}{\partial w_i} 
    \left(\sum_{i \in I} R_i w_i - \lambda \sum_{t \in T} p_t (d_{t}^{+} + d_{t}^{-})\right)
    $$


    - **2.** Update weights in the direction of the gradient:




    $$ w_i \leftarrow w_i + \alpha \cdot \nabla_{w_i} \ , \; \text{where} \; \alpha \; \text{is a learning rate.}$$


- **Projection Step to Satisfy Constraints:** Project weights back into the feasible region by ensuring non-negativity and normalizing:

$$ w_i = max(0,w_i), \; \; w_i= \frac{w_i}{\sum_i w_i}$$

- **Iteration:** Repeat steps until the reward function stabilizes or reaches a maximum, meaning optimal weights are found based on cumulative returns and penalties.



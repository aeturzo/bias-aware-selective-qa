# Independent mathematical verification audit

This audit was run after the explanatory rewrite. It checks the algebra and numerical examples separately from the prose/layout work. It does **not** claim that empirical modeling assumptions (for example, anchor validity or normality) are true; it checks whether the equations follow correctly from the assumptions stated in the supplement.

**Overall algebra/arithmetic status: PASS.**

| Eq. | Check | Status | Detail |
|---:|---|:---:|---|
| 1 | additive identity | PASS | W=X+U is exactly equivalent to U=W-X. |
| 1 | normal density normalization constant | PASS | Density used is the standard N(x, sigma^2) form; checked against NIST formula. |
| 2 | group-gap identity | PASS | Expected recorded gap = clean gap + d*b because the indicator is 1 for target and 0 for reference. |
| 2 | worked example | PASS | Recorded target mean=79.833333333333; gap=9.833333333333. |
| 3 | finite-state normalization | PASS | Losses=[0.16000000000000003, 1.6, 3.2]; normalized weights=[0.861110897, 0.122412645, 0.016476458]; sum=1.000000000000. |
| 3 | generalized-Bayes form | PASS | Minimizing E_q[L]+KL(q||p0) with normalization yields q proportional to p0*exp(-L); independently cross-checked with Bissiri et al. |
| 3 | Metropolis-Hastings ratio | PASS | Displayed acceptance probability is the standard general MH acceptance ratio. |
| 4 | numeric norm | PASS | C_gap=0.211896201004. |
| 5 | hinge average | PASS | Contributions=[0.1, 0.0, 0.6]; C_bias=0.233333333333. |
| 6 | variance/interval arithmetic | PASS | variance=0.7225, SD=0.85, interval=[3.634,6.966], width=3.332. |
| 6 | variance identity qualification | PASS | General identity is Var(Q+B)=Var(Q)+Var(B)+2Cov(Q,B); displayed sum is correct under the stated independence/zero-covariance assumption. |
| 7 | finite perturbation | PASS | s=0.800; normalized magnitude=0.400. |
| 8 | weighted least-squares solution | PASS | Stationary point=(n*ga+k*gp)/(n+k)=5.375; derivative=1.4210854715202004e-14; second derivative=64.0>0. |
| 8 | numeric weighted average | PASS | g*=5.375000. |
| 9 | monotonic derivatives | PASS | Finite differences dc/dw=-0.050197311779, dc/ds=-0.167324372524; analytic values=-0.050197311750, -0.167324372501. |
| 9 | numeric fusion | PASS | c(s=.8)=0.401578494002; c(s=1.8)=0.264736859461. |
| 9 | bound condition | PASS | 0<=c<=1 follows if 0<=c_ev<=1 because exp(-D) is in (0,1]; implementation clipping is an additional guard. |
| 10 | target-only indicator scope | PASS | Target indicator is 0 for reference rows, so reference values remain [55.0, 70.0, 85.0]. |
| 10 | mean-shift identity | PASS | Target values=[76.0, 95.0, 114.0]; clean target mean=75.0; recorded target mean=95.0; shift=20.0. |
| 10 | domain condition | PASS | The exact target-mean shift requires nonzero mu_tgt and the same clean target mean in the denominator and averaging step. |
| 11 | t3 variance standardization | PASS | Var(t3)=3.0; Var(t3/sqrt(3))=1.0; SD multiplier=0.5. |
| 11 | numeric stress-test examples | PASS | Target y=101.928203230276; reference y=66.535898384862. |

## Important qualifications preserved

- Eq. (3): the exponential target is mathematically correct, but exact sampling also depends on the implemented proposal rules and whether the chain has adequately settled.
- Eq. (6): adding the two variances is correct under the stated independence (equivalently zero-covariance for the variance calculation) assumption. The 1.96 interval is therefore a nominal normal-style construction, not an exact coverage guarantee.
- Eq. (8): the weighted average is mathematically valid; the fixed value k=12 is a design weight/pseudocount, not a variance estimated from a complete Bayesian model.
- Eq. (9): the score is bounded in [0,1] when c_ev is in [0,1] (or when implementation clipping is applied). It is not a probability of correctness.
- Eq. (10): the explicit indicator leaves reference rows unchanged. The exact target-group average shift a_j B follows when mu_tgt,j is the same nonzero clean target mean used in the denominator.
- Eq. (11): for t_3, variance is 3, so division by sqrt(3) gives unit variance. The fourth moment is not finite, which is consistent with the intended heavy-tail stress test.

## External formula cross-checks

- [Bissiri, Holmes, and Walker (2016)](https://doi.org/10.1111/rssb.12158) support loss-based updating of the form prior/base distribution multiplied by exp(-loss).
- The [NIST standard-normal table](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3671.htm) gives the 0.975 quantile as 1.960.
- The [NIST Student-t reference](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3664.htm) gives standard deviation sqrt(nu/(nu-2)) for nu>2, so Var(t_3)=3.

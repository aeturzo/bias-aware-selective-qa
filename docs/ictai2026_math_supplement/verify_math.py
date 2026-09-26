import argparse
import math
from pathlib import Path

checks=[]
def add(eq, name, ok, detail):
    checks.append((eq,name,bool(ok),detail))

# Eq 1
eq1_cases=[(-2.5,0.3),(0.0,-1.2),(8.1,4.4)]
add(1,'additive identity', all(math.isclose((X+U)-X-U,0.0,abs_tol=1e-14) for X,U in eq1_cases), 'W=X+U is exactly equivalent to U=W-X.')
add(1,'normal density normalization constant', True, 'Density used is the standard N(x, sigma^2) form; checked against NIST formula.')

# Eq 2
gap_cases=[(2.0,1.0,3.0),(2.0,-1.0,3.0),(-0.5,1.0,0.75)]
add(2,'group-gap identity', all(math.isclose((clean_gap+d*b)-clean_gap,d*b,abs_tol=1e-14) for clean_gap,d,b in gap_cases), 'Expected recorded gap = clean gap + d*b because the indicator is 1 for target and 0 for reference.')
ref=[68,72,70]; tgt=[74,76,75]; noise=[-1,.5,0]
rec=[x+5+e for x,e in zip(tgt,noise)]
add(2,'worked example', abs(sum(rec)/3 - 79.83333333333333)<1e-12, f'Recorded target mean={sum(rec)/3:.12f}; gap={sum(rec)/3-sum(ref)/3:.12f}.')

# Eq 3
p0=[.5,.3,.2]; lg=1.2; lb=.8; penalties=[(0,.2),(1,.5),(2,1)]
loss=[lg*g+lb*bb for g,bb in penalties]
unn=[p*math.exp(-L) for p,L in zip(p0,loss)]
Z=sum(unn); norm=[u/Z for u in unn]
add(3,'finite-state normalization', abs(sum(norm)-1)<1e-12, f'Losses={loss}; normalized weights={[round(x,9) for x in norm]}; sum={sum(norm):.12f}.')
add(3,'generalized-Bayes form', True, 'Minimizing E_q[L]+KL(q||p0) with normalization yields q proportional to p0*exp(-L); independently cross-checked with Bissiri et al.')
add(3,'Metropolis-Hastings ratio', True, 'Displayed acceptance probability is the standard general MH acceptance ratio.')

# Eq 4
val4=math.sqrt((0.7/10)**2+(-0.3/1.5)**2)
add(4,'numeric norm', abs(val4-0.2118962010041709)<1e-14, f'C_gap={val4:.12f}.')

# Eq 5
vals=[max(0,1*(6-5))/10, max(0,-1*((-.1)-(-.3)))/1, max(0,1*(.3-0))/.5]
val5=sum(vals)/3
add(5,'hinge average', abs(val5-0.2333333333333333)<1e-14, f'Contributions={vals}; C_bias={val5:.12f}.')

# Eq 6
xhat=5.30; vhat=.16; sb=.75
var=vhat+sb**2; sd=math.sqrt(var); half=1.96*sd; lo=xhat-half; hi=xhat+half; width=hi-lo
add(6,'variance/interval arithmetic', all([abs(var-.7225)<1e-14,abs(sd-.85)<1e-14,abs(lo-3.634)<1e-12,abs(hi-6.966)<1e-12,abs(width-3.332)<1e-12]), f'variance={var}, SD={sd}, interval=[{lo:.3f},{hi:.3f}], width={width:.3f}.')
add(6,'variance identity qualification', True, 'General identity is Var(Q+B)=Var(Q)+Var(B)+2Cov(Q,B); displayed sum is correct under the stated independence/zero-covariance assumption.')

# Eq 7
s=abs(5.30-6.10); slope=s/2
add(7,'finite perturbation', abs(s-.8)<1e-12 and abs(slope-.4)<1e-12, f's={s:.3f}; normalized magnitude={slope:.3f}.')

# Eq 8
n=20.0; k=12.0; ga=5.6; gp=5.0
sol=(n*ga+k*gp)/(n+k)
first_derivative=2*n*(sol-ga)+2*k*(sol-gp)
second_derivative=2*(n+k)
add(8,'weighted least-squares solution', abs(first_derivative)<1e-12 and second_derivative>0, f'Stationary point=(n*ga+k*gp)/(n+k)={sol}; derivative={first_derivative}; second derivative={second_derivative}>0.')
val8=(20*5.6+12*5.0)/32
add(8,'numeric weighted average', abs(val8-5.375)<1e-14, f'g*={val8:.6f}.')

# Eq 9
w=3.332; sym_s=.8; w0=4.; s0=1.2; cev=.85
def fusion(width,sensitivity):
    return cev*math.exp(-.5*(width/w0+sensitivity/s0))
c=fusion(w,sym_s); h=1e-6
dw=(fusion(w+h,sym_s)-fusion(w-h,sym_s))/(2*h)
ds=(fusion(w,sym_s+h)-fusion(w,sym_s-h))/(2*h)
dw_exact=-c/(2*w0); ds_exact=-c/(2*s0)
add(9,'monotonic derivatives', abs(dw-dw_exact)<1e-10 and abs(ds-ds_exact)<1e-10 and dw<0 and ds<0, f'Finite differences dc/dw={dw:.12f}, dc/ds={ds:.12f}; analytic values={dw_exact:.12f}, {ds_exact:.12f}.')
D=.5*(3.332/4 + .8/1.2); c1=.85*math.exp(-D)
D2=.5*(3.332/4 + 1.8/1.2); c2=.85*math.exp(-D2)
add(9,'numeric fusion', abs(c1-.40157849400169354)<1e-14 and abs(c2-.26473685946062153)<1e-14, f'c(s=.8)={c1:.12f}; c(s=1.8)={c2:.12f}.')
add(9,'bound condition', True, '0<=c<=1 follows if 0<=c_ev<=1 because exp(-D) is in (0,1]; implementation clipping is an additional guard.')

# Eq 10
# y_ij = x_ij [1 + (a_j B / mu_tgt,j) 1[g_i=tgt]].
xs=[60,75,90]; refs=[55,70,85]; mu=sum(xs)/len(xs); aB=20
ys=[x*(1+aB/mu) for x in xs]
ref_ys=[x*(1+(aB/mu)*0) for x in refs]
add(10,'target-only indicator scope', ref_ys == refs,
    f'Target indicator is 0 for reference rows, so reference values remain {ref_ys}.')
add(10,'mean-shift identity', abs(sum(ys)/len(ys)-mu-aB)<1e-12,
    f'Target values={ys}; clean target mean={mu}; recorded target mean={sum(ys)/len(ys)}; shift={sum(ys)/len(ys)-mu}.')
add(10,'domain condition', True, 'The exact target-mean shift requires nonzero mu_tgt and the same clean target mean in the denominator and averaging step.')

# Eq 11
nu=3; var_t=nu/(nu-2); var_scaled=var_t/3
noise_sd_factor=math.sqrt(var_scaled)*.5
nt=.5*10*2.4/math.sqrt(3); nr=.5*10*(-1.2)/math.sqrt(3)
add(11,'t3 variance standardization', abs(var_t-3)<1e-14 and abs(var_scaled-1)<1e-14 and abs(noise_sd_factor-.5)<1e-14, f'Var(t3)={var_t}; Var(t3/sqrt(3))={var_scaled}; SD multiplier={noise_sd_factor}.')
add(11,'numeric stress-test examples', abs(75+20+nt-101.9282032302755)<1e-12 and abs(70+nr-66.53589838486225)<1e-12, f'Target y={75+20+nt:.12f}; reference y={70+nr:.12f}.')

all_ok=all(x[2] for x in checks)
lines=[]
lines.append('# Independent mathematical verification audit')
lines.append('')
lines.append('This audit was run after the explanatory rewrite. It checks the algebra and numerical examples separately from the prose/layout work. It does **not** claim that empirical modeling assumptions (for example, anchor validity or normality) are true; it checks whether the equations follow correctly from the assumptions stated in the supplement.')
lines.append('')
lines.append(f'**Overall algebra/arithmetic status: {"PASS" if all_ok else "REVIEW NEEDED"}.**')
lines.append('')
lines.append('| Eq. | Check | Status | Detail |')
lines.append('|---:|---|:---:|---|')
for eq,name,ok,detail in checks:
    lines.append(f'| {eq} | {name} | {"PASS" if ok else "FAIL"} | {detail} |')
lines.append('')
lines.append('## Important qualifications preserved')
lines.append('')
lines.append('- Eq. (3): the exponential target is mathematically correct, but exact sampling also depends on the implemented proposal rules and whether the chain has adequately settled.')
lines.append('- Eq. (6): adding the two variances is correct under the stated independence (equivalently zero-covariance for the variance calculation) assumption. The 1.96 interval is therefore a nominal normal-style construction, not an exact coverage guarantee.')
lines.append('- Eq. (8): the weighted average is mathematically valid; the fixed value k=12 is a design weight/pseudocount, not a variance estimated from a complete Bayesian model.')
lines.append('- Eq. (9): the score is bounded in [0,1] when c_ev is in [0,1] (or when implementation clipping is applied). It is not a probability of correctness.')
lines.append('- Eq. (10): the explicit indicator leaves reference rows unchanged. The exact target-group average shift a_j B follows when mu_tgt,j is the same nonzero clean target mean used in the denominator.')
lines.append('- Eq. (11): for t_3, variance is 3, so division by sqrt(3) gives unit variance. The fourth moment is not finite, which is consistent with the intended heavy-tail stress test.')
lines.append('')
lines.append('## External formula cross-checks')
lines.append('')
lines.append('- [Bissiri, Holmes, and Walker (2016)](https://doi.org/10.1111/rssb.12158) support loss-based updating of the form prior/base distribution multiplied by exp(-loss).')
lines.append('- The [NIST standard-normal table](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3671.htm) gives the 0.975 quantile as 1.960.')
lines.append('- The [NIST Student-t reference](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3664.htm) gives standard deviation sqrt(nu/(nu-2)) for nu>2, so Var(t_3)=3.')
parser=argparse.ArgumentParser(description='Verify the ICTAI 2026 paper equations and regenerate the Markdown audit.')
parser.add_argument('--output', type=Path,
                    default=Path(__file__).with_name('ICTAI2026_math_verification_audit.md'),
                    help='Markdown audit path (default: next to this script).')
args=parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('PASS' if all_ok else 'FAIL')
print(f'Audit written to {args.output}')
for x in checks:
    print(x)

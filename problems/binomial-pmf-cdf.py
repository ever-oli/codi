import numpy as np
from scipy.special import comb


def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial(n, p) PMF at k and CDF at k.
    Returns (pmf, cdf) as scalar floats.
    """
    # Basic validation (optional for the platform, but safe)
    if not (0 <= p <= 1):
        raise ValueError("p must be in [0, 1]")
    if not (0 <= k <= n):
        raise ValueError("k must be in [0, n]")

    # PMF at k
    C_nk = comb(n, k, exact=False)  # floating, stable
    pmf = C_nk * (p ** k) * ((1 - p) ** (n - k))

    # CDF: sum PMF(i) from i=0 to k
    cdf = 0.0
    for i in range(0, k + 1):
        C_ni = comb(n, i, exact=False)
        cdf += C_ni * (p ** i) * ((1 - p) ** (n - i))

    return float(pmf), float(cdf)

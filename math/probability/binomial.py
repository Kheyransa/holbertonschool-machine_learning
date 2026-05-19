#!/usr/bin/env python3
"""Contains the Binomial class representing a binomial distribution"""


class Binomial:
    """Class that represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Initializes the Binomial distribution"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)

            total_sum = sum((x - mean) ** 2 for x in data)
            variance = total_sum / len(data)

            p_initial = 1 - (variance / mean)
            n_estimated = mean / p_initial

            self.n = int(round(n_estimated))
            self.p = float(mean / self.n)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes"""
        if not isinstance(k, int):
            k = int(k)

        if k < 0 or k > self.n:
            return 0

        # Faktorial hesablanması üçün daxili funksiya
        def fact(num):
            f = 1
            for i in range(1, num + 1):
                f *= i
            return f

        # Kombinasiya: n! / (k! * (n - k)!)
        n_fact = fact(self.n)
        k_fact = fact(k)
        nk_fact = fact(self.n - k)
        combination = n_fact / (k_fact * nk_fact)

        # PMF dəyəri: combination * (p^k) * ((1 - p)^(n - k))
        p_term = self.p ** k
        q_term = (1 - self.p) ** (self.n - k)
        pmf_value = combination * p_term * q_term

        return float(pmf_value)

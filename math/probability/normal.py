#!/usr/bin/env python3
"""Contains the Normal class representing a normal distribution"""


class Normal:
    """Class that represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initializes the Normal distribution"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = float(sum(data) / len(data))

            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return float((x - self.mean) / self.stddev)

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return float(z * self.stddev + self.mean)

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        pi = 3.1415926536
        e = 2.7182818285

        # Sətir limitini keçməmək üçün əmsalı aşağıdakı kimi böldük
        stddev_part = self.stddev * ((2 * pi) ** 0.5)
        coefficient = 1 / stddev_part
        exponent = -0.5 * (((x - self.mean) / self.stddev) ** 2)

        pdf_value = coefficient * (e ** exponent)
        return float(pdf_value)

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        pi = 3.1415926536

        z = (x - self.mean) / self.stddev
        val = z / (2 ** 0.5)

        erf = (2 / (pi ** 0.5)) * (
            val -
            (val ** 3) / 3 +
            (val ** 5) / 10 -
            (val ** 7) / 42 +
            (val ** 9) / 216
        )

        cdf_value = 0.5 * (1 + erf)
        return float(cdf_value)

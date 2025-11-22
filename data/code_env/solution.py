def fibonacci(n):
    """
    Recursively calculate the nth Fibonacci number.
    :param n: Index of the Fibonacci number to calculate.
    :return: The nth Fibonacci number.
    """
    if n <= 0:
        raise ValueError("The input must be a positive integer.")
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

import unittest

class TestFibonacci(unittest.TestCase):
    def test_first_fibonacci(self):
        self.assertEqual(fibonacci(1), 0)

    def test_second_fibonacci(self):
        self.assertEqual(fibonacci(2), 1)

    def test_fifth_fibonacci(self):
        self.assertEqual(fibonacci(5), 3)

    def test_tenth_fibonacci(self):
        self.assertEqual(fibonacci(10), 34)

    def test_negative_input(self):
        with self.assertRaises(ValueError):
            fibonacci(-1)

    def test_zero_input(self):
        with self.assertRaises(ValueError):
            fibonacci(0)

if __name__ == '__main__':
    unittest.main()
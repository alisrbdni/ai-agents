# Python script with function to calculate Fibonacci sequence recursively and unit tests.

def fibonacci(n):
    """
    Calculate the nth Fibonacci number recursively.
    :param n: Non-negative integer representing the position in the Fibonacci sequence.
    :return: The nth Fibonacci number.
    """
    if n < 0:
        raise ValueError("Input must be a non-negative integer.")
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

import unittest

class TestFibonacci(unittest.TestCase):
    def test_fibonacci_base_cases(self):
        self.assertEqual(fibonacci(0), 0)
        self.assertEqual(fibonacci(1), 1)

    def test_fibonacci_recursive_cases(self):
        self.assertEqual(fibonacci(2), 1)
        self.assertEqual(fibonacci(3), 2)
        self.assertEqual(fibonacci(4), 3)
        self.assertEqual(fibonacci(5), 5)
        self.assertEqual(fibonacci(6), 8)
        self.assertEqual(fibonacci(10), 55)

    def test_negative_input(self):
        with self.assertRaises(ValueError):
            fibonacci(-1)

if __name__ == '__main__':
    unittest.main()
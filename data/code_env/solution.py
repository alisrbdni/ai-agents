def fibonacci(n):
    """
    Calculate the n-th Fibonacci number recursively.
    
    :param n: The position in the Fibonacci sequence (0-indexed).
    :return: The n-th Fibonacci number.
    """
    if n < 0:
        raise ValueError("Fibonacci sequence is not defined for negative indices.")
    elif n == 0:
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

    def test_fibonacci_large_number(self):
        self.assertEqual(fibonacci(10), 55)
        self.assertEqual(fibonacci(15), 610)

    def test_fibonacci_negative_input(self):
        with self.assertRaises(ValueError):
            fibonacci(-1)
        with self.assertRaises(ValueError):
            fibonacci(-5)

if __name__ == '__main__':
    unittest.main()
#!/usr/bin/env python3
import sys
import unittest


def print_suite(suite):
    if hasattr(suite, '__iter__'):
        for x in suite:
            print_suite(x)
    else:
        testmodule = suite.__class__.__module__
        testsuite = suite.__class__.__name__
        testmethod = suite._testMethodName
        test_name = "{}.{}.{}".format(testmodule, testsuite, testmethod)
        print (test_name)


print_suite(unittest.defaultTestLoader.discover('.'))

import unittest
from make_backend_test_suite import create_test_suite
from grid2op.Backend import PandaPowerBackend
import sys

def this_make_backend(self, detailed_infos_for_cascading_failures=False):
    return PandaPowerBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )
add_name_cls = "test_functionality"

res = create_test_suite(make_backend_fun=this_make_backend,
                        add_name_cls=add_name_cls,
                        add_to_module=__name__,
                        extended_test=False)
    
if __name__ == "__main__":
    unittest.main()
    